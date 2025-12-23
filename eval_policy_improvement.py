# eval_policy_improvement.py
# Pit latest checkpoint policy vs an older snapshot policy on a LOCAL Showdown server.
#
# Example:
#   python eval_policy_improvement.py --ckpt_dir checkpoints --old_ckpt checkpoints/learner_update_000250.pt \
#       --host localhost --port 8000 --format gen9randombattle --battles 50 --device cpu
#
# Notes:
# - Uses two separate poke-env Players connected to your local server.
# - Runs deterministic greedy actions (greedy=True).
# - Logs per-turn + battle_end JSONL for each side.

from __future__ import annotations

import argparse
import glob
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Player

from config import RunConfig
from showdown_obs import build_tokens  # must match training
from ppo_core import masked_sample


# -------------------------
# Checkpoint helpers
# -------------------------

def list_ckpts(ckpt_dir: str) -> list[str]:
    pats = [
        os.path.join(ckpt_dir, "learner_update_*.pt"),
        os.path.join(ckpt_dir, "*.pt"),
    ]
    paths: list[str] = []
    for p in pats:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))
    return paths

def latest_ckpt_path(ckpt_dir: str) -> str:
    paths = list_ckpts(ckpt_dir)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir!r}")
    return paths[-1]

def _extract_state_dict(ckpt: dict) -> dict:
    """
    Try a few common save formats:
      - {"model": state_dict, ...}
      - {"state_dict": state_dict, ...}
      - raw state_dict
    """
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "net", "policy", "actor_critic"):
            v = ckpt.get(k, None)
            if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
                return v
        if any(isinstance(x, torch.Tensor) for x in ckpt.values()):
            return ckpt
    raise RuntimeError("Could not find a model state_dict inside the checkpoint.")

def make_server_conf(host: str, port: int) -> ServerConfiguration:
    ws_url = f"ws://{host}:{port}/showdown/websocket"
    http_action = f"http://{host}:{port}/action.php?"
    return ServerConfiguration(ws_url, http_action)


# -------------------------
# Loaded policy
# -------------------------

@dataclass
class LoadedPolicy:
    cfg: RunConfig
    device: str
    net: torch.nn.Module
    act_dim: int

    @staticmethod
    def load(ckpt_path: str, cfg: RunConfig, device: str) -> "LoadedPolicy":
        dev = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = _extract_state_dict(ckpt)

        net = cfg.make_model().to(dev).eval()
        net.load_state_dict(state, strict=True)

        torch.set_grad_enabled(False)
        return LoadedPolicy(cfg=cfg, device=str(dev), net=net, act_dim=int(cfg.env.act_dim))

    def legal_action_mask_and_map(self, battle):
        mask = np.zeros((self.act_dim,), dtype=np.float32)
        idx_to_action: Dict[int, tuple] = {}

        moves = list(battle.available_moves)
        switches = list(battle.available_switches)

        can_tera = bool(getattr(battle, "can_tera", False))

        # ---- Moves ----
        for i in range(4):
            if i < len(moves):
                # normal move
                mask[i] = 1.0
                idx_to_action[i] = ("move", i, False)

                # tera move
                tera_idx = 4 + i
                if can_tera and tera_idx < self.act_dim:
                    mask[tera_idx] = 1.0
                    idx_to_action[tera_idx] = ("move", i, True)

        # ---- Switches ----
        base = 8
        for j in range(6):
            a = base + j
            if a < self.act_dim and j < len(switches):
                mask[a] = 1.0
                idx_to_action[a] = ("switch", j)

        return mask, idx_to_action, moves, switches

    @torch.no_grad()
    def act_from_battle(self, battle) -> Tuple[int, float, float, np.ndarray, Dict[int, tuple], list, list]:
        """
        Returns: (action_idx, logp, value, amask_np, idx_to_action, moves, switches)
        """
        obs = self.cfg.obs
        tb = build_tokens(battle, obs)  # padded to t_max

        amask, idx_to_action, moves, switches = self.legal_action_mask_and_map(battle)

        # safety: ensure at least one legal action
        if float(amask.sum()) <= 0.0:
            amask[0] = 1.0
            idx_to_action[0] = ("move", 0, False)

        # add batch dim = 1
        ff   = torch.from_numpy(tb.float_feats[None, ...]).to(self.device, dtype=torch.float32)
        tt   = torch.from_numpy(tb.tok_type[None, ...]).to(self.device, dtype=torch.long)
        own  = torch.from_numpy(tb.owner[None, ...]).to(self.device, dtype=torch.long)
        pos  = torch.from_numpy(tb.pos[None, ...]).to(self.device, dtype=torch.long)
        sub  = torch.from_numpy(tb.subpos[None, ...]).to(self.device, dtype=torch.long)
        eid  = torch.from_numpy(tb.entity_id[None, ...]).to(self.device, dtype=torch.long)
        tmsk = torch.from_numpy(tb.attn_mask[None, ...]).to(self.device, dtype=torch.float32)
        am   = torch.from_numpy(amask[None, ...]).to(self.device, dtype=torch.float32)

        logits, value = self.net(ff, tt, own, pos, sub, eid, tmsk)  # [1,A], [1]
        a_t, logp_t, _ent = masked_sample(logits, am, greedy=True)

        a_idx = int(a_t.item())
        logp = float(logp_t.item())
        v = float(value.item())
        return a_idx, logp, v, amask, idx_to_action, moves, switches


# -------------------------
# Player wrapper with logging
# -------------------------

class PolicyPlayer(Player):
    def __init__(self, *, policy: LoadedPolicy, log_path: str, tag: str, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy
        self.tag = str(tag)

        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.log_path = log_path
        self._fh = open(log_path, "a", encoding="utf-8")

        self.battles_done = 0
        self.wins = 0
        self.losses = 0

    def close(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def _write(self, obj: Dict[str, Any]):
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()

    def _name(self, x) -> str:
        try:
            return getattr(x, "id", None) or getattr(x, "species", None) or str(x)
        except Exception:
            return str(x)

    async def choose_move(self, battle):
        try:
            a_idx, logp, value, amask_np, idx_to_action, moves, switches = self.policy.act_from_battle(battle)
        except Exception as e:
            self._write({
                "type": "warn",
                "ts": time.time(),
                "tag": self.tag,
                "msg": "policy act failed; using random move",
                "err": repr(e),
            })
            return self.choose_random_move(battle)

        action = idx_to_action.get(int(a_idx), None)
        if action is None:
            self._write({
                "type": "warn",
                "ts": time.time(),
                "tag": self.tag,
                "msg": "policy chose illegal/unmapped action; using random move",
                "a_idx": int(a_idx),
                "legal_mask": amask_np.tolist(),
            })
            return self.choose_random_move(battle)

        turn = getattr(battle, "turn", None)
        self._write({
            "type": "move",
            "ts": time.time(),
            "tag": self.tag,
            "turn": int(turn) if isinstance(turn, int) else turn,
            "a_idx": int(a_idx),
            "logp": float(logp),
            "value": float(value),
            "legal_mask": amask_np.tolist(),
            "available_moves": [self._name(m) for m in moves],
            "available_switches": [self._name(s) for s in switches],
            "chosen_tuple": action,
        })

        kind = action[0]
        if kind == "move":
            _, move_slot, use_tera = action
            if move_slot >= len(moves):
                return self.choose_random_move(battle)
            return self.create_order(moves[move_slot], terastallize=bool(use_tera))

        if kind == "switch":
            _, switch_slot = action
            if switch_slot >= len(switches):
                return self.choose_random_move(battle)
            return self.create_order(switches[switch_slot])

        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        won = bool(getattr(battle, "won", False))
        self.battles_done += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1

        # Try to capture opponent name if available
        opp = None
        try:
            opp = getattr(battle, "opponent_username", None) or getattr(battle, "opponent", None)
            opp = str(opp) if opp is not None else None
        except Exception:
            opp = None

        self._write({
            "type": "battle_end",
            "ts": time.time(),
            "tag": self.tag,
            "won": bool(won),
            "wins": int(self.wins),
            "losses": int(self.losses),
            "battles_done": int(self.battles_done),
            "opponent": opp,
        })
        return super()._battle_finished_callback(battle)


# -------------------------
# Main
# -------------------------

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints", help="Checkpoint directory (for --latest_ckpt)")
    ap.add_argument("--latest_ckpt", default="", help="Path to latest checkpoint .pt (blank => autoload latest from --ckpt_dir)")
    ap.add_argument("--old_ckpt", required=True, help="Path to OLD checkpoint .pt (snapshot opponent)")

    ap.add_argument("--host", default="localhost", help="Local showdown host")
    ap.add_argument("--port", type=int, default=8000, help="Local showdown port")
    ap.add_argument("--format", default="gen9randombattle", help="Showdown format")
    ap.add_argument("--battles", type=int, default=50, help="How many head-to-head battles to play")

    ap.add_argument("--device", default="cpu", help="cpu or cuda (single device for both nets)")
    ap.add_argument("--log_dir", default="eval_logs_old", help="Directory for JSONL logs")
    ap.add_argument("--max_concurrent_battles", type=int, default=1)

    # Optional: fixed usernames (else random)
    ap.add_argument("--latest_user", default="", help="Username for latest policy (blank => random)")
    ap.add_argument("--old_user", default="", help="Username for old policy (blank => random)")

    args = ap.parse_args()

    cfg = RunConfig.default()

    latest_path = args.latest_ckpt.strip()
    if not latest_path:
        latest_path = latest_ckpt_path(args.ckpt_dir)
        print(f"[eval_old] autoload latest ckpt: {latest_path}")

    old_path = args.old_ckpt.strip()
    if not os.path.exists(old_path):
        raise FileNotFoundError(f"--old_ckpt not found: {old_path}")

    server_conf = make_server_conf(args.host, int(args.port))

    # Load policies
    latest_pol = LoadedPolicy.load(latest_path, cfg=cfg, device=args.device)
    old_pol = LoadedPolicy.load(old_path, cfg=cfg, device=args.device)

    # Accounts (local server: no password needed)
    run_tag = secrets.token_hex(3)
    latest_user = args.latest_user.strip() or f"eval_latest_{run_tag}"
    old_user = args.old_user.strip() or f"eval_old_{run_tag}"

    os.makedirs(args.log_dir, exist_ok=True)
    latest_log = os.path.join(args.log_dir, f"latest_vs_old_latest.jsonl")
    old_log = os.path.join(args.log_dir, f"latest_vs_old_old.jsonl")

    p_latest = PolicyPlayer(
        policy=latest_pol,
        log_path=latest_log,
        tag="latest",
        battle_format=args.format,
        max_concurrent_battles=int(args.max_concurrent_battles),
        server_configuration=server_conf,
        account_configuration=AccountConfiguration(latest_user, None),
        log_level=30,
        start_listening=True,
        open_timeout=30.0,
    )

    p_old = PolicyPlayer(
        policy=old_pol,
        log_path=old_log,
        tag="old",
        battle_format=args.format,
        max_concurrent_battles=int(args.max_concurrent_battles),
        server_configuration=server_conf,
        account_configuration=AccountConfiguration(old_user, None),
        log_level=30,
        start_listening=True,
        open_timeout=30.0,
    )

    try:
        # Head-to-head evaluation
        # poke-env API supports:
        #   await player.battle_against(opponent, n_battles)
        # If your poke-env version differs, switch to `await p_latest.battle_against(p_old, n_battles=args.battles)`
        t0 = time.time()
        await p_latest.battle_against(p_old, n_battles=int(args.battles))
        dt = time.time() - t0

        print(f"[eval_old] done in {dt:.1f}s")
        print(f"[eval_old] latest_ckpt: {latest_path}")
        print(f"[eval_old] old_ckpt:    {old_path}")
        print(f"[eval_old] battles: {p_latest.battles_done}")
        print(f"[eval_old] latest: wins={p_latest.wins} losses={p_latest.losses}")
        print(f"[eval_old] old:    wins={p_old.wins} losses={p_old.losses}")
        print(f"[eval_old] logs:\n  latest -> {latest_log}\n  old    -> {old_log}")

    finally:
        p_latest.close()
        p_old.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
