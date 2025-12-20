# eval.py
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import Player

from config import RunConfig
from showdown_obs import build_tokens  # must match training (TokenBatch w/ ff, tt, own, pos, subpos, eid, attn_mask)
from ppo_core import masked_sample     


def latest_ckpt_path(ckpt_dir: str) -> str:
    # supports your common patterns; add more if needed
    pats = [
        os.path.join(ckpt_dir, "learner_update_*.pt"),
        os.path.join(ckpt_dir, "*.pt"),
    ]
    paths = []
    for p in pats:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))
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
        # maybe it *is* the state_dict already
        if any(isinstance(x, torch.Tensor) for x in ckpt.values()):
            return ckpt
    raise RuntimeError("Could not find a model state_dict inside the checkpoint.")


@dataclass
class LoadedPolicy:
    cfg: RunConfig
    device: str
    net: torch.nn.Module

    @staticmethod
    def load(ckpt_path: str, cfg: RunConfig, device: str) -> "LoadedPolicy":
        dev = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = _extract_state_dict(ckpt)

        net = cfg.make_model().to(dev).eval()
        net.load_state_dict(state, strict=True)

        torch.set_grad_enabled(False)
        return LoadedPolicy(cfg=cfg, device=str(dev), net=net)

    @torch.no_grad()
    def act_from_battle(self, battle) -> Tuple[int, float, float, np.ndarray]:
        """
        Returns: (action_idx, logp, value, amask_np)
        """
        obs = self.cfg.obs

        tb = build_tokens(battle, obs)  # must match training schema; padded to t_max already

        # action mask (same mapping as training)
        amask = np.zeros((self.cfg.env.act_dim,), dtype=np.float32)
        moves = list(battle.available_moves)[:4]
        switches = list(battle.available_switches)[:6]
        for i, _mv in enumerate(moves):
            if i >= self.cfg.env.act_dim:
                break
            amask[i] = 1.0
        for j, _sw in enumerate(switches):
            a = 4 + j
            if a >= self.cfg.env.act_dim:
                break
            amask[a] = 1.0

        # safety: if showdown gives a weird state
        if amask.sum() <= 0:
            amask[0] = 1.0

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
        return a_idx, logp, v, amask


class EvalPlayer(Player):
    def __init__(self, policy: LoadedPolicy, log_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.log_path = log_path

        self.battles_done = 0
        self.wins = 0
        self.losses = 0

        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8")

    def close(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def _write(self, obj: Dict[str, Any]):
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()

    def _legal_action_map(self, battle) -> Tuple[np.ndarray, Dict[int, Any], list, list]:
        """
        Mirrors training mapping:
          0..3  = moves (up to 4)
          4..9  = switches (up to 6)   (or up to cfg.env.act_dim)
        """
        act_dim = int(self.policy.cfg.env.act_dim)
        mask = np.zeros((act_dim,), dtype=np.float32)
        idx_to_action: Dict[int, Any] = {}

        moves = list(battle.available_moves)[:4]
        switches = list(battle.available_switches)[:6]

        for i, mv in enumerate(moves):
            if i >= act_dim:
                break
            mask[i] = 1.0
            idx_to_action[i] = mv

        for j, sw in enumerate(switches):
            a = 4 + j
            if a >= act_dim:
                break
            mask[a] = 1.0
            idx_to_action[a] = sw

        return mask, idx_to_action, moves, switches

    async def choose_move(self, battle):
        # build mapping so we can translate action index -> move/switch object
        amask_np, idx_to_action, moves, switches = self._legal_action_map(battle)

        # act
        try:
            a_idx, logp, value, _amask_used = self.policy.act_from_battle(battle)
        except Exception as e:
            self._write({
                "type": "warn",
                "ts": time.time(),
                "msg": "policy act failed; using random move",
                "err": repr(e),
            })
            return self.choose_random_move(battle)

        # if policy picked something illegal under current state, fallback
        action_obj = idx_to_action.get(int(a_idx), None)
        if action_obj is None:
            self._write({
                "type": "warn",
                "ts": time.time(),
                "msg": "policy chose illegal/unmapped action; using random move",
                "a_idx": int(a_idx),
                "legal_mask": amask_np.tolist(),
            })
            return self.choose_random_move(battle)

        def _name(x):
            try:
                return getattr(x, "id", None) or getattr(x, "species", None) or str(x)
            except Exception:
                return str(x)

        turn = getattr(battle, "turn", None)
        self._write({
            "type": "move",
            "ts": time.time(),
            "turn": int(turn) if isinstance(turn, int) else turn,
            "a_idx": int(a_idx),
            "logp": float(logp),
            "value": float(value),
            "legal_mask": amask_np.tolist(),
            "available_moves": [_name(m) for m in moves],
            "available_switches": [_name(s) for s in switches],
            "chosen": _name(action_obj),
        })

        return self.create_order(action_obj)

    def _battle_finished_callback(self, battle):
        won = bool(getattr(battle, "won", False))
        self.battles_done += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1

        self._write({
            "type": "battle_end",
            "ts": time.time(),
            "won": bool(won),
            "wins": int(self.wins),
            "losses": int(self.losses),
            "battles_done": int(self.battles_done),
        })
        return super()._battle_finished_callback(battle)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="Path to learner checkpoint .pt (blank => autoload latest from --ckpt_dir)")
    ap.add_argument("--ckpt_dir", default="checkpoints", help="Checkpoint directory for autoload")
    ap.add_argument("--format", default="gen9randombattle", help="Showdown format for laddering")
    ap.add_argument("--battles", type=int, default=10, help="How many ladder battles to play")
    ap.add_argument("--device", default="", help="cpu or cuda (blank => use cfg.infer.device)")
    ap.add_argument("--username", required=True, help="Showdown username")
    ap.add_argument("--password", required=True, help="Password")
    ap.add_argument("--log", default="eval_logs/eval.jsonl", help="Where to write JSONL logs")
    args = ap.parse_args()

    # use your new unified config
    cfg = RunConfig.default()

    ckpt_path = args.ckpt.strip()
    if not ckpt_path:
        ckpt_path = latest_ckpt_path(args.ckpt_dir)
        print(f"[eval] autoload ckpt: {ckpt_path}")

    device = "cpu"
    policy = LoadedPolicy.load(ckpt_path, cfg=cfg, device=device)

    player = EvalPlayer(
        policy=policy,
        log_path=args.log,
        battle_format=args.format,
        server_configuration=ShowdownServerConfiguration,  # public server
        account_configuration=AccountConfiguration(args.username.strip(), args.password),
        max_concurrent_battles=1,
        log_level=30,
        open_timeout=30.0,
    )

    try:
        await player.ladder(args.battles)
        print(f"[eval] done: battles={player.battles_done} wins={player.wins} losses={player.losses}")
        print(f"[eval] logs: {args.log}")
    finally:
        player.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
