# worker.py
from __future__ import annotations
import asyncio
import secrets
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from poke_env.player import Player
from poke_env import AccountConfiguration, ServerConfiguration

from showdown_obs import build_unified_tokens, count_faints_from_tokens

import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ACT_DIM = 10  # 4 moves + 6 switches


def make_server_conf(host: str, port: int) -> ServerConfiguration:
    ws_url = f"ws://{host}:{port}/showdown/websocket"
    http_action = f"http://{host}:{port}/action.php?"
    return ServerConfiguration(ws_url, http_action)


def battle_id_for(agent_id: int, battle) -> str:
    tag = getattr(battle, "battle_tag", None)
    if not tag:
        tag = f"pyid_{id(battle)}"
    return f"{agent_id}:{tag}"


@dataclass
class RolloutConfig:
    battle_format: str = "gen9randombattle"
    rooms_per_pair: int = 16

    infer_timeout_s: float = 5.0
    open_timeout: float = 30.0
    ping_interval: float = 20.0
    ping_timeout: float = 20.0

    # NEW: local batching knobs (per RolloutWorker)
    infer_min_batch: int = 256
    infer_max_batch: int = 4096
    infer_wait_ms: float = 2.0

    # NEW: backpressure (prevents unbounded pending requests)
    infer_max_pending: int = 20000
    
    learn_min_episodes: int = 32
    learn_max_episodes: int = 256
    learn_wait_ms: float = 5.0
    learn_max_pending_episodes: int = 2000

class WorkerLearnerClient:
    def __init__(self, learner_actor, *, min_episodes, max_episodes, wait_ms, max_pending_episodes):
        self.learner_actor = learner_actor
        self.min_episodes = int(min_episodes)
        self.max_episodes = int(max_episodes)
        self.wait_ms = float(wait_ms)

        self._q: asyncio.Queue = asyncio.Queue()
        self._pending = 0
        self._max_pending = int(max_pending_episodes)

        self.total_episodes = 0
        self.total_flushes = 0
        self.total_batches = 0
        self.dropped = 0

        self._loop = asyncio.get_event_loop()
        self._task = self._loop.create_task(self._loop_coro())

    def submit_episode(self, tokens, tmask, amask, act, logp, val, rew, done) -> None:
        def _enqueue():
            if self._pending >= self._max_pending:
                self.dropped += 1
                return
            self._pending += 1
            self.total_episodes += 1
            self._q.put_nowait((tokens, tmask, amask, act, logp, val, rew, done))

        try:
            self._loop.call_soon_threadsafe(_enqueue)
        except RuntimeError:
            self.dropped += 1

    async def _loop_coro(self):
        while True:
            items = []
            try:
                first = await asyncio.wait_for(self._q.get(), timeout=self.wait_ms / 1000.0)
                items.append(first)
            except asyncio.TimeoutError:
                continue

            for _ in range(self.max_episodes - 1):
                try:
                    items.append(self._q.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if len(items) < self.min_episodes:
                await asyncio.sleep(self.wait_ms / 1000.0)
                for _ in range(self.max_episodes - len(items)):
                    try:
                        items.append(self._q.get_nowait())
                    except asyncio.QueueEmpty:
                        break

            # pack -> one Ray call
            lengths = np.asarray([it[3].shape[0] for it in items], dtype=np.int32)

            tokens_cat = np.concatenate([it[0] for it in items], axis=0).astype(np.float32, copy=False)
            tmask_cat  = np.concatenate([it[1] for it in items], axis=0).astype(np.float32, copy=False)
            amask_cat  = np.concatenate([it[2] for it in items], axis=0).astype(np.float32, copy=False)
            act_cat    = np.concatenate([it[3] for it in items], axis=0).astype(np.int64,  copy=False)
            logp_cat   = np.concatenate([it[4] for it in items], axis=0).astype(np.float32, copy=False)
            val_cat    = np.concatenate([it[5] for it in items], axis=0).astype(np.float32, copy=False)
            rew_cat    = np.concatenate([it[6] for it in items], axis=0).astype(np.float32, copy=False)
            done_cat   = np.concatenate([it[7] for it in items], axis=0).astype(np.float32, copy=False)

            try:
                self.learner_actor.submit_packed_batch.remote(
                    tokens_cat, tmask_cat, amask_cat, act_cat, logp_cat, val_cat, rew_cat, done_cat, lengths
                )
                self.total_flushes += 1
                self.total_batches += len(items)
            finally:
                self._pending = max(0, self._pending - len(items))



class WorkerInferenceClient:
    """
    Lives inside ONE RolloutWorker process.
    Batches many per-step requests into ONE Ray RPC: infer.infer_batch.remote(...)
    """
    def __init__(
        self,
        inference_actor,
        *,
        min_batch: int,
        max_batch: int,
        wait_ms: float,
        max_pending: int,
    ):
        self.inference_actor = inference_actor
        self.min_batch = int(min_batch)
        self.max_batch = int(max_batch)
        self.wait_ms = float(wait_ms)
        
        self._pending_sem = asyncio.Semaphore(int(max_pending))

        self._q: asyncio.Queue = asyncio.Queue()

        self._task = asyncio.get_event_loop().create_task(self._loop())

    async def request(self, tokens: np.ndarray, tmask: np.ndarray, amask: np.ndarray, *, timeout_s: float) -> Tuple[int, float, float]:
        await self._pending_sem.acquire()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._q.put_nowait((fut, tokens, tmask, amask))
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        finally:
            # release happens when fut is resolved OR times out
            self._pending_sem.release()

    async def _loop(self):
        while True:
            items: List[Tuple[asyncio.Future, np.ndarray, np.ndarray, np.ndarray]] = []

            # wait for at least one
            try:
                first = await asyncio.wait_for(self._q.get(), timeout=self.wait_ms / 1000.0)
                items.append(first)
            except asyncio.TimeoutError:
                continue

            # drain quickly
            for _ in range(self.max_batch - 1):
                try:
                    items.append(self._q.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # small-batch wait to improve batching
            if len(items) < self.min_batch:
                await asyncio.sleep(self.wait_ms / 1000.0)
                for _ in range(self.max_batch - len(items)):
                    try:
                        items.append(self._q.get_nowait())
                    except asyncio.QueueEmpty:
                        break

            # Stack to [B,...]
            tokens_b = np.stack([it[1] for it in items], axis=0).astype(np.float32, copy=False)  # [B,N,D]
            tmask_b  = np.stack([it[2] for it in items], axis=0).astype(np.float32, copy=False)  # [B,N]
            amask_b  = np.stack([it[3] for it in items], axis=0).astype(np.float32, copy=False)  # [B,A]

            # ONE Ray RPC for the whole batch
            try:
                act_b, logp_b, val_b = await self.inference_actor.infer_batch.remote(tokens_b, tmask_b, amask_b)
            except Exception as e:
                # fail the whole batch (callers will fall back to random move)
                for fut, *_ in items:
                    if not fut.done():
                        fut.set_exception(e)
                continue

            # respond
            for i, (fut, *_rest) in enumerate(items):
                if fut.done():
                    continue
                fut.set_result((int(act_b[i]), float(logp_b[i]), float(val_b[i])))


class RayBatchedPlayer(Player):
    """
    One poke-env Player, actions come via per-worker local batcher -> InferenceActor.
    """
    def __init__(self, *, infer_client: WorkerInferenceClient, learn_client: WorkerLearnerClient, learner_actor, agent_id: int, cfg: RolloutConfig, **kwargs):
        super().__init__(**kwargs)
        self.infer_client = infer_client
        self.learn_client = learn_client
        self.learner_actor = learner_actor
        self.agent_id = int(agent_id)
        self.cfg = cfg

        self._traj: Dict[Tuple[int, str], Dict[str, List[Any]]] = {}

    def legal_action_mask_and_map(self, battle):
        mask = np.zeros((ACT_DIM,), dtype=np.float32)
        idx_to_action: Dict[int, Any] = {}

        moves = list(battle.available_moves)[:4]
        switches = list(battle.available_switches)[:6]

        for i, mv in enumerate(moves):
            mask[i] = 1.0
            idx_to_action[i] = mv

        for j, sw in enumerate(switches):
            a = 4 + j
            mask[a] = 1.0
            idx_to_action[a] = sw

        return mask, idx_to_action

    async def choose_move(self, battle):
        tokens, token_mask = build_unified_tokens(battle)
        amask, idx_to_action = self.legal_action_mask_and_map(battle)

        bid = battle_id_for(self.agent_id, battle)
        key = (self.agent_id, bid)

        # LOCAL enqueue -> worker batch -> ONE Ray call per batch
        try:
            a, logp, v = await self.infer_client.request(
                tokens, token_mask, amask,
                timeout_s=self.cfg.infer_timeout_s,
            )
        except Exception:
            return self.choose_random_move(battle)

        if key not in self._traj:
            self._traj[key] = {"tokens": [], "tmask": [], "amask": [], "act": [], "logp": [], "val": []}

        self._traj[key]["tokens"].append(tokens)
        self._traj[key]["tmask"].append(token_mask)
        self._traj[key]["amask"].append(amask)
        self._traj[key]["act"].append(int(a))
        self._traj[key]["logp"].append(float(logp))
        self._traj[key]["val"].append(float(v))

        action_obj = idx_to_action.get(int(a), None)
        if action_obj is None:
            return self.choose_random_move(battle)
        return self.create_order(action_obj)

    def _battle_finished_callback(self, battle):
        try:
            bid = battle_id_for(self.agent_id, battle)
            key = (self.agent_id, bid)
            buf = self._traj.pop(key, None)
            if not buf or len(buf["act"]) == 0:
                return super()._battle_finished_callback(battle)

            T = len(buf["act"])

            tokens = np.stack(buf["tokens"], axis=0).astype(np.float32)  # [T,N,D]
            tmask  = np.stack(buf["tmask"], axis=0).astype(np.float32)   # [T,N]
            amask  = np.stack(buf["amask"], axis=0).astype(np.float32)   # [T,A]
            act    = np.asarray(buf["act"], dtype=np.int64)
            logp   = np.asarray(buf["logp"], dtype=np.float32)
            val    = np.asarray(buf["val"], dtype=np.float32)

            rew = np.zeros((T,), dtype=np.float32)
            done = np.zeros((T,), dtype=np.float32)
            done[-1] = 1.0

            prev_self, prev_opp = count_faints_from_tokens(tokens[0])
            for t in range(1, T):
                cur_self, cur_opp = count_faints_from_tokens(tokens[t])
                delta_opp  = max(0, cur_opp - prev_opp)
                delta_self = max(0, cur_self - prev_self)
                rew[t] = float(delta_opp - delta_self)
                prev_self, prev_opp = cur_self, cur_opp

            # fire-and-forget (don’t wrap in create_task; just submit)
            self.learn_client.submit_episode(tokens, tmask, amask, act, logp, val, rew, done)

        finally:
            try:
                tag = getattr(battle, "battle_tag", None)
                if tag is not None:
                    if hasattr(self, "_battles") and isinstance(self._battles, dict):
                        self._battles.pop(tag, None)
                    if hasattr(self, "battles") and isinstance(self.battles, dict):
                        self.battles.pop(tag, None)
            except Exception:
                pass
            return super()._battle_finished_callback(battle)


# -------- Helpers --------

def mk_name(run_tag: str, p: int, side: str) -> str:
    return f"p{run_tag}{p:03d}{side}"


async def wait_for_login(player: RayBatchedPlayer, timeout_s: float = 30.0) -> None:
    c = getattr(player, "ps_client", None)
    if c is None:
        raise RuntimeError(f"{player.username} has no ps_client")
    fn = getattr(c, "wait_for_login", None)
    if not callable(fn):
        raise RuntimeError(f"{player.username}.ps_client has no wait_for_login()")
    await asyncio.wait_for(fn(), timeout=timeout_s)


async def safe_send(player: RayBatchedPlayer, message: str, room: str, *, retries: int = 8) -> None:
    await wait_for_login(player)
    last_err = None
    for i in range(retries):
        try:
            await player.ps_client.send_message(message, room=room)
            return
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.05 * (i + 1))
    raise RuntimeError(f"[safe_send] failed: {message!r} ({last_err!r})")


async def join_lobby(player: RayBatchedPlayer) -> None:
    await safe_send(player, "/join lobby", room="")


class RolloutWorker:
    """
    One Ray actor runs multiple account-pairs (to reduce websocket count).
    Now it also owns a local inference microbatcher.
    """
    def __init__(
        self,
        cfg: RolloutConfig,
        inference_actor,
        learner_actor,
        *,
        pairs_in_worker: int,
        server_port: int,
    ):
        self.cfg = cfg
        self.inference_actor = inference_actor
        self.learner_actor = learner_actor
        self.pairs_in_worker = int(pairs_in_worker)

        self.server_conf = make_server_conf("localhost", server_port)
        self.run_tag = secrets.token_hex(3)

        self.infer_client = WorkerInferenceClient(
            inference_actor,
            min_batch=cfg.infer_min_batch,
            max_batch=cfg.infer_max_batch,
            wait_ms=cfg.infer_wait_ms,
            max_pending=cfg.infer_max_pending,
        )
        
        self.learn_client = WorkerLearnerClient(
            learner_actor,
            min_episodes=cfg.learn_min_episodes,
            max_episodes=cfg.learn_max_episodes,
            wait_ms=cfg.learn_wait_ms,
            max_pending_episodes=cfg.learn_max_pending_episodes,
        )

    async def run(self, *, rooms_per_pair: Optional[int] = None):
        rooms_per_pair = int(rooms_per_pair or self.cfg.rooms_per_pair)
        battle_format = self.cfg.battle_format

        pairs: list[tuple[RayBatchedPlayer, RayBatchedPlayer]] = []
        for p in range(self.pairs_in_worker):
            a_name = mk_name(self.run_tag, p, "a")
            b_name = mk_name(self.run_tag, p, "b")

            pA = RayBatchedPlayer(
                infer_client=self.infer_client,
                learn_client=self.learn_client,
                learner_actor=self.learner_actor,
                agent_id=(2 * p + 0),
                cfg=self.cfg,
                battle_format=battle_format,
                max_concurrent_battles=rooms_per_pair,
                log_level=40,
                open_timeout=self.cfg.open_timeout,
                ping_interval=self.cfg.ping_interval,
                ping_timeout=self.cfg.ping_timeout,
                server_configuration=self.server_conf,
                account_configuration=AccountConfiguration(a_name, None),
                start_listening=True,
            )
            pB = RayBatchedPlayer(
                infer_client=self.infer_client,
                learn_client=self.learn_client,
                learner_actor=self.learner_actor,
                agent_id=(2 * p + 1),
                cfg=self.cfg,
                battle_format=battle_format,
                max_concurrent_battles=rooms_per_pair,
                log_level=40,
                open_timeout=self.cfg.open_timeout,
                ping_interval=self.cfg.ping_interval,
                ping_timeout=self.cfg.ping_timeout,
                server_configuration=self.server_conf,
                account_configuration=AccountConfiguration(b_name, None),
                start_listening=True,
            )
            pairs.append((pA, pB))

        await asyncio.gather(*[wait_for_login(p) for pair in pairs for p in pair])
        await asyncio.gather(*[join_lobby(p) for pair in pairs for p in pair])

        await asyncio.gather(*[
            safe_send(
                pA,
                f"/rlautospawn {pA.username}, {pB.username}, {battle_format}, {rooms_per_pair}",
                room="lobby",
            )
            for (pA, pB) in pairs
        ])

        print(f"[worker] started pairs={self.pairs_in_worker} rooms_per_pair={rooms_per_pair}")

        while True:
            await asyncio.sleep(1.0)

    async def heartbeat(self):
        return {"pairs_in_worker": self.pairs_in_worker, "run_tag": self.run_tag}
