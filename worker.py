# worker.py
from __future__ import annotations
import asyncio
import secrets
import ray
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from poke_env.player import Player
from poke_env import AccountConfiguration, ServerConfiguration

from showdown_obs import build_tokens
from config import RunConfig

import traceback
import sys
import os


def install_asyncio_exception_logging(loop: asyncio.AbstractEventLoop, *, crash: bool) -> None:
    def handler(loop, context):
        # context keys often include: "message", "exception", "future"/"task"
        msg = context.get("message", "asyncio exception")
        exc = context.get("exception")

        print("\n" + "="*80)
        print(f"[ASYNCIO] {msg}")
        if exc is not None:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        else:
            # sometimes it's just a message + a task
            print(context)
        print("="*80 + "\n", flush=True)

        if crash:
            # Crash the actor process so Ray restarts it
            # Using SystemExit reliably terminates the process.
            os._exit(1)

    loop.set_exception_handler(handler)

def crash_on_task_error(task: asyncio.Task, name: str) -> None:
    def _done(t: asyncio.Task):
        try:
            exc = t.exception()  # <-- this is the important part
        except asyncio.CancelledError:
            return
        if exc is not None:
            print("\n" + "=" * 80)
            print(f"[TASK FAILED] {name}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            print("=" * 80 + "\n", flush=True)
            os._exit(1)  # crash actor so Ray restarts it
    task.add_done_callback(_done)


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def count_faints_from_float_feats(float_feats_t: np.ndarray, obs) -> tuple[int, int]:
    """
    float_feats_t: [N, F_DIM] for one timestep
    Assumes token layout: index 1..12 are pokemon tokens (6 self + 6 opp)
    """
    start = 1
    end = 1 + 2 * obs.n_slots
    pok = float_feats_t[start:end]
    faint = (pok[:, obs.F_FAINTED] > 0.5).astype(np.int32)
    return int(faint[:6].sum()), int(faint[6:].sum())


def make_server_conf(host: str, port: int) -> ServerConfiguration:
    ws_url = f"ws://{host}:{port}/showdown/websocket"
    http_action = f"http://{host}:{port}/action.php?"
    return ServerConfiguration(ws_url, http_action)


def battle_id_for(agent_id: int, battle) -> str:
    tag = getattr(battle, "battle_tag", None)
    if not tag:
        tag = f"pyid_{id(battle)}"
    return f"{agent_id}:{tag}"

class WorkerLearnerClient:
    def __init__(self, learner_actor, *, min_episodes, max_episodes, wait_ms, max_pending_episodes):
        self.learner_actor = learner_actor
        self.min_episodes = int(min_episodes)
        self.max_episodes = int(max_episodes)
        self.wait_ms = float(wait_ms)

        # episode-level backpressure
        self._max_pending = int(max_pending_episodes)
        self._ep_sem = asyncio.Semaphore(self._max_pending)

        self._q: asyncio.Queue = asyncio.Queue()  # can be unbounded; sem is the bound
        self._pending = 0

        self.total_episodes = 0
        self.total_flushes = 0
        self.total_batches = 0

        self._loop = asyncio.get_event_loop()
        self._task = self._loop.create_task(self._loop_coro())
        crash_on_task_error(self._task, "WorkerLearnerClient._loop_coro")

    async def acquire_episode_slot(self) -> None:
        # blocks when too many episodes are waiting to be flushed
        await self._ep_sem.acquire()

    def submit_episode(self, float_feats, tok_type, owner, pos, subpos, entity_id, tmask,
                       amask, act, logp, val, rew, done) -> None:
        """
        Assumes caller has already acquired an episode slot via acquire_episode_slot().
        This must NEVER drop; if the loop is dead, it's better to crash the actor.
        """
        def _enqueue():
            self._pending += 1
            self.total_episodes += 1
            self._q.put_nowait((float_feats, tok_type, owner, pos, subpos, entity_id, tmask,
                                amask, act, logp, val, rew, done))

        try:
            self._loop.call_soon_threadsafe(_enqueue)
        except RuntimeError as e:
            # event loop is gone: release permit so we don't leak, then crash
            try:
                self._loop.call_soon_threadsafe(self._ep_sem.release)
            except Exception:
                pass
            raise

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

            lengths = np.asarray([it[0].shape[0] for it in items], dtype=np.int32)

            ff_cat   = np.concatenate([it[0] for it in items], axis=0).astype(np.float32, copy=False)
            tt_cat   = np.concatenate([it[1] for it in items], axis=0).astype(np.int64,  copy=False)
            own_cat  = np.concatenate([it[2] for it in items], axis=0).astype(np.int64,  copy=False)
            pos_cat  = np.concatenate([it[3] for it in items], axis=0).astype(np.int64,  copy=False)
            sub_cat  = np.concatenate([it[4] for it in items], axis=0).astype(np.int64,  copy=False)
            eid_cat  = np.concatenate([it[5] for it in items], axis=0).astype(np.int64,  copy=False)
            tmsk_cat = np.concatenate([it[6] for it in items], axis=0).astype(np.float32, copy=False)

            amask_cat= np.concatenate([it[7] for it in items], axis=0).astype(np.float32, copy=False)
            act_cat  = np.concatenate([it[8] for it in items], axis=0).astype(np.int64,  copy=False)
            logp_cat = np.concatenate([it[9] for it in items], axis=0).astype(np.float32, copy=False)
            val_cat  = np.concatenate([it[10] for it in items], axis=0).astype(np.float32, copy=False)
            rew_cat  = np.concatenate([it[11] for it in items], axis=0).astype(np.float32, copy=False)
            done_cat = np.concatenate([it[12] for it in items], axis=0).astype(np.float32, copy=False)

            try:
                ref = self.learner_actor.submit_packed_batch.remote(
                    ff_cat, tt_cat, own_cat, pos_cat, sub_cat, eid_cat, tmsk_cat,
                    amask_cat, act_cat, logp_cat, val_cat, rew_cat, done_cat, lengths
                )
                await asyncio.to_thread(ray.get, ref) 
                self.total_flushes += 1
                self.total_batches += len(items)
            finally:
                # bookkeeping + release episode slots (critical!)
                n_eps = int(lengths.shape[0])
                self._pending = max(0, self._pending - n_eps)
                for _ in range(n_eps):
                    self._ep_sem.release()


class WorkerInferenceClient:
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

        # BOUNDED queue: provides real backpressure
        self._q: asyncio.Queue = asyncio.Queue(maxsize=int(max_pending))

        self._task = asyncio.get_event_loop().create_task(self._loop())
        crash_on_task_error(self._task, "WorkerInferenceClient._loop")

    async def request(
        self,
        float_feats: np.ndarray,
        tok_type: np.ndarray,
        owner: np.ndarray,
        pos: np.ndarray,
        subpos: np.ndarray,
        entity_id: np.ndarray,
        tmask: np.ndarray,
        amask: np.ndarray,
        *,
        timeout_s: Optional[float] = None,  # keep optional if you still want it
    ) -> Tuple[int, float, float]:
        """
        If overloaded, this will BLOCK at `await self._q.put(...)`.
        If timeout_s is None, it will wait forever for inference.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        # This is the key: block producers when queue is full
        await self._q.put((fut, float_feats, tok_type, owner, pos, subpos, entity_id, tmask, amask))

        if timeout_s is None:
            return await fut
        return await asyncio.wait_for(fut, timeout=timeout_s)

    async def _loop(self):
        while True:
            items = []
            try:
                first = await asyncio.wait_for(self._q.get(), timeout=self.wait_ms / 1000.0)
                items.append(first)
            except asyncio.TimeoutError:
                continue

            for _ in range(self.max_batch - 1):
                try:
                    items.append(self._q.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if len(items) < self.min_batch:
                await asyncio.sleep(self.wait_ms / 1000.0)
                for _ in range(self.max_batch - len(items)):
                    try:
                        items.append(self._q.get_nowait())
                    except asyncio.QueueEmpty:
                        break

            ff_b   = np.stack([it[1] for it in items], axis=0).astype(np.float32, copy=False)
            tt_b   = np.stack([it[2] for it in items], axis=0).astype(np.int64,  copy=False)
            own_b  = np.stack([it[3] for it in items], axis=0).astype(np.int64,  copy=False)
            pos_b  = np.stack([it[4] for it in items], axis=0).astype(np.int64,  copy=False)
            sub_b  = np.stack([it[5] for it in items], axis=0).astype(np.int64,  copy=False)
            eid_b  = np.stack([it[6] for it in items], axis=0).astype(np.int64,  copy=False)
            tmsk_b = np.stack([it[7] for it in items], axis=0).astype(np.float32, copy=False)
            amask_b= np.stack([it[8] for it in items], axis=0).astype(np.float32, copy=False)

            try:
                ref = self.inference_actor.infer_batch.remote(
                    ff_b, tt_b, own_b, pos_b, sub_b, eid_b, tmsk_b, amask_b
                )
                # If your Ray version does NOT support `await ref`, use:
                # act_b, logp_b, val_b = await asyncio.to_thread(ray.get, ref)
                act_b, logp_b, val_b = await asyncio.to_thread(ray.get, ref)
            except Exception as e:
                for fut, *_ in items:
                    if not fut.done():
                        fut.set_exception(e)
                raise

            for i, (fut, *_rest) in enumerate(items):
                if fut.done():
                    continue
                fut.set_result((int(act_b[i]), float(logp_b[i]), float(val_b[i])))



class RayBatchedPlayer(Player):
    """
    One poke-env Player, actions come via per-worker local batcher -> InferenceActor.
    """
    def __init__(self, 
                 *, 
                 infer_client: WorkerInferenceClient, 
                 learn_client: WorkerLearnerClient, 
                 learner_actor, 
                 agent_id: int, 
                 cfg: RunConfig,
                 **kwargs):
        super().__init__(**kwargs)
        self.infer_client = infer_client
        self.learn_client = learn_client
        self.learner_actor = learner_actor
        self.agent_id = int(agent_id)
        self.cfg: RunConfig = cfg
        self.obs = cfg.obs
        self.act_dim = int(cfg.env.act_dim)
        self._episode_slot_acquired: set[tuple[int, str]] = set()
        self._traj: Dict[Tuple[int, str], Dict[str, List[Any]]] = {}
    
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
    
        return mask, idx_to_action
    
    async def choose_move(self, battle):
        tb = build_tokens(battle, self.obs)
        amask, idx_to_action = self.legal_action_mask_and_map(battle)
    
        bid = battle_id_for(self.agent_id, battle)
        key = (self.agent_id, bid)
        
        if key not in self._episode_slot_acquired:
            await self.learn_client.acquire_episode_slot()
            self._episode_slot_acquired.add(key)
    
        # Retry forever if inference throws (optional, but aligns with "always wait")
        while True:
            try:
                a, logp, v = await self.infer_client.request(
                    tb.float_feats, tb.tok_type, tb.owner, tb.pos, tb.subpos, tb.entity_id,
                    tb.attn_mask, amask,
                    timeout_s=None,  # <-- no timeout
                )
                break
            except Exception as e:
                print(f"[choose_move] infer failed (will retry): {e!r}", flush=True)
                await asyncio.sleep(0.25)
    
        if key not in self._traj:
            self._traj[key] = {
                "ff": [], "tt": [], "own": [], "pos": [], "sub": [], "eid": [], "tmask": [],
                "amask": [], "act": [], "logp": [], "val": []
            }
    
        # record step
        self._traj[key]["ff"].append(tb.float_feats)
        self._traj[key]["tt"].append(tb.tok_type)
        self._traj[key]["own"].append(tb.owner)
        self._traj[key]["pos"].append(tb.pos)
        self._traj[key]["sub"].append(tb.subpos)
        self._traj[key]["eid"].append(tb.entity_id)
        self._traj[key]["tmask"].append(tb.attn_mask)
    
        self._traj[key]["amask"].append(amask)
        self._traj[key]["act"].append(int(a))
        self._traj[key]["logp"].append(float(logp))
        self._traj[key]["val"].append(float(v))
    
        action = idx_to_action.get(int(a), None)
        if action is None:
            # If the net picks an illegal action, that’s a modeling/bug issue.
            return self.choose_random_move(battle)
        
        kind = action[0]
        
        if kind == "move":
            _, move_slot, use_tera = action
            moves = list(battle.available_moves)
            if move_slot >= len(moves):
                return self.choose_random_move(battle)
            return self.create_order(moves[move_slot], terastallize=use_tera)
        
        elif kind == "switch":
            _, switch_slot = action
            switches = list(battle.available_switches)
            if switch_slot >= len(switches):
                return self.choose_random_move(battle)
            return self.create_order(switches[switch_slot])

    def _battle_finished_callback(self, battle):
        bid = battle_id_for(self.agent_id, battle)
        key = (self.agent_id, bid)
        try:
            buf = self._traj.pop(key, None)
            if not buf or len(buf["act"]) == 0:
                return super()._battle_finished_callback(battle)

            T = len(buf["act"])

            ff   = np.stack(buf["ff"],   axis=0).astype(np.float32)  # [T,N,F]
            tt   = np.stack(buf["tt"],   axis=0).astype(np.int64)
            own  = np.stack(buf["own"],  axis=0).astype(np.int64)
            pos  = np.stack(buf["pos"],  axis=0).astype(np.int64)
            sub  = np.stack(buf["sub"],  axis=0).astype(np.int64)
            eid  = np.stack(buf["eid"],  axis=0).astype(np.int64)
            tmsk = np.stack(buf["tmask"],axis=0).astype(np.float32)  # [T,N]
            
            amask = np.stack(buf["amask"], axis=0).astype(np.float32)  # [T,A]
            act   = np.asarray(buf["act"],  dtype=np.int64)
            logp  = np.asarray(buf["logp"], dtype=np.float32)
            val   = np.asarray(buf["val"],  dtype=np.float32)
            
            rew  = np.zeros((T,), dtype=np.float32)
            done = np.zeros((T,), dtype=np.float32)
            done[-1] = 1.0
            
            prev_self, prev_opp = count_faints_from_float_feats(ff[0], self.obs)
            for t in range(1, T):
                cur_self, cur_opp = count_faints_from_float_feats(ff[t], self.obs)
                delta_opp  = max(0, cur_opp - prev_opp)
                delta_self = max(0, cur_self - prev_self)
                rew[t] = float(delta_opp - delta_self)
                prev_self, prev_opp = cur_self, cur_opp
            
            self.learn_client.submit_episode(
                ff, tt, own, pos, sub, eid, tmsk,
                amask, act, logp, val, rew, done
            )


        finally:
            self._episode_slot_acquired.discard(key)
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
        cfg: RunConfig,
        inference_actor,
        learner_actor,
        *,
        pairs_in_worker: int,
        server_port: int,
    ):
        loop = asyncio.get_event_loop()
        install_asyncio_exception_logging(loop, crash=True)
        
        self.inference_actor = inference_actor
        self.learner_actor = learner_actor
        self.pairs_in_worker = int(pairs_in_worker)
        
        self.cfg: RunConfig = cfg

        self.server_conf = make_server_conf("localhost", server_port)
        self.run_tag = secrets.token_hex(3)

        self.infer_client = WorkerInferenceClient(
            inference_actor,
            min_batch=cfg.rollout.infer_min_batch,
            max_batch=cfg.rollout.infer_max_batch,
            wait_ms=cfg.rollout.infer_wait_ms,
            max_pending=cfg.rollout.infer_max_pending,
        )
        
        self.learn_client = WorkerLearnerClient(
            learner_actor,
            min_episodes=cfg.rollout.learn_min_episodes,
            max_episodes=cfg.rollout.learn_max_episodes,
            wait_ms=cfg.rollout.learn_wait_ms,
            max_pending_episodes=cfg.rollout.learn_max_pending_episodes
        )


    async def run(self, *, rooms_per_pair: Optional[int] = None):
        rooms_per_pair = int(rooms_per_pair or self.cfg.rollout.rooms_per_pair)
        battle_format  = self.cfg.env.battle_format

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
                open_timeout=self.cfg.rollout.open_timeout,
                ping_interval=self.cfg.rollout.ping_interval,
                ping_timeout=self.cfg.rollout.ping_timeout,
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
                open_timeout=self.cfg.rollout.open_timeout,
                ping_interval=self.cfg.rollout.ping_interval,
                ping_timeout=self.cfg.rollout.ping_timeout,
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
