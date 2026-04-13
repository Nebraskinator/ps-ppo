"""
Distributed Rollout Worker for Pokémon Showdown RL.

This module handles the orchestration of self-play battles, interfacing with 
Ray actors for inference and experience collection. It includes robust 
reconciliation logic to handle server-side disconnects and memory leaks.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import random
import secrets
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Final

import numpy as np
import poke_env
import ray
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import DefaultBattleOrder, Player
from poke_env.player.baselines import SimpleHeuristicsPlayer
from ray.exceptions import ActorUnavailableError, GetTimeoutError

from config import RunConfig
from obs_assembler import ObservationAssembler

# Global Constants
GET_TIMEOUT_S: Final[float] = 8.0
ZOMBIE_LIMIT_S: Final[float] = 300.0  # Threshold for marking a battle as dead
REPAIR_INTERVAL_S: Final[float] = 30.0

# Setup logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MONKEY PATCHES & SYSTEM FIXES
# ---------------------------------------------------------------------------
_original_handle_message = poke_env.ps_client.PSClient._handle_message

async def _loud_handle_message(self, message):
    try:
        await _original_handle_message(self, message)
    except Exception as e:
        print(f"\n[CRITICAL POKE-ENV CRASH] Error handling message: {message[:100]}...", flush=True)
        import traceback
        traceback.print_exc()
        raise e

poke_env.ps_client.PSClient._handle_message = _loud_handle_message

def _apply_patches():
    """Applies necessary patches to poke_env and asyncio for stability."""
    
    # 1. Poke-env silent crash patch
    original_handle = poke_env.ps_client.PSClient._handle_message
    async def loud_handle(self, message):
        try:
            await original_handle(self, message)
        except Exception:
            logger.critical(f"Poke-env crash on message: {message[:100]}...")
            traceback.print_exc()
            raise
    poke_env.ps_client.PSClient._handle_message = loud_handle

    # 2. Windows Proactor Pipe fix
    if sys.platform == "win32":
        import asyncio.proactor_events
        def silence_proactor_error(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                try: return func(self, *args, **kwargs)
                except (AssertionError, OSError): pass
            return wrapper
        asyncio.proactor_events._ProactorBaseWritePipeTransport._loop_writing = \
            silence_proactor_error(asyncio.proactor_events._ProactorBaseWritePipeTransport._loop_writing)

_apply_patches()

def make_server_conf(host: str, port: int) -> ServerConfiguration:
    ws_url = f"ws://{host}:{port}/showdown/websocket"
    http_action = f"http://{host}:{port}/action.php?"
    return ServerConfiguration(ws_url, http_action)

def battle_tag_for(battle) -> str:
    """Prevents dictionary key collisions if the server omits a tag."""
    tag = getattr(battle, "battle_tag", None)
    if not tag: tag = f"pyid_{id(battle)}"
    return str(tag)

def mk_name(run_tag: str, p: int, side: str) -> str:
    """Creates strictly alphanumeric usernames for Showdown."""
    return f"p{run_tag}{p:03d}{side}"

async def wait_for_login(player: RayBatchedPlayer, timeout_s: float = 30.0) -> None:
    c = getattr(player, "ps_client", None)
    if c is None: raise RuntimeError(f"{player.username} has no ps_client")
    fn = getattr(c, "wait_for_login", None)
    if not callable(fn): raise RuntimeError(f"{player.username}.ps_client has no wait_for_login()")
    await asyncio.wait_for(fn(), timeout=timeout_s)

async def safe_send(player: RayBatchedPlayer, message: str, room: str, *, retries: int = 8) -> None:
    """Retries sending messages to handle brief websocket desyncs."""
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
    
# ---------------------------------------------------------------------------
# CLIENT CLASSES
# ---------------------------------------------------------------------------

class WorkerLearnerClient:
    """
    Handles asynchronous submission of trajectory data to the Learner actor.
    
    Uses an internal queue and background threads for array concatenation to 
    prevent blocking the main rollout event loop.
    """
    def __init__(self, learner_actor, cfg: RunConfig):
        self.learner_actor = learner_actor
        self.cfg = cfg.rollout
        self._ep_sem = asyncio.Semaphore(self.cfg.learn_max_pending_episodes)
        self._q: asyncio.Queue = asyncio.Queue()
        self._loop = asyncio.get_event_loop()
        self._task = self._loop.create_task(self._loop_coro())
        
    async def acquire_episode_slot(self) -> None:
        """Blocks until a slot is available for a new trajectory."""
        await self._ep_sem.acquire()

    def drop_episode(self) -> None:
        """Releases a slot without submission (e.g., if a battle errored)."""
        try: self._ep_sem.release()
        except ValueError: pass

    def submit_episode(self, *data: np.ndarray) -> None:
        """Enqueues finished episode data for batching and submission."""
        self._loop.call_soon_threadsafe(self._q.put_nowait, data)

    async def _loop_coro(self):
        """Main consumer loop that batches episodes and pushes to Ray."""
        while True:
            items = []
            try:
                # Wait for the first episode
                first = await asyncio.wait_for(self._q.get(), timeout=self.cfg.learn_wait_ms / 1000.0)
                items.append(first)
            except asyncio.TimeoutError:
                continue

            # Greedily gather up to max_episodes
            while len(items) < self.cfg.learn_max_episodes:
                try: items.append(self._q.get_nowait())
                except asyncio.QueueEmpty: break

            if len(items) < self.cfg.learn_min_episodes:
                await asyncio.sleep(self.cfg.learn_wait_ms / 1000.0)

            # Offload heavy NumPy concatenation to a thread
            packed = await asyncio.to_thread(self._prepare_batch, items)
            
            try:
                await self.learner_actor.submit_packed_batch.remote(*packed)
            except Exception as e:
                logger.error(f"Learner submission failed: {e}")
            finally:
                for _ in range(len(items)): self.drop_episode()

    @staticmethod
    def _prepare_batch(items: List[Tuple]) -> Tuple:
        """Concatenates lists of episode arrays into unified batch tensors."""
        lengths = np.asarray([it[1].shape[0] for it in items], dtype=np.int32)
        # indices: 0=obs, 1=act, 2=logp, 3=val, 4=rew, 5=done
        return (
            np.concatenate([it[0] for it in items], axis=0),
            np.concatenate([it[1] for it in items], axis=0).astype(np.int64),
            np.concatenate([it[2] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[3] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[4] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[5] for it in items], axis=0).astype(np.float32),
            lengths
        )


class WorkerInferenceClient:
    """
    Batches individual move requests into large matrices for high-throughput 
    GPU inference via the InferenceActor.
    """
    def __init__(self, inference_actor, cfg: RunConfig):
        self.inference_actor = inference_actor
        self.cfg = cfg.rollout
        self._q: asyncio.Queue = asyncio.Queue(maxsize=self.cfg.infer_max_pending)
        self._task = asyncio.get_event_loop().create_task(self._loop())

    async def request(self, tag: str, obs_flat: np.ndarray) -> Tuple[int, float, float]:
        """Submits an observation and waits for the action result."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self._q.put((fut, tag, obs_flat))
        return await fut
    
    def clear_cache(self, tag: str) -> None:
        """Fire-and-forget cache clear for finished/rescued battles."""
        try:
            self.inference_actor.clear_cache.remote(tag)
        except Exception:
            pass

    async def _loop(self):
        """Consumes the request queue and calls the remote inference actor."""
        while True:
            try:
                try:
                    # If the queue is empty, wait slightly.
                    # If the queue has items, get immediately.
                    first = await asyncio.wait_for(self._q.get(), timeout=0.01)
                    items = [first]
                except asyncio.TimeoutError:
                    continue

                # 2. Greedily grab everything currently in the queue up to max_batch
                # This removes the "wait_ms" latency for queues that are already full.
                while len(items) < self.cfg.infer_max_batch:
                    try:
                        items.append(self._q.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                
                tags = [it[1] for it in items]
                obs_batch = np.stack([it[2] for it in items], axis=0)

                try:
                    # Await Ray ObjectRef directly for performance
                    act, logp, val = await self.inference_actor.infer_batch.remote(tags, obs_batch)
                    
                    for i, (fut, _, _) in enumerate(items):
                        if not fut.done():
                            fut.set_result((int(act[i]), float(logp[i]), float(val[i])))

                except (GetTimeoutError, ActorUnavailableError) as e:
                    logger.warning(f"Inference Actor busy/dead: {e}")
                    for fut, _, _ in items: fut.set_exception(e)
            except Exception as e:
                logger.error(f"Inference loop error: {e}")
                await asyncio.sleep(0.1)

# ---------------------------------------------------------------------------
# PLAYER LOGIC
# ---------------------------------------------------------------------------

class RayBatchedPlayer(SimpleHeuristicsPlayer):
    """
    A Pokémon Showdown player that uses an InferenceClient for decision making 
    and records trajectories for PPO training.
    """
    def __init__(self, infer_client: WorkerInferenceClient, learn_client: WorkerLearnerClient, 
                 agent_id: int, cfg: RunConfig, **kwargs):
        super().__init__(**kwargs)
        self.infer_client = infer_client
        self.learn_client = learn_client
        self.agent_id = agent_id
        self.cfg = cfg
        self.assembler = ObservationAssembler()
        
        self._episode_slot_state: Dict[str, str] = {}
        self._traj: Dict[str, Dict[str, List[Any]]] = {}
        self._battle_starts: Dict[str, float] = {}
        self._last_act_time: Dict[str, float] = {}

    async def choose_move(self, battle):
        tag = battle_tag_for(battle)
        now = time.time()
        self._last_act_time[tag] = now
        self._battle_starts.setdefault(tag, now)
        
        # Acquire Learner Slot if first turn
        if tag not in self._episode_slot_state:
            await self.learn_client.acquire_episode_slot()
            self._episode_slot_state[tag] = "acquired"
            
        obs_flat = self.assembler.assemble(battle)
        
        # Decide: Heuristic (Imitation) or Model (PPO)
        if self.cfg.learner.mode == "imitation":
            order = super().choose_move(battle)
            
            # Create the full action array and randomize the memory tokens
            action_idx = self.assembler.map_order_to_index(order, battle)
            logp, val = 0.0, 0.0
        else:
            action_idx, logp, val = await self.infer_client.request(tag, obs_flat)
        
        # Log trajectory
        traj = self._traj.setdefault(tag, {"observations": [], "act": [], "logp": [], "val": []})
        traj["observations"].append(obs_flat)
        traj["act"].append(action_idx)
        traj["logp"].append(logp)
        traj["val"].append(val)

        # Map index back to poke-env order
        action_obj, kwargs = self.assembler.map_index_to_order(action_idx, battle)
        if action_obj == "DEFAULT": return DefaultBattleOrder()
        return self.create_order(action_obj, **kwargs)

    def _battle_finished_callback(self, battle):
        tag = battle_tag_for(battle)
        if tag in self._traj:
            self._process_experience(battle, tag)
        self._cleanup_local_battle(tag)
        super()._battle_finished_callback(battle)

    def _process_experience(self, battle, tag: str):
        """Calculates rewards and submits the completed trajectory to the Learner."""
        buf = self._traj[tag]
        if not buf["act"]: 
            return

        T = len(buf["act"])
        obs_stacked = np.stack(buf["observations"], axis=0)
        
        # Reward calculation
        terminal_reward = self.cfg.reward.terminal_win if battle.won else self.cfg.reward.terminal_loss
        rewards = np.zeros(T, dtype=np.float32)
        
        if self.cfg.reward.use_faint_reward:
            b_start, b_end = self.assembler.offsets["pokemon_body"]
            meta = self.assembler.meta
            
            dim_body  = meta["dim_pokemon_body"]
            faint_idx = meta["faint_internal_idx"] # Pulled dynamically
            
            body_history = obs_stacked[:, b_start:b_end].reshape(T, 12, dim_body)
        
            is_fainted = body_history[:, :, faint_idx] > 0.5
            
            faints_self = is_fainted[:, :6].sum(axis=1)
            faints_opp  = is_fainted[:, 6:].sum(axis=1)
        
            ds = np.diff(faints_self, prepend=faints_self[0])
            do = np.diff(faints_opp,  prepend=faints_opp[0])

            rewards = (np.maximum(0, ds) * float(self.cfg.reward.faint_self)) + \
                  (np.maximum(0, do) * float(self.cfg.reward.faint_opp))

        rewards[-1] += terminal_reward
        dones = np.zeros(T, dtype=np.float32)
        dones[-1] = 1.0
        
        self.learn_client.submit_episode(
            obs_stacked, np.array(buf["act"]), np.array(buf["logp"]), 
            np.array(buf["val"]), rewards, dones
        )
        
        if tag in self._episode_slot_state:
            self._episode_slot_state[tag] = "submitted"

    def _cleanup_local_battle(self, tag: str):
        """Wipes battle from all memory caches to prevent leaks."""
        self._battle_starts.pop(tag, None)
        self._last_act_time.pop(tag, None)
        self._traj.pop(tag, None)
        self.infer_client.clear_cache(tag)
        state = self._episode_slot_state.pop(tag, None)
        if state == "acquired":
            # Slot was taken, but data was never submitted. We must release it here.
            self.learn_client.drop_episode()
        
        # ADD THIS LINE: Manually clear from poke-env internal dictionary
        if hasattr(self, "_battles"):
            self._battles.pop(tag, None)

    async def run_reconciliation(self):
        """Issues a server-side query to find orphaned rooms."""
        try: await self.ps_client.send_message("/rlactive", room="")
        except Exception: pass
    
        
    def get_debug_stats(self):
        # Active battles = size of our tracking dict
        n_active = len(self._battle_starts)
            
        return n_active
    
    def _handle_query(self, query_type: str, data: Any) -> None:
        """Handles server-side room recovery broadcast for missing rooms."""
        super()._handle_query(query_type, data)
        
        if query_type == "rlactive":
            if not data or isinstance(data, list):
                server_ids = set()
            else:
                server_ids = set(str(data).split(","))
            
            server_ids.discard("")
            local_ids = set(self._battle_starts.keys())
            
            # 1. GHOSTS: Server has it, we don't. -> JOIN (Fixes missed Init)
            ghosts = list(server_ids - local_ids)
            for bid in ghosts:
                asyncio.create_task(self.ps_client.send_message(f"/join {bid}", room=""))
            
            # 2. STALLS: We have it, but haven't acted in > 60s. -> RESCUE
            now = time.time()
            stalls = [bid for bid in local_ids if bid in server_ids and (now - self._last_act_time.get(bid, 0)) > 60.0]
            
            for bid in stalls:
                tag = bid.split(":", 1)[1] if ":" in bid else bid
                self._cleanup_local_battle(tag)
                asyncio.create_task(self.ps_client.send_message(f"/rlrescue {bid}", room=""))
                asyncio.create_task(self.ps_client.send_message(f"/join {bid}", room=""))

# ---------------------------------------------------------------------------
# ROLLOUT WORKER (RAY ACTOR)
# ---------------------------------------------------------------------------

class RolloutWorker:
    """
    The top-level Ray Worker that manages multiple pairs of self-play bots.
    
    This class handles the lifecycle of the websocket connections and runs 
    periodic maintenance tasks (GC, Reconciliation).
    """
    def __init__(self, cfg: RunConfig, inference_actor, learner_actor, pairs: int, server_port: int):
        self.cfg = cfg
        self.infer_client = WorkerInferenceClient(inference_actor, cfg)
        self.learn_client = WorkerLearnerClient(learner_actor, cfg)
        self.server_conf = make_server_conf("127.0.0.1", server_port)
        self.pairs_count = pairs
        self.active_pairs: List[Tuple[RayBatchedPlayer, RayBatchedPlayer]] = []
        self.run_tag = secrets.token_hex(3)

    async def run(self):
        """Initializes all players and enters the main maintenance loop."""
        try:
            for i in range(self.pairs_count):
                a_name = mk_name(self.run_tag, i, "a")
                b_name = mk_name(self.run_tag, i, "b")
                pA = self._make_player(a_name, 2*i)
                pB = self._make_player(b_name, 2*i + 1)
                self.active_pairs.append((pA, pB))
            
            # Login and start spawning (Exact original sequence)
            players = [p for pair in self.active_pairs for p in pair]
            async def wait_for_login_retry(player, attempts=5, delay=1.0):
                last_err = None
                for i in range(attempts):
                    try:
                        await wait_for_login(player)
                        return
                    except Exception as e:
                        last_err = e
                        print(
                            f"[login state] user={player.username} "
                            f"logged_in={getattr(player.ps_client, 'logged_in', None).is_set() if getattr(player, 'ps_client', None) else 'no_client'}",
                            flush=True,
                        )
                        print(f"[login retry] {player.username} attempt {i+1}/{attempts} failed: {type(e).__name__}: {e}", flush=True)
                        await asyncio.sleep(delay)
                raise last_err
            for p in players:
                await wait_for_login_retry(p)
                await asyncio.sleep(0.05)
            await asyncio.gather(*[join_lobby(p) for p in players])
            await asyncio.gather(*[
                safe_send(pA, f"/rlautospawn {pA.username}, {pB.username}, {self.cfg.env.battle_format}, {self.cfg.rollout.rooms_per_pair}", room="lobby")
                for (pA, pB) in self.active_pairs
            ])
            loop_idx = 0
            while True:
                await asyncio.sleep(REPAIR_INTERVAL_S)
                loop_idx += 1
                for pA, pB in self.active_pairs:
                    asyncio.create_task(pA.run_reconciliation())
                    asyncio.create_task(pB.run_reconciliation())
                    
                # Periodically re-assert the spawn target to fix drift
                if loop_idx % 6 == 0:
                    for pA, pB in self.active_pairs:
                        asyncio.create_task(
                            safe_send(pA, f"/rlautospawn {pA.username}, {pB.username}, {self.cfg.env.battle_format}, {self.cfg.rollout.rooms_per_pair}", room="lobby")
                        )
                        
        except Exception as e:
            # Safely get the websocket URL, defaulting to "Unknown Server" if missing
            url = getattr(self.server_conf, "websocket_url", "Unknown Server")
            print(f"\n[CRITICAL WORKER CRASH] {url} failed in run(): {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _make_player(self, name: str, agent_id: int) -> RayBatchedPlayer:
        return RayBatchedPlayer(
            infer_client=self.infer_client, learn_client=self.learn_client,
            agent_id=agent_id, cfg=self.cfg,
            account_configuration=AccountConfiguration(name, None),
            server_configuration=self.server_conf,
            max_concurrent_battles=self.cfg.rollout.rooms_per_pair,
            battle_format=self.cfg.env.battle_format, 
            open_timeout=self.cfg.rollout.open_timeout, # Restored timeout
            ping_interval=None,                       
            ping_timeout=None,                        
            start_listening=True, log_level=40
        )
    
    async def heartbeat(self):
        # Measure Event Loop Lag
        t0 = time.time()
        await asyncio.sleep(0) # Yield to loop
        lag = (time.time() - t0) * 1000 # Milliseconds
        
        total_active = 0
        total_lib_count = 0 
        
        for pA, pB in self.active_pairs:
            total_active += pA.get_debug_stats() + pB.get_debug_stats()
            # Peek into private library counts
            if hasattr(pA, "_battles"): total_lib_count += len(pA._battles)
            if hasattr(pB, "_battles"): total_lib_count += len(pB._battles)
        
        return {
            "active_battles_worker": total_active,
            "active_battles_library": total_lib_count, # Compare this to worker!
            "loop_lag_ms": round(lag, 2),
        }
