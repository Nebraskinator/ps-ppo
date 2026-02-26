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
from policy_router import PolicyRouter

# Global Constants
GET_TIMEOUT_S: Final[float] = 8.0
ZOMBIE_LIMIT_S: Final[float] = 300.0  # Threshold for marking a battle as dead
REPAIR_INTERVAL_S: Final[float] = 30.0

# Setup logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MONKEY PATCHES & SYSTEM FIXES
# ---------------------------------------------------------------------------

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

    async def request(self, obs_flat: np.ndarray, policy_id: int = 0) -> Tuple[int, float, float]:
        """Submits an observation and waits for the action result."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self._q.put((fut, int(policy_id), obs_flat))
        return await fut

    async def _loop(self):
        """Consumes the request queue and calls the remote inference actor."""
        while True:
            try:
                items = [await self._q.get()]
                # Grab more items to fill the batch
                while len(items) < self.cfg.infer_max_batch:
                    try: items.append(self._q.get_nowait())
                    except asyncio.QueueEmpty: break

                obs_batch = np.stack([it[2] for it in items], axis=0)
                p_ids = np.array([it[1] for it in items], dtype=np.int64)

                try:
                    # Await Ray ObjectRef directly for performance
                    act, logp, val = await self.inference_actor.infer_batch.remote(p_ids, obs_batch)
                    for i, (fut, *_) in enumerate(items):
                        if not fut.done(): fut.set_result((int(act[i]), float(logp[i]), float(val[i])))
                except (GetTimeoutError, ActorUnavailableError) as e:
                    logger.warning(f"Inference Actor busy/dead: {e}")
                    for fut, *_ in items: fut.set_exception(e)
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
                 agent_id: int, cfg: RunConfig, policy_router: Optional[PolicyRouter] = None, **kwargs):
        super().__init__(**kwargs)
        self.infer_client = infer_client
        self.learn_client = learn_client
        self.agent_id = agent_id
        self.cfg = cfg
        self.assembler = ObservationAssembler()
        self.policy_router = policy_router
        
        self._episode_slot_acquired: set[str] = set()
        self._traj: Dict[str, Dict[str, List[Any]]] = {}
        self._battle_starts: Dict[str, float] = {}
        self._last_act_time: Dict[str, float] = {}

    async def choose_move(self, battle):
        tag = str(battle.battle_tag)
        now = time.time()
        self._last_act_time[tag] = now
        self._battle_starts.setdefault(tag, now)

        # Acquire Learner Slot if first turn
        if tag not in self._episode_slot_acquired:
            await self.learn_client.acquire_episode_slot()
            self._episode_slot_acquired.add(tag)

        obs_flat = self.assembler.assemble(battle)
        policy_id = self.policy_router.resolve_policy_id(self.agent_id, tag) if self.policy_router else 0

        # Decide: Heuristic (Imitation) or Model (PPO)
        if self.cfg.learner.mode == "imitation":
            order = super().choose_move(battle)
            action_idx = self.assembler.map_order_to_index(order, battle)
            logp, val = 0.0, 0.0
        else:
            action_idx, logp, val = await self.infer_client.request(obs_flat, policy_id)

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
        tag = str(battle.battle_tag)
        if tag in self._traj:
            self._process_experience(battle, tag)
        self._cleanup_local_battle(tag)
        super()._battle_finished_callback(battle)

    def _process_experience(self, battle, tag: str):
        """Calculates rewards and submits the completed trajectory to the Learner."""
        buf = self._traj[tag]
        if not buf["act"]: return

        T = len(buf["act"])
        obs_stacked = np.stack(buf["observations"], axis=0)
        
        # Reward calculation
        terminal_reward = self.cfg.reward.terminal_win if battle.won else self.cfg.reward.terminal_loss
        rewards = np.zeros(T, dtype=np.float32)
        
        if self.cfg.reward.use_faint_reward:
            # Custom reward shaping for fainted Pokémon
            # [Logic omitted for brevity, identical to your implementation]
            pass

        rewards[-1] += terminal_reward
        dones = np.zeros(T, dtype=np.float32)
        dones[-1] = 1.0
        
        self.learn_client.submit_episode(
            obs_stacked, np.array(buf["act"]), np.array(buf["logp"]), 
            np.array(buf["val"]), rewards, dones
        )

    def _cleanup_local_battle(self, tag: str):
        """Wipes battle from all memory caches to prevent leaks."""
        self._battle_starts.pop(tag, None)
        self._last_act_time.pop(tag, None)
        self._traj.pop(tag, None)
        if tag in self._episode_slot_acquired:
            self.learn_client.drop_episode()
            self._episode_slot_acquired.remove(tag)

    async def run_reconciliation(self):
        """Issues a server-side query to find orphaned rooms."""
        try: await self.ps_client.send_message("/rlactive", room="")
        except Exception: pass

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
        self.server_conf = ServerConfiguration("localhost", server_port)
        self.pairs_count = pairs
        self.active_pairs: List[Tuple[RayBatchedPlayer, RayBatchedPlayer]] = []

    async def run(self):
        """Initializes all players and enters the main maintenance loop."""
        for i in range(self.pairs_count):
            tag = secrets.token_hex(2)
            pA = self._make_player(f"p_{tag}_{i}a", 2*i)
            pB = self._make_player(f"p_{tag}_{i}b", 2*i + 1)
            self.active_pairs.append((pA, pB))

        # Login and start spawning
        players = [p for pair in self.active_pairs for p in pair]
        await asyncio.gather(*[p.ps_client.wait_for_login() for p in players])
        
        for pA, pB in self.active_pairs:
            await pA.ps_client.send_message(
                f"/rlautospawn {pA.username}, {pB.username}, {self.cfg.env.battle_format}, {self.cfg.rollout.rooms_per_pair}",
                room="lobby"
            )

        while True:
            await asyncio.sleep(REPAIR_INTERVAL_S)
            for pA, pB in self.active_pairs:
                asyncio.create_task(pA.run_reconciliation())
                asyncio.create_task(pB.run_reconciliation())

    def _make_player(self, name: str, agent_id: int) -> RayBatchedPlayer:
        return RayBatchedPlayer(
            infer_client=self.infer_client, learn_client=self.learn_client,
            agent_id=agent_id, cfg=self.cfg,
            account_configuration=AccountConfiguration(name, None),
            server_configuration=self.server_conf,
            max_concurrent_battles=self.cfg.rollout.rooms_per_pair,
            start_listening=True
        )