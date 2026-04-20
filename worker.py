"""
Distributed Rollout Worker for Pokémon Showdown RL.

This module uses a Thread-Bridge architecture to quarantine poke-env's async 
websocket logic into a background thread. The main thread operates completely 
synchronously, gathering perfect numpy batches and dispatching them to a central 
GPU Inference Actor via zero-copy Ray RPCs.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import sys
import time
import traceback
import threading
import queue
import secrets
from typing import Any, Dict, List, Tuple, Final
from collections import Counter

import numpy as np
import ray
import poke_env
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import DefaultBattleOrder
from poke_env.player.baselines import SimpleHeuristicsPlayer
from config import RunConfig
from obs_assembler import ObservationAssembler

# Global Constants
ZOMBIE_LIMIT_S: Final[float] = 300.0  
REPAIR_INTERVAL_S: Final[float] = 30.0

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MONKEY PATCHES & SYSTEM FIXES
# ---------------------------------------------------------------------------
_original_handle_message = poke_env.ps_client.PSClient._handle_message

def _apply_patches():
    """Applies necessary patches to poke_env and asyncio for stability."""
    original_handle = poke_env.ps_client.PSClient._handle_message
    async def loud_handle(self, message):
        try:
            if "|error|[Invalid choice]" in message:
                room_id = message.split("\n")[0].replace(">", "").strip()
                logger.debug(f"NN chose invalid move in {room_id}. Forcing /choose default.")
                await self.send_message("/choose default", room=room_id)
            await original_handle(self, message)
        except Exception:
            logger.critical(f"Poke-env crash on message: {message[:100]}...")
            traceback.print_exc()
            raise
    poke_env.ps_client.PSClient._handle_message = loud_handle

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
    tag = getattr(battle, "battle_tag", None)
    if not tag: tag = f"pyid_{id(battle)}"
    return str(tag)

def mk_name(run_tag: str, p: int, side: str) -> str:
    return f"p{run_tag}{p:03d}{side}"

# --- RESTORED NETWORKING HELPERS ---
async def wait_for_login(player: SyncBridgePlayer, timeout_s: float = 30.0) -> None:
    c = getattr(player, "ps_client", None)
    if c is None: raise RuntimeError(f"{player.username} has no ps_client")
    fn = getattr(c, "wait_for_login", None)
    if not callable(fn): raise RuntimeError(f"{player.username}.ps_client has no wait_for_login()")
    await asyncio.wait_for(fn(), timeout=timeout_s)

async def safe_send(player: SyncBridgePlayer, message: str, room: str, *, retries: int = 8) -> None:
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

async def join_lobby(player: SyncBridgePlayer) -> None:
    await safe_send(player, "/join lobby", room="")

# ---------------------------------------------------------------------------
# SYNC LEARNER CLIENT
# ---------------------------------------------------------------------------

class SyncLearnerClient:
    def __init__(self, learner_actor, cfg: RunConfig):
        self.learner_actor = learner_actor
        self.cfg = cfg.rollout
        self.q = queue.Queue(maxsize=self.cfg.learn_max_pending_batches)
        threading.Thread(target=self._worker, daemon=True).start()

    def submit_episode(self, obs: np.ndarray, act: np.ndarray, logp: np.ndarray, 
                       val: np.ndarray, rew: np.ndarray, done: np.ndarray) -> bool:
        if self.q.full():
            return False
        self.q.put_nowait((obs, act, logp, val, rew, done))
        return True

    def _worker(self):
        while True:
            items = [self.q.get()]
            while len(items) < self.cfg.learn_max_episodes:
                try: items.append(self.q.get_nowait())
                except queue.Empty: break
            
            packed = self._prepare_batch(items)
            try:
                self.learner_actor.submit_packed_batch.remote(*packed)
            except Exception as e:
                logger.error(f"Learner submission failed: {e}")

    @staticmethod
    def _prepare_batch(items: List[Tuple]) -> Tuple:
        lengths = np.asarray([it[1].shape[0] for it in items], dtype=np.int32)
        return (
            np.concatenate([it[0] for it in items], axis=0),
            np.concatenate([it[1] for it in items], axis=0).astype(np.int64),
            np.concatenate([it[2] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[3] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[4] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[5] for it in items], axis=0).astype(np.float32),
            lengths
        )

# ---------------------------------------------------------------------------
# TOLLBOOTH PLAYER (RUNS IN ASYNC THREAD)
# ---------------------------------------------------------------------------

class SyncBridgePlayer(SimpleHeuristicsPlayer):
    def __init__(self, cfg: RunConfig, event_queue: queue.Queue, action_futures: dict, ep_sem: asyncio.Semaphore, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.event_queue = event_queue
        self.action_futures = action_futures
        self.ep_sem = ep_sem
        self.assembler = ObservationAssembler()
        self._battle_starts = {}
        self._last_act_time = {}
        self._episode_slot_state = {} 
        self._battle_events = {}

    def _get_unique_tag(self, battle):
        return f"{self.username}_{battle_tag_for(battle)}"
    
    async def _handle_battle_message(self, split_messages: list[list[str]]):
        """Corrected for 0.15: Processes a batch of pre-split protocol lines."""
        if not split_messages or not split_messages[0]:
            await super()._handle_battle_message(split_messages)
            return

        # 1. Extract Room ID from the first line
        room_id = None
        if split_messages[0] and split_messages[0][0].startswith(">"):
            room_id = split_messages[0][0][1:].strip()

        # 2. Harvest Events
        if room_id:
            for parts in split_messages:
                # Valid protocol lines start with an empty string (from the '|' split)
                if len(parts) > 1 and parts[0] == "":
                    cmd = parts[1]
                    # Filter out structural tags we don't need for transitions
                    if cmd not in ("t:", "request", "turn"):
                        # Save the parts exactly as obs_transitions expects them
                        self._battle_events.setdefault(room_id, []).append(parts)

        # 3. Pass to parent so poke-env can update its internal Battle object
        await super()._handle_battle_message(split_messages)

    async def choose_move(self, battle):
        try:
            tag = self._get_unique_tag(battle)
            if getattr(battle, 'finished', False):
                return DefaultBattleOrder()
                
            now = time.time()
            self._last_act_time[tag] = now
            self._battle_starts.setdefault(tag, now)

            if tag not in self._episode_slot_state:
                await self.ep_sem.acquire()
                if tag not in self._battle_starts:
                    self._safe_release()
                    return DefaultBattleOrder()
                self._episode_slot_state[tag] = "acquired"
            
            events = self._battle_events.pop(battle.battle_tag, [])
            obs_flat = self.assembler.assemble(battle, events=events)
            
            if getattr(self.cfg.learner, "mode", "ppo") == "imitation":
                order = super().choose_move(battle)
                action_idx = self.assembler.map_order_to_index(order, battle)
                
                # Send directly to the trajectory compiler
                self.event_queue.put_nowait(("IMITATION", tag, obs_flat, action_idx))
                
                action_obj, kwargs = self.assembler.map_index_to_order(action_idx, battle)
                if action_obj == "DEFAULT": return DefaultBattleOrder()
                return self.create_order(action_obj, **kwargs)

            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self.action_futures[tag] = (fut, loop) 

            self.event_queue.put_nowait(("STEP", tag, obs_flat, battle))

            action_idx = await fut

            action_obj, kwargs = self.assembler.map_index_to_order(action_idx, battle)
            if action_obj == "DEFAULT": return DefaultBattleOrder()
            return self.create_order(action_obj, **kwargs)

        except Exception as e:
            logger.error(f"SyncBridgePlayer crashed for {battle.battle_tag}: {e}. Using fallback.")
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        tag = self._get_unique_tag(battle)
        self.event_queue.put_nowait(("DONE", tag, None, battle))
        
        self._battle_starts.pop(tag, None)
        self._last_act_time.pop(tag, None)
        self._episode_slot_state.pop(tag, None)
        
        self.action_futures.pop(tag, None)
        base_tag = tag.split("_", 1)[1] if "_" in tag else tag
        if hasattr(self, "_battles"):
            self._battles.pop(base_tag, None)
        self._battle_events.pop(base_tag, None)
        
        super()._battle_finished_callback(battle)

    def _cleanup_local_battle(self, tag: str):
        self._battle_starts.pop(tag, None)
        self._last_act_time.pop(tag, None)
        self._episode_slot_state.pop(tag, None)
        
        self.action_futures.pop(tag, None) 
        
        self.event_queue.put_nowait(("CLEANUP", tag, None, None))
        
        base_tag = tag.split("_", 1)[1] if "_" in tag else tag
        if hasattr(self, "_battles"):
            self._battles.pop(base_tag, None)
        self._battle_events.pop(base_tag, None)

    def _safe_release(self):
        try: self.ep_sem.release()
        except ValueError: pass

    async def run_reconciliation(self):
        try: await self.ps_client.send_message("/rlactive", room="")
        except Exception: pass

    def _handle_query(self, query_type: str, data: Any) -> None:
        super()._handle_query(query_type, data)
        if query_type == "rlactive":
            if not data or isinstance(data, list):
                server_ids = set()
            else:
                server_ids = set(str(data).split(","))
            
            server_ids.discard("")
            local_map = {tag: (tag.split("_", 1)[1] if "_" in tag else tag) for tag in self._battle_starts.keys()}
            local_base_ids = set(local_map.values())
            
            ghosts = list(server_ids - local_base_ids)
            for bid in ghosts:
                asyncio.create_task(self.ps_client.send_message(f"/join {bid}", room=""))
            
            now = time.time()
            stalls = [tag for tag, bid in local_map.items() if bid in server_ids and (now - self._last_act_time.get(tag, 0)) > 60.0]
            for tag in stalls:
                bid = local_map[tag]
                self._cleanup_local_battle(tag)
                asyncio.create_task(self.ps_client.send_message(f"/rlrescue {bid}", room=""))
                asyncio.create_task(self.ps_client.send_message(f"/join {bid}", room=""))
                
            zombies = [tag for tag, bid in local_map.items() if bid not in server_ids and (now - self._battle_starts.get(tag, 0)) > ZOMBIE_LIMIT_S]
            for tag in zombies:
                self._cleanup_local_battle(tag)

# ---------------------------------------------------------------------------
# VECTOR WRAPPER (THE BRIDGE)
# ---------------------------------------------------------------------------

class PokeEnvVectorWrapper:
    def __init__(self, cfg: RunConfig, pairs_count: int, server_port: int):
        self.cfg = cfg
        self.pairs_count = pairs_count
        self.server_conf = make_server_conf("127.0.0.1", server_port)
        self.run_tag = secrets.token_hex(3)
        
        self.event_queue = queue.Queue()
        self.action_futures = {} 
        self.active_pairs = []

        self.async_loop = asyncio.new_event_loop()
        self.bg_thread = threading.Thread(target=self._run_bg_loop, daemon=True)
        self.bg_thread.start()

    def _run_bg_loop(self):
        asyncio.set_event_loop(self.async_loop)
        self.ep_sem = asyncio.Semaphore(self.cfg.rollout.learn_max_pending_episodes)
        self.async_loop.run_until_complete(self._init_players_and_spawn())
        self.async_loop.run_forever()

    async def _init_players_and_spawn(self):
        for i in range(self.pairs_count):
            pA = SyncBridgePlayer(
                cfg=self.cfg,
                event_queue=self.event_queue, action_futures=self.action_futures, ep_sem=self.ep_sem,
                account_configuration=AccountConfiguration(mk_name(self.run_tag, i, "a"), None),
                server_configuration=self.server_conf, max_concurrent_battles=self.cfg.rollout.rooms_per_pair,
                battle_format=self.cfg.env.battle_format, start_listening=True, log_level=40
            )
            pB = SyncBridgePlayer(
                cfg=self.cfg,
                event_queue=self.event_queue, action_futures=self.action_futures, ep_sem=self.ep_sem,
                account_configuration=AccountConfiguration(mk_name(self.run_tag, i, "b"), None),
                server_configuration=self.server_conf, max_concurrent_battles=self.cfg.rollout.rooms_per_pair,
                battle_format=self.cfg.env.battle_format, start_listening=True, log_level=40
            )
            self.active_pairs.append((pA, pB))

        players = [p for pair in self.active_pairs for p in pair]
        
        async def wait_for_login_retry(player, attempts=5, delay=1.0):
            last_err = None
            for i in range(attempts):
                try:
                    await wait_for_login(player)
                    return
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(delay)
            raise last_err

        for p in players:
            await wait_for_login_retry(p)
            await asyncio.sleep(0.05)
            
        await asyncio.gather(*[join_lobby(p) for p in players])
        
        for (pA, pB) in self.active_pairs:
            await safe_send(pA, f"/rlautospawn {pA.username}, {pB.username}, {self.cfg.env.battle_format}, {self.cfg.rollout.rooms_per_pair}", room="lobby")

        self.async_loop.create_task(self._maintenance_loop())

    async def _maintenance_loop(self):
        loop_idx = 0
        while True:
            await asyncio.sleep(REPAIR_INTERVAL_S)
            loop_idx += 1
            for pA, pB in self.active_pairs:
                asyncio.create_task(pA.run_reconciliation())
                asyncio.create_task(pB.run_reconciliation())
                
            if loop_idx % 6 == 0:
                for pA, pB in self.active_pairs:
                    asyncio.create_task(
                        safe_send(pA, f"/rlautospawn {pA.username}, {pB.username}, {self.cfg.env.battle_format}, {self.cfg.rollout.rooms_per_pair}", room="lobby")
                    )

    def release_slot(self):
        def _safe_release():
            try: self.ep_sem.release()
            except ValueError: pass
        self.async_loop.call_soon_threadsafe(_safe_release)

    def step(self, actions_dict: dict) -> List[Tuple]:
        for tag, action_idx in actions_dict.items():
            if tag in self.action_futures:
                fut, loop = self.action_futures.pop(tag)
                if not fut.done():
                    loop.call_soon_threadsafe(fut.set_result, action_idx)

        events = [self.event_queue.get()]
        time.sleep(0.003)
        
        while True:
            try: events.append(self.event_queue.get_nowait())
            except queue.Empty: break

        return events

# ---------------------------------------------------------------------------
# ROLLOUT WORKER (RAY ACTOR) - PURE SYNC MAIN THREAD
# ---------------------------------------------------------------------------

class RolloutWorker:
    def __init__(self, cfg: RunConfig, inference_actor, learner_actor, pairs: int, server_port: int):
        self.cfg = cfg
        self.learner_client = SyncLearnerClient(learner_actor, cfg)
        
        # --- NEW: Store the Central Inference Actor ---
        self.inference_actor = inference_actor
        
        self.vec_env = PokeEnvVectorWrapper(cfg, pairs, server_port)
        self._traj = {}
        self.assembler = ObservationAssembler()

    async def run(self):
        threading.Thread(target=self._run_rl_loop, daemon=True).start()
        while True:
            await asyncio.sleep(1.0)

    def _run_rl_loop(self):
        actions_dict = {}
        
        while True:
            events = self.vec_env.step(actions_dict)
            actions_dict = {}
            
            step_tags = []
            step_obs = []
            
            for event_type, tag, payload, extra_data in events:
                if event_type == "STEP":
                    step_tags.append(tag)
                    step_obs.append(payload) 
                elif event_type == "IMITATION":
                    traj = self._traj.setdefault(tag, {"obs": [], "act": [], "logp": [], "val": []})
                    traj["obs"].append(payload)    # payload is obs_flat
                    traj["act"].append(extra_data)      # extra is action_idx
                    traj["logp"].append(0.0)
                    traj["val"].append(0.0)
                elif event_type == "DONE":
                    self._finalize_trajectory(tag, extra_data)
                elif event_type == "CLEANUP":
                    self._cleanup_trajectory(tag)

            if step_obs:
                obs_batch = np.ascontiguousarray(np.stack(step_obs, axis=0), dtype=np.float32)
                
                # --- FIRE TO CENTRAL INFERENCE GPU ---
                # This blocks the worker thread until Ray returns the answers.
                # Because obs_batch is contiguous np.float32, Ray zero-copies it!
                try:
                    acts, logps, vals = ray.get(
                        self.inference_actor.infer_batch.remote(step_tags, obs_batch)
                    )
                except Exception as e:
                    logger.error(f"Inference Actor call failed: {e}")
                    # Fallback if InferenceActor crashes (avoids complete deadlock)
                    N = len(step_obs)
                    acts = np.zeros(N, dtype=np.int64)
                    logps = np.zeros(N, dtype=np.float32)
                    vals = np.zeros(N, dtype=np.float32)
                
                for i, tag in enumerate(step_tags):
                    actions_dict[tag] = int(acts[i])
                    
                    traj = self._traj.setdefault(tag, {"obs": [], "act": [], "logp": [], "val": []})
                    traj["obs"].append(step_obs[i])
                    traj["act"].append(acts[i])
                    traj["logp"].append(logps[i])
                    traj["val"].append(vals[i])

    def _cleanup_trajectory(self, tag: str):
        """Called when a Zombie/Stall is forcibly killed by the async maintenance thread."""
        self.inference_actor.clear_cache.remote(tag) # Fire and forget
        self._traj.pop(tag, None)
        self.vec_env.release_slot()

    def _finalize_trajectory(self, tag: str, battle):
        """Calculates rewards and sends to learner."""
        self.inference_actor.clear_cache.remote(tag) # Fire and forget
        
        if tag in self._traj: 
            buf = self._traj.pop(tag)
            if buf["act"]: 
                T = len(buf["act"])
                obs_stacked = np.stack(buf["obs"], axis=0)
                
                terminal_reward = self.cfg.reward.terminal_win if battle.won else self.cfg.reward.terminal_loss
                rewards = np.zeros(T, dtype=np.float32)
                
                if self.cfg.reward.use_faint_reward:
                    b_start, b_end = self.assembler.offsets["pokemon_body"]
                    faint_idx = self.assembler.meta["faint_internal_idx"]
                    
                    body_history = obs_stacked[:, b_start:b_end].reshape(T, 12, self.assembler.meta["dim_pokemon_body"])
                    is_fainted = body_history[:, :, faint_idx] > 0.5
                    
                    ds = np.diff(is_fainted[:, :6].sum(axis=1), prepend=is_fainted[0, :6].sum())
                    do = np.diff(is_fainted[:, 6:].sum(axis=1), prepend=is_fainted[0, 6:].sum())

                    rewards = (np.maximum(0, ds) * float(self.cfg.reward.faint_self)) + \
                              (np.maximum(0, do) * float(self.cfg.reward.faint_opp))

                rewards[-1] += terminal_reward
                dones = np.zeros(T, dtype=np.float32)
                dones[-1] = 1.0
                
                self.learner_client.submit_episode(
                    obs_stacked, np.array(buf["act"]), np.array(buf["logp"]),
                    np.array(buf["val"]), rewards, dones
                )
                
        self.vec_env.release_slot()

    def heartbeat(self):
        """Called by Train.py telemetry."""
        sem_val = getattr(self.vec_env, "ep_sem", None)
        return {
            "active_battles_worker": sum(len(p._battles) for pA, pB in self.vec_env.active_pairs for p in (pA, pB) if hasattr(p, "_battles")),
            "learner_q_size": self.learner_client.q.qsize(),
            "traj_in_memory": len(self._traj),
            "ep_sem_value": sem_val._value if sem_val else -1
        }