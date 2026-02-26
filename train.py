"""
Cluster Driver for Distributed Pokémon Showdown Reinforcement Learning.

This script initializes the Ray cluster, spawns fractional GPU actors for 
inference and training, and distributes rollout workers across multiple 
Pokémon Showdown server instances.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from typing import List, Final

import ray
from config import RunConfig
from inference import InferenceActor, WeightStore
from learner import LearnerActor
from worker import RolloutWorker

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# List of ports where local Pokémon Showdown server instances are running
SERVER_PORTS: Final[List[int]] = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009]



async def main():
    """
    Main entry point for the training cluster.
    
    Orchestrates the following:
    1. Ray initialization and configuration pinning.
    2. Fractional GPU allocation for Learner (0.65) and Inference (0.35).
    3. Distribution of self-play pairs across available local servers.
    4. Real-time telemetry and monitoring.
    """
    try:
        ray.init(ignore_reinit_error=True)
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        return

    # 1. Global Configuration (Single Source of Truth)
    cfg = RunConfig.default()
    rooms_per_pair = int(cfg.rollout.rooms_per_pair)
    target_concurrent = int(cfg.rollout.target_concurrent_battles)

    # Calculate distribution
    total_pairs = math.ceil(target_concurrent / rooms_per_pair)
    n_workers = len(SERVER_PORTS)
    
    # Balance pairs across worker actors
    base_pairs = total_pairs // n_workers
    remainder = total_pairs % n_workers
    pairs_per_worker = [base_pairs + (1 if i < remainder else 0) for i in range(n_workers)]

    logger.info(
        f"Initializing Run: Target Battles={target_concurrent}, "
        f"Total Pairs={total_pairs}, Workers={n_workers}"
    )

    # 2. Initialize Shared Infrastructure
    # We pin the config in Ray's object store to prevent unnecessary serialization
    cfg_ref = ray.put(cfg)
    weight_store = WeightStore.remote()

    # Define Remote Actors with Resource Constraints
    # Fractional GPUs allow both actors to coexist on a single high-end GPU
    InferRemote = ray.remote(num_gpus=0.35)(InferenceActor)
    LearnerRemote = ray.remote(num_gpus=0.65)(LearnerActor)
    WorkerRemote = ray.remote(num_cpus=1)(RolloutWorker).options(
        max_restarts=-1,  # Infinite restarts for robustness
        max_task_retries=0
    )

    # Spawn Core Actors
    infer = InferRemote.remote(cfg_ref, weight_store)
    learner = LearnerRemote.remote(cfg_ref, infer, weight_store)

    # 3. Spawn Rollout Workers
    workers = []
    for i, port in enumerate(SERVER_PORTS):
        count = pairs_per_worker[i]
        if count <= 0:
            continue

        worker_actor = WorkerRemote.remote(
            cfg_ref,
            infer,
            learner,
            pairs_in_worker=count,
            server_port=port,
        )
        workers.append(worker_actor)

    logger.info(f"Deployment complete. Distributed {len(workers)} rollout workers.")

    # 4. Staggered Startup
    # Allow core actors to initialize and load checkpoints before starting battles
    logger.info("Awaiting actor warm-up (20s)...")
    await asyncio.sleep(20)
    
    # Launch asynchronous rollout tasks
    _ = [w.run.remote(rooms_per_pair=rooms_per_pair) for w in workers]

    # 5. Monitoring Loop
    logger.info("Training started. Entering telemetry loop.")
    while True:
        try:
            # Gather telemetry from all actors in parallel
            istats_ref = infer.get_stats.remote()
            lstats_ref = learner.get_stats.remote()
            wstats_refs = [w.heartbeat.remote() for w in workers]

            # Ray 2.0+ pattern: Combine refs and fetch via ray.get in a thread
            results = await asyncio.to_thread(ray.get, [istats_ref, lstats_ref] + wstats_refs)
            
            istats = results[0]
            lstats = results[1]
            wstats_list = results[2:]

            # Aggregate worker-side metrics
            total_active = sum(w.get("active_battles_worker", 0) for w in wstats_list)
            total_library = sum(w.get("active_battles_library", 0) for w in wstats_list)
            avg_lag = sum(w.get("loop_lag_ms", 0) for w in wstats_list) / len(wstats_list)

            # Professional telemetry print
            # wbat: battles tracked by worker, pbat: battles tracked by poke-env library
            # MS: Average event loop lag in milliseconds
            stats_msg = (
                f"[Telemetry] WBAT: {total_active} | PBAT: {total_library} | LAG: {avg_lag:.2f}ms | "
                f"INFER: {istats} | TRAIN: {lstats}"
            )
            logger.info(stats_msg)

        except Exception as e:
            logger.warning(f"Telemetry loop encountered an error: {e}")
            # Do not exit; the cluster might still be healthy
        
        await asyncio.sleep(5.0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Terminating cluster...")
        ray.shutdown()