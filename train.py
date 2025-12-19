# train.py
from __future__ import annotations

import asyncio
import math
import sys
import ray

from config import RunConfig
from inference import InferenceActor
from learner import LearnerActor
from worker import RolloutWorker


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

SERVER_PORTS = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]


async def main():
    ray.init(ignore_reinit_error=True)

    # ---- global config (single source of truth) ----
    cfg = RunConfig.default()
    
    rooms_per_pair = int(cfg.rollout.rooms_per_pair)
    target_concurrent_battles = int(cfg.rollout.target_concurrent_battles)

    # how many pairs total?
    total_pairs = math.ceil(target_concurrent_battles / rooms_per_pair)

    # how many pairs per worker actor (reduce websocket overhead)
    n_workers = len(SERVER_PORTS)
    base = total_pairs // n_workers
    rem = total_pairs % n_workers
    pairs_per_worker_list = [base + (1 if i < rem else 0) for i in range(n_workers)]

    effective_concurrency = total_pairs * rooms_per_pair

    print(
        f"[driver] target_concurrent_battles={target_concurrent_battles} rooms_per_pair={rooms_per_pair} "
        f"total_pairs={total_pairs} n_workers(one per server)={n_workers} "
        f"effective_concurrency~={effective_concurrency} "
        f"pairs_per_worker={pairs_per_worker_list}"
    )
    
    # ---- Create inference + learner actors  ----
    InferRemote = ray.remote(num_gpus=0.35)(InferenceActor)
    LearnerRemote = ray.remote(num_gpus=0.65)(LearnerActor)
    cfg_ref = ray.put(cfg)
    _PINNED_REFS = [cfg_ref]
    
    infer = InferRemote.remote(cfg_ref)
    learner = LearnerRemote.remote(cfg_ref, infer)

    # ---- Rollout workers (CPU only) ----
    WorkerRemote = ray.remote(num_cpus=1)(RolloutWorker).options(max_restarts=-1, max_task_retries=0)

    workers = []
    for i, port in enumerate(SERVER_PORTS):
        pairs_here = pairs_per_worker_list[i]
        if pairs_here <= 0:
            continue

        w = WorkerRemote.remote(
            cfg_ref,
            infer,
            learner,
            pairs_in_worker=pairs_here,
            server_port=port,
        )
        workers.append(w)

    print(f"[driver] started rollout_workers={len(workers)} (<= num_servers={len(SERVER_PORTS)})")

    # start workers
    _run_refs = [w.run.remote(rooms_per_pair=rooms_per_pair) for w in workers]
    
    # monitor loop (Ray ObjectRefs are not awaitables)
    while True:
        istats_ref = infer.get_stats.remote()
        lstats_ref = learner.get_stats.remote()
        istats, lstats = await asyncio.to_thread(ray.get, [istats_ref, lstats_ref])
        print(f"[stats] infer={istats} learner={lstats}")
        await asyncio.sleep(4.0)


if __name__ == "__main__":
    asyncio.run(main())
