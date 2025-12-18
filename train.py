# train.py
from __future__ import annotations
import asyncio
import math
import ray

from inference import InferenceActor
from learner import LearnerActor, LearnerConfig
from worker import RolloutWorker, RolloutConfig

import sys, asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

SERVER_PORTS = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]

async def main():
    ray.init(ignore_reinit_error=True)

    # ---- knobs ----
    target_concurrent_battles = 4096
    rooms_per_pair = 16

    # how many pairs total?
    total_pairs = math.ceil(target_concurrent_battles / rooms_per_pair)

    # how many pairs per worker actor (reduce websocket overhead)
    n_workers = len(SERVER_PORTS)
    
    base = total_pairs // n_workers
    rem  = total_pairs % n_workers
    pairs_per_worker_list = [base + (1 if i < rem else 0) for i in range(n_workers)]

    effective_concurrency = total_pairs * rooms_per_pair

    print(
        f"[driver] target_concurrent_battles={target_concurrent_battles} rooms_per_pair={rooms_per_pair} "
        f"total_pairs={total_pairs} n_workers(one per server)={n_workers} "
        f"effective_concurrency~={effective_concurrency} "
        f"pairs_per_worker={pairs_per_worker_list}"
    )

    # ---- Create inference actor on GPU ----
    # IMPORTANT: Ray schedules GPU by num_gpus. This actor owns the GPU for inference.
    infer = ray.remote(num_gpus=0.35)(InferenceActor).remote(
        n_tokens=74,
        model_dim=64,
        n_layers=4,
        n_heads=1,
        act_dim=10,
        device="cuda",
    )

    # ---- Create learner actor on GPU ----
    learner_cfg = LearnerConfig(
        n_tokens=74, 
        model_dim=64, 
        n_layers=4, 
        n_heads=1, 
        act_dim=10,
        gamma=0.99, 
        gae_lambda=0.95,
        lr=3e-4, 
        update_epochs=4, 
        minibatch_size=4096,
        clip_coef=0.2, 
        ent_coef=0.01, 
        vf_coef=0.5,
        steps_per_update=32768,
        device="cuda",
    )
    learner = ray.remote(num_gpus=0.65)(LearnerActor).remote(learner_cfg, infer)

    # ---- Rollout workers (CPU only) ----
    wcfg = RolloutConfig(battle_format="gen9randombattle", rooms_per_pair=rooms_per_pair)

    WorkerRemote = ray.remote(num_cpus=1)(RolloutWorker)

    workers = []
    for i, port in enumerate(SERVER_PORTS):
        pairs_here = pairs_per_worker_list[i]
        if pairs_here <= 0:
            continue  # if fewer pairs than servers, some servers idle

        w = WorkerRemote.remote(
            wcfg,
            infer,
            learner,
            pairs_in_worker=pairs_here,
            server_port=port,
        )
        workers.append(w)

    print(f"[driver] started rollout_workers={len(workers)} (<= num_servers={len(SERVER_PORTS)})")

    # start workers
    _run_refs  = [w.run.remote(rooms_per_pair=rooms_per_pair) for w in workers]

    # small monitor loop
    while True:
        istats, lstats = await asyncio.gather(
            infer.get_stats.remote(),
            learner.get_stats.remote(),
        )
        print(f"[stats] infer={istats} learner={lstats}")
        await asyncio.sleep(2.0)


if __name__ == "__main__":
    asyncio.run(main())
