# learner.py
from __future__ import annotations

import asyncio
import glob
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from config import RunConfig
from ppo_core import ActorCriticTransformer, gae_from_episode, AsyncEpisodeDataset, ppo_update


class LearnerActor:
    """
    Receives completed episodes from workers, trains PPO, and sends weights to InferenceActor.
    Now driven by RunConfig (single source of truth).
    """
    def __init__(self, cfg: RunConfig, inference_actor):
        self.run_cfg = cfg
        self.cfg = cfg.learner
        self.inference_actor = inference_actor

        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("LearnerActor device=cuda but CUDA not available")

        self.net: Optional[ActorCriticTransformer] = None
        self.opt: Optional[optim.Optimizer] = None
        self.sched: Optional[optim.lr_scheduler._LRScheduler] = None

        # dataset stores on CPU; training copies to cfg.device per update (same as you do now)
        self.dataset = AsyncEpisodeDataset(act_dim=self.run_cfg.env.act_dim, device="cpu")

        self.update_idx = 0
        self.total_episodes = 0
        self.total_steps = 0

        self._q: asyncio.Queue = asyncio.Queue(
            maxsize=int(self.run_cfg.rollout.learn_max_pending_batches)
        )

        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop())
        self._task.add_done_callback(self._on_loop_done) 
        self._startup_task = loop.create_task(self._maybe_resume_latest())

    def _init_if_needed(self):
        if self.net is not None:
            return
        self.net = self.run_cfg.make_model().to(self.cfg.device).train()
        self.opt = optim.Adam(self.net.parameters(), 
                              lr=self.cfg.lr, 
                              eps=1e-5,
                              weight_decay=0.01,
                              )
        
        warmup_steps = int(getattr(self.cfg, "lr_warmup_steps", 1000))  # or hardcode 1000
        if warmup_steps > 0:
            def lr_lambda(step: int) -> float:
                # linear warmup to 1.0 multiplier
                return min(1.0, float(step + 1) / float(warmup_steps))
            self.sched = optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)
        else:
            self.sched = None
        
    def _on_loop_done(self, task: asyncio.Task):
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            print("[learner] loop task cancelled")
            return
        except Exception as e:
            print(f"[learner] loop task exception() failed: {e!r}")
            return

        if exc is not None:
            import traceback
            print("[learner] loop task DIED with exception:")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        else:
            print("[learner] loop task exited normally (unexpected)")


    # ---------- checkpointing helpers (unchanged except cfg payload) ----------

    def _ckpt_path_for_update(self, update_idx: int) -> str:
        return os.path.join(self.cfg.ckpt_dir, f"learner_update_{update_idx:06d}.pt")

    def _latest_ckpt_path(self) -> Optional[str]:
        paths = sorted(glob.glob(os.path.join(self.cfg.ckpt_dir, "learner_update_*.pt")))
        return paths[-1] if paths else None

    def _rotate_checkpoints(self):
        k = int(self.cfg.keep_last)
        if k <= 0:
            return
        paths = sorted(glob.glob(os.path.join(self.cfg.ckpt_dir, "learner_update_*.pt")))
        if len(paths) <= k:
            return
        for p in paths[:-k]:
            try:
                os.remove(p)
            except Exception:
                pass

    async def _maybe_resume_latest(self):
        if not self.cfg.resume:
            return
        path = self._latest_ckpt_path()
        if not path:
            return
        try:
            self._load_checkpoint(path)
            assert self.net is not None
            sd_cpu = {k: v.detach().to("cpu") for k, v in self.net.state_dict().items()}
            self.inference_actor.set_weights.remote(sd_cpu)
            print(f"[learner] resumed from {path}")
        except Exception as e:
            print(f"[learner] resume failed from {path}: {e!r}")

    def _save_checkpoint(self, path: str):
        assert self.net is not None and self.opt is not None

        payload = {
            "update_idx": int(self.update_idx),
            "total_episodes": int(self.total_episodes),
            "total_steps": int(self.total_steps),

            "model": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": (None if self.sched is None else self.sched.state_dict()),

            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),

            # Save the full RunConfig so everything is reproducible
            "run_cfg": self.run_cfg.as_dict(),
        }

        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        self._init_if_needed()

        assert self.net is not None and self.opt is not None
        self.net.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        if self.sched is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
            self.sched.load_state_dict(ckpt["scheduler"])


        self.update_idx = int(ckpt.get("update_idx", 0))
        self.total_episodes = int(ckpt.get("total_episodes", 0))
        self.total_steps = int(ckpt.get("total_steps", 0))

        if "torch_rng" in ckpt:
            rng = ckpt["torch_rng"]
            if isinstance(rng, torch.Tensor):
                rng = rng.detach().to("cpu")
                if rng.dtype != torch.uint8:
                    rng = rng.to(torch.uint8)
            torch.set_rng_state(rng)

        if "numpy_rng" in ckpt:
            np.random.set_state(ckpt["numpy_rng"])
        if "python_rng" in ckpt:
            random.setstate(ckpt["python_rng"])

    def _should_save_now(self) -> bool:
        if self.net is None or self.opt is None:
            return False
        return (self.cfg.save_every_updates > 0) and (self.update_idx % self.cfg.save_every_updates == 0)

    async def save_now(self, path: Optional[str] = None) -> str:
        if self.net is None or self.opt is None:
            raise RuntimeError("Cannot save: model not initialized yet (no episodes seen).")
        if path is None:
            path = self._ckpt_path_for_update(self.update_idx)
        self._save_checkpoint(path)
        self._rotate_checkpoints()
        return path

    # ----------------------------
    # Public API
    # ----------------------------

    async def get_stats(self) -> dict:
        return {
            "update": self.update_idx,
            "episodes": self.total_episodes,
            "steps_in_dataset": len(self.dataset),
            "total_steps": self.total_steps,
        }
    
    def submit_episode(self, tokens, tmask, amask, act, logp, val, rew, done):
        self._q.put_nowait((tokens, tmask, amask, act, logp, val, rew, done))
    
    async def submit_packed_batch(
        self,
        ff_cat: np.ndarray,    # [S,N,F]
        tt_cat: np.ndarray,    # [S,N]
        own_cat: np.ndarray,   # [S,N]
        pos_cat: np.ndarray,   # [S,N]
        sub_cat: np.ndarray,   # [S,N]
        eid_cat: np.ndarray,   # [S,N]
        tmask_cat: np.ndarray, # [S,N]
        amask_cat: np.ndarray, # [S,A]
        act_cat: np.ndarray,   # [S]
        logp_cat: np.ndarray,  # [S]
        val_cat: np.ndarray,   # [S]
        rew_cat: np.ndarray,   # [S]
        done_cat: np.ndarray,  # [S]
        lengths: np.ndarray,   # [B]
    ):
        await self._q.put(("packed", ff_cat, tt_cat, own_cat, pos_cat, sub_cat, eid_cat, tmask_cat,
                            amask_cat, act_cat, logp_cat, val_cat, rew_cat, done_cat, lengths))
        return True


    # ----------------------------
    # Main training loop
    # ----------------------------
    
    async def _loop(self):
        while True:
            item = await self._q.get()
            
            if isinstance(item, tuple) and len(item) > 0 and item[0] == "packed":
                _, ff_cat, tt_cat, own_cat, pos_cat, sub_cat, eid_cat, tmask_cat, amask_cat, act_cat, logp_cat, val_cat, rew_cat, done_cat, lengths = item
                self._init_if_needed()
                
                ff_all   = torch.from_numpy(ff_cat).float()
                tt_all   = torch.from_numpy(tt_cat).long()
                own_all  = torch.from_numpy(own_cat).long()
                pos_all  = torch.from_numpy(pos_cat).long()
                sub_all  = torch.from_numpy(sub_cat).long()
                eid_all  = torch.from_numpy(eid_cat).long()
                tmask_all= torch.from_numpy(tmask_cat).float()
                amask_all= torch.from_numpy(amask_cat).float()
                act_all  = torch.from_numpy(act_cat).long()
                logp_all = torch.from_numpy(logp_cat).float()
                val_all  = torch.from_numpy(val_cat).float()
                rew_all  = torch.from_numpy(rew_cat).float()
                done_all = torch.from_numpy(done_cat).float()


                adv_chunks = []
                ret_chunks = []
                start = 0
                for L in lengths.tolist():
                    end = start + int(L)
                    adv_t, ret_t = gae_from_episode(
                        rewards=rew_all[start:end],
                        values=val_all[start:end],
                        dones=done_all[start:end],
                        gamma=self.cfg.gamma,
                        lam=self.cfg.gae_lambda,
                        last_value=0.0,
                    )
                    adv_chunks.append(adv_t)
                    ret_chunks.append(ret_t)
                    start = end
                adv_all = torch.cat(adv_chunks, dim=0)
                ret_all = torch.cat(ret_chunks, dim=0)

                self.dataset.add_steps(
                    ff_all, tt_all, own_all, pos_all, sub_all, eid_all, tmask_all,
                    amask_all, act_all, logp_all, val_all, adv_all, ret_all
                )
                self.total_episodes += int(lengths.shape[0])
                self.total_steps += int(act_all.shape[0])
                    
    
            if len(self.dataset) < self.cfg.steps_per_update:
                continue

            # ---- PPO update ----
            assert self.net is not None and self.opt is not None
            
            try:
                ff_u, tt_u, own_u, pos_u, sub_u, eid_u, tmask_u, amask_u, act_u, logp_u, val_u, adv_u, ret_u = self.dataset.swap_out_tensor_cache()
    
                adv_u = (adv_u - adv_u.mean()) / (adv_u.std().clamp_min(1e-8))
    
                train_ds = AsyncEpisodeDataset(act_dim=self.run_cfg.env.act_dim, device=ff_u.device)
                train_ds.add_steps(
                    ff_u, tt_u, own_u, pos_u, sub_u, eid_u, tmask_u,
                    amask_u, act_u, logp_u, val_u, adv_u, ret_u
                )

    
                stats = ppo_update(
                    net=self.net,
                    opt=self.opt,
                    dataset=train_ds,
                    scheduler=self.sched,
                    **self.cfg.ppo_kwargs(),
                )
    
                self.update_idx += 1
    
                # push weights to inference actor (CPU tensors for Ray)
                sd_cpu = {k: v.detach().to("cpu") for k, v in self.net.state_dict().items()}
                self.inference_actor.set_weights.remote(sd_cpu)
    
                print(
                    f"[learner] upd={self.update_idx} "
                    f"kl={stats.approx_kl:.4f} clip={stats.clip_frac:.3f} ent={stats.entropy:.3f} "
                    f"vloss={stats.v_loss:.3f} ploss={stats.pg_loss:.3f} n_mb={stats.n_mb}"
                )
            
            except Exception as e:
                import traceback
                print(f"[learner] PPO update FAILED at update_idx={self.update_idx} with {e!r}")
                traceback.print_exception(type(e), e, e.__traceback__)

                # Optional safety: if bad batch poisons you, clear so you can recover.
                # Comment this out if you want to retry the same dataset.
                self.dataset.clear()
            
            # ---- checkpoint ----
            if self._should_save_now():
                try:
                    path = self._ckpt_path_for_update(self.update_idx)
                    self._save_checkpoint(path)
                    self._rotate_checkpoints()
                    print(f"[learner] saved checkpoint: {path}")
                except Exception as e:
                    print(f"[learner] checkpoint save failed: {e!r}")
