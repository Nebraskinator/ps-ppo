"""
Learner Actor Module for Distributed Proximal Policy Optimization (PPO).

The LearnerActor is the central training hub. It aggregates experience from 
remote workers, performs vectorized GAE (Generalized Advantage Estimation), 
executes PPO updates, and manages the lifecycle of model checkpoints and 
the self-play league.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import random
import traceback
from typing import Dict, List, Optional, Tuple, Final

import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import RunConfig
import ppo_core
from ppo_core import AsyncEpisodeDataset, gae_from_episode, ppo_update


# Configure logging
logger = logging.getLogger(__name__)

class LearnerActor:
    """
    Ray actor responsible for model optimization and weight distribution.
    
    Attributes:
        run_cfg: The master configuration (RunConfig).
        dataset: CPU-side buffer for asynchronous experience collection.
        net: The central neural network being optimized.
        opt: AdamW optimizer with layer-specific parameter groups.
    """

    def __init__(self, cfg: RunConfig, weight_store: ray.actor.ActorHandle):
        self.run_cfg = cfg
        self.cfg = cfg.learner
        self.weight_store = weight_store

        # Device verification
        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("LearnerActor configured for CUDA, but no GPU was detected.")

        self.net: Optional[nn.Module] = None
        self.opt: Optional[optim.Optimizer] = None
        self.sched: Optional[optim.lr_scheduler.LambdaLR] = None
        
        # CPU-side storage for experience; moved to GPU only during PPO minibatches
        self.dataset = AsyncEpisodeDataset(act_dim=self.run_cfg.env.act_dim, device="cpu")

        self.update_idx = 0
        self.total_episodes = 0
        self.total_steps = 0

        # Backpressure-aware queue for incoming trajectory batches
        self._q: asyncio.Queue = asyncio.Queue(
            maxsize=int(self.run_cfg.rollout.learn_max_pending_batches)
        )

        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

        # Initialize main loops
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop())
        self._task.add_done_callback(self._on_loop_done)
        self._startup_task = loop.create_task(self._maybe_resume_latest())
        
        self.ep_counter = 0

    def _init_optimizer(self):
        """
        Initializes the optimizer with specialized parameter groups.
        
        Implements Weight Decay exclusion for 1D parameters (biases, LayerNorms)
        and applies specific learning rate multipliers for the Backbone vs. Heads.
        """
        if self.net is None:
            return
        
        # Phase-Based Freezing
        for p in self.net.parameters():
            p.requires_grad = True

        if self.cfg.mode == "imitation":
            for name, p in self.net.named_parameters():
                if name.startswith("v_head.") or "critic_tok" in name:
                    p.requires_grad = False
        elif self.cfg.mode == "warmup":
            for p in self.net.parameters():
                p.requires_grad = False
            for name, p in self.net.named_parameters():
                if name.startswith("v_head.") or "critic_tok" in name:
                    p.requires_grad = True

        # 1. Topology-aware grouping
        t_decay, t_no_decay = [], []  # Transformer Trunk
        s_decay, s_no_decay = [], []  # Feature Subnets
        pi_decay, pi_no_decay = [], []  # Policy Head
        v_decay, v_no_decay = [], []  # Value Head

        wd_val = float(getattr(self.cfg, "weight_decay", 0.01))

        for name, p in self.net.named_parameters():
            if not p.requires_grad:
                continue

            # Standard Transformer Rule: No decay on LayerNorm, Bias, or Embeddings
            no_decay_condition = (
                any(x in name for x in ["_emb", "norm"]) or 
                name.endswith(".bias") or 
                p.ndim == 1
            )

            if "pi_head" in name:
                pi_no_decay.append(p) if no_decay_condition else pi_decay.append(p)
            elif "v_head" in name or "critic_tok" in name:
                v_no_decay.append(p) if no_decay_condition else v_decay.append(p)
            elif "transformer" in name or "actor_tok" in name:
                t_no_decay.append(p) if no_decay_condition else t_decay.append(p)
            else:
                s_no_decay.append(p) if no_decay_condition else s_decay.append(p)

        # 2. Assign specialized LRs per group
        base_lr = float(self.cfg.lr)
        lr_back = base_lr * self.cfg.lr_backbone_mult
        lr_pi = base_lr * self.cfg.lr_pi_mult
        lr_v = base_lr * self.cfg.lr_v_mult

        param_groups = [
            {"params": t_decay, "lr": lr_back, "weight_decay": wd_val, "name": "transformer_wd"},
            {"params": t_no_decay, "lr": lr_back, "weight_decay": 0.0, "name": "transformer_stable"},
            {"params": s_decay, "lr": lr_back, "weight_decay": wd_val, "name": "subnets_wd"},
            {"params": s_no_decay, "lr": lr_back, "weight_decay": 0.0, "name": "subnets_stable"},
            {"params": pi_decay, "lr": lr_pi, "weight_decay": wd_val, "name": "pi_wd"},
            {"params": pi_no_decay, "lr": lr_pi, "weight_decay": 0.0, "name": "pi_stable"},
            {"params": v_decay, "lr": lr_v, "weight_decay": wd_val, "name": "v_wd"},
            {"params": v_no_decay, "lr": lr_v, "weight_decay": 0.0, "name": "v_stable"},
        ]

        self.opt = optim.AdamW([pg for pg in param_groups if pg["params"]], eps=1e-5)
        
        # 3. Learning Rate Scheduler (Linear Warmup + Hold + Power Decay)
        self._init_scheduler()

    def _init_scheduler(self):
        """Sets up the LambdaLR scheduler based on configured warmup and hold steps."""
        w_steps = int(getattr(self.cfg, "lr_warmup_steps", 0))
        h_steps = int(getattr(self.cfg, "lr_hold_steps", 0))
        t_steps = int(getattr(self.cfg, "lr_total_steps", 0))

        def lr_lambda(step: int) -> float:
            if step < w_steps:
                return float(step + 1) / float(w_steps)
            if step < (w_steps + h_steps):
                return 1.0
            
            anneal_start = w_steps + h_steps
            progress = min(1.0, (step - anneal_start) / max(1, t_steps - anneal_start))
            return 1.0 / ((8 * progress + 1) ** 1.5)

        self.sched = optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)
        
    def _enable_compiled_flex_attention(self):
        """
        Compile flex_attention only inside the learner actor process.
        Safe because Ray actors are separate Python processes.
        """
        if getattr(ppo_core, "_learner_flex_compiled", False):
            return

        ppo_core.flex_attention = torch.compile(ppo_core.flex_attention)
        ppo_core._learner_flex_compiled = True
        logger.info("LearnerActor enabled compiled flex_attention")

    async def _loop(self):
        """
        The main training orchestration loop.
        
        Continually consumes packed experience batches, computes advantages, 
        and triggers PPO updates when sufficient steps are collected.
        """
        self._init_if_needed()
        
        while True:
            # item = ("packed", obs, act, logp, val, rew, done, lengths)
            msg = await self._q.get()
            if not isinstance(msg, tuple) or msg[0] != "packed":
                continue

            _, obs_cat, act_cat, logp_cat, val_cat, rew_cat, done_cat, lengths = msg
            
            # Convert to CPU Tensors
            obs_all = torch.from_numpy(obs_cat)
            act_all = torch.from_numpy(act_cat).long()
            val_all = torch.from_numpy(val_cat).float()
            rew_all = torch.from_numpy(rew_cat).float()
            done_all = torch.from_numpy(done_cat).float()

            # Vectorized GAE Calculation
            adv_chunks, ret_chunks, id_chunks = [], [], []
            curr = 0
            for length in lengths.tolist():
                end = curr + int(length)
                adv, ret = gae_from_episode(
                    rew_all[curr:end], val_all[curr:end], done_all[curr:end],
                    gamma=self.cfg.gamma, lam=self.cfg.gae_lambda
                )
                adv_chunks.append(adv)
                ret_chunks.append(ret)
                # Generate matching Episode IDs for this chunk
                id_chunks.append(torch.full((int(length),), self.ep_counter, dtype=torch.long))
                
                curr = end
                self.ep_counter += 1 # Increment unique global episode ID

            self.dataset.add_steps(
                obs_all, act_all, torch.from_numpy(logp_cat), val_all,
                torch.cat(adv_chunks), torch.cat(ret_chunks), torch.zeros(len(act_all)),
                torch.cat(id_chunks)
            )
            
            self.total_episodes += len(lengths)
            self.total_steps += len(act_all)

            # Trigger Update
            if len(self.dataset) >= self.cfg.steps_per_update:
                await self._perform_update()
                
    @staticmethod
    def _stats_1d(x: torch.Tensor, prefix: str) -> Dict[str, float]:
        """
        Basic robust stats for a 1D tensor. Always returns finite floats when possible.
        """
        if x is None:
            return {f"{prefix}_n": 0.0}
        x = x.detach()
        if x.numel() == 0:
            return {f"{prefix}_n": 0.0}
        x = x.float().view(-1)

        # guard against NaNs/Infs
        finite = torch.isfinite(x)
        if not bool(finite.all()):
            xf = x[finite]
        else:
            xf = x
        if xf.numel() == 0:
            return {
                f"{prefix}_n": float(x.numel()),
                f"{prefix}_finite": 0.0,
            }

        def q(p: float) -> float:
            # torch.quantile is fine on GPU/CPU, but convert to float at the end
            return float(torch.quantile(xf, torch.tensor(p, device=xf.device)))

        out = {
            f"{prefix}_n": float(x.numel()),
            f"{prefix}_finite": float(xf.numel()),
            f"{prefix}_mean": float(xf.mean()),
            f"{prefix}_std": float(xf.std(unbiased=False)),
            f"{prefix}_min": float(xf.min()),
            f"{prefix}_p05": q(0.05),
            f"{prefix}_p50": q(0.50),
            f"{prefix}_p95": q(0.95),
            f"{prefix}_max": float(xf.max()),
            f"{prefix}_abs_mean": float(xf.abs().mean()),
            f"{prefix}_abs_max": float(xf.abs().max()),
        }
        return out

    @staticmethod
    def _fmt(d: Dict[str, float], keys) -> str:
        parts = []
        for k in keys:
            v = d.get(k, None)
            if v is None:
                continue
            # ints encoded as float: print without decimals
            if k.endswith("_n") or k.endswith("_finite"):
                parts.append(f"{k}={int(v)}")
            else:
                parts.append(f"{k}={v:.4g}")
        return " ".join(parts)

    async def _perform_update(self):
        """Executes a PPO update and synchronizes weights with Inference."""
        try:
            # 1. Prepare training data
            obs_u, act_u, logp_u, val_u, adv_u, ret_u, next_hp_u, ep_ids_u = self.dataset.swap_out_tensor_cache()
            
            diag = {}
            diag.update(self._stats_1d(adv_u, "adv_pre"))
            diag.update(self._stats_1d(ret_u, "ret"))
            diag.update(self._stats_1d(val_u, "v_old"))
            diag.update(self._stats_1d(logp_u, "logp_old"))
            
            # 2. Advantage Normalization
            adv_u = (adv_u - adv_u.mean()) / (adv_u.std() + 1e-8)
            adv_u = torch.clamp(adv_u, -10.0, 10.0)
            
            diag["adv_norm_mean"] = float(adv_u.mean())
            diag["adv_norm_std"]  = float(adv_u.std(unbiased=False))
            
            train_ds = AsyncEpisodeDataset(self.run_cfg.env.act_dim, obs_u.device)
            train_ds.add_steps(obs_u, act_u, logp_u, val_u, adv_u, ret_u, next_hp_u, ep_ids_u)

            # 3. PPO Update Step
            stats = ppo_update(
                net=self.net, opt=self.opt, dataset=train_ds, scheduler=self.sched,
                mode=self.cfg.mode, **self.cfg.ppo_kwargs(),
                v_min=self.cfg.v_min, v_max=self.cfg.v_max, v_bins=self.cfg.v_bins
            )

            self.update_idx += 1
            
            # 4. Broadcast weights to Inference (via WeightStore)
            weights = {k: v.cpu().detach() for k, v in self.net.state_dict().items()}
            self.weight_store.update.remote(weights, self.update_idx)
            
            # Update Temperature
            new_temp = self.cfg.get_temp(self.total_steps)
            self.weight_store.set_temp.remote(new_temp)

            logger.info(f"Update {self.update_idx}: Loss={stats.total_loss:.3f}, KL={stats.approx_kl:.4f}")
            print(
                f"[learner] upd={self.update_idx} "
                f"kl={stats.approx_kl:.4f} clip={stats.clip_frac:.3f} ent={stats.entropy:.3f} "
                f"vloss={stats.v_loss:.3f} ploss={stats.pg_loss:.3f} loss={stats.total_loss:.3f} "
                f"n_mb={stats.n_mb}"
            )

            # 5. Checkpointing
            if self.update_idx % self.cfg.save_every_updates == 0:
                self._save_checkpoint(self._ckpt_path_for_update(self.update_idx))

        except Exception as e:
            logger.error(f"PPO Update failed: {e}")
            self.dataset.clear()
            
    def submit_episode(self, tokens, tmask, amask, act, logp, val, rew, done):
        """Legacy manual submission endpoint."""
        self._q.put_nowait((tokens, tmask, amask, act, logp, val, rew, done))

    async def save_now(self, path: Optional[str] = None) -> str:
        """Forces an immediate checkpoint to disk."""
        if self.net is None or self.opt is None:
            raise RuntimeError("Cannot save: model not initialized yet.")
        if path is None:
            path = self._ckpt_path_for_update(self.update_idx)
        self._save_checkpoint(path)
        return path
            
    async def submit_packed_batch(
        self,
        obs_cat: np.ndarray,
        act_cat: np.ndarray,   # [S]
        logp_cat: np.ndarray,  # [S]
        val_cat: np.ndarray,   # [S]
        rew_cat: np.ndarray,   # [S]
        done_cat: np.ndarray,  # [S]
        lengths: np.ndarray,   # [B]
    ):
        await self._q.put(("packed", obs_cat, act_cat, logp_cat, val_cat, rew_cat, done_cat, lengths))
        return True
    
    def _ckpt_path_for_update(self, update_idx: int) -> str:
        return os.path.join(self.cfg.ckpt_dir, f"learner_update_{update_idx:06d}.pt")
    
    def _save_checkpoint(self, path: str):
        """Serializes model, optimizer, and RNG states to disk."""
        payload = {
            "model": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict() if self.sched else None,
            "update_idx": self.update_idx,
            "total_steps": self.total_steps,
            "run_cfg": self.run_cfg.as_dict(),
            "total_episodes": self.total_episodes,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
        torch.save(payload, path + ".tmp")
        os.replace(path + ".tmp", path)

    def _init_if_needed(self):
        """Lazy initialization of the network and optimizer."""
        if self.net is None:
            self._enable_compiled_flex_attention()
            self.net = self.run_cfg.make_model().to(self.cfg.device).train()
            self.net.enable_bf16_recurrent_path()
            self._init_optimizer()
            
    def _latest_ckpt_path(self) -> Optional[str]:
        paths = sorted(glob.glob(os.path.join(self.cfg.ckpt_dir, "learner_update_*.pt")))
        return paths[-1] if paths else None
            
    async def _maybe_resume_latest(self):
        # Always initialize so we have *some* weights to publish.
        self._init_if_needed()
        assert self.net is not None
    
        loaded = False
        if self.cfg.resume:
            path = self._latest_ckpt_path()
            if path:
                try:
                    self._load_checkpoint(path)
                    loaded = True
                    print(f"[learner] resumed from {path}", flush=True)
                except Exception as e:
                    print(f"[learner] resume failed from {path}: {e!r}", flush=True)
    
        # IMPORTANT: publish policy 0 exactly once at startup.
        # - If resume worked: pushes resumed weights
        # - Else: pushes fresh init weights
        sd_cpu = {k: v.detach().to("cpu") for k, v in self.net.state_dict().items()}
        self.weight_store.update.remote(sd_cpu, self.update_idx)
    
        if loaded:
            print("[learner] pushed resumed policy 0 to inference", flush=True)
        else:
            print("[learner] pushed init policy 0 to inference", flush=True)

    def _on_loop_done(self, task: asyncio.Task):
        """Safety callback to detect and log crashes in the main training loop."""
        if not task.cancelled() and task.exception():
            logger.critical("Learner training loop CRASHED!")
            traceback.print_exception(None, task.exception(), None)
            
    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        self._init_if_needed()

        assert self.net is not None and self.opt is not None
        
        ckpt_mode = ckpt.get("run_cfg", {}).get("learner", {}).get("mode", "unknown")
        current_mode = self.cfg.mode
        
        should_reset_opt = (current_mode != ckpt_mode)
        model_weights = {k: v for k, v in ckpt["model"].items() if "attn_mask" not in k}
        
        # If transitioning from imitation, filter out the dead memory weights
        # strict=False allows the fresh memory weights to persist
        #if should_reset_opt and current_mode == "ppo":
        #    print(f"[learner] 🧹 Scrubbing dead memory weights from {ckpt_mode} checkpoint.")
        #    model_weights = {k: v for k, v in model_weights.items() if "mem" not in k}
        #    self.net.load_state_dict(model_weights, strict=False)
        #else:
            # FIX: We intentionally scrubbed attn_mask, so strict must be False
        self.net.load_state_dict(model_weights, strict=False)

        if should_reset_opt:
            print(f"[learner] 🛑 PHASE CHANGE DETECTED ({ckpt_mode} -> {current_mode}).")
            print(f"[learner] 🧹 SKIPPING Optimizer/Scheduler load to enforce fresh LR.")
            
            # We treat this as a fresh start, just with pre-trained weights.
            self.update_idx = 0
            self.total_episodes = 0
            self.total_steps = 0
            return 
        else:
            # Normal Resume Logic (Same Phase)
            try:
                self.opt.load_state_dict(ckpt["optimizer"])
                print("[learner] Optimizer state loaded successfully.")
            except Exception as e:
                print(f"[learner] Optimizer topology mismatch. Resetting optimizer state.")

            # 1. ALWAYS LOAD SCHEDULER FIRST
            if self.sched is not None and "scheduler" in ckpt:
                try:
                    self.sched.load_state_dict(ckpt["scheduler"])
                    print("[learner] Scheduler state loaded successfully.")
                except:
                    pass

            # 2. APPLY OVERRIDE AFTERWARD
            try:
                new_base_lr = float(self.cfg.lr)
                mults = {
                    "pi": float(getattr(self.cfg, "lr_pi_mult", 1.0)),
                    "v": float(getattr(self.cfg, "lr_v_mult", 1.0)),
                    "backbone": float(getattr(self.cfg, "lr_backbone_mult", 1.0))
                }

                for pg in self.opt.param_groups:
                    group_name = pg.get("name", "")
                    if group_name.startswith("pi"):
                        target_lr = new_base_lr * mults["pi"]
                    elif group_name.startswith("v"):
                        target_lr = new_base_lr * mults["v"]
                    elif group_name.startswith("transformer") or group_name.startswith("subnets"):
                        target_lr = new_base_lr * mults["backbone"]
                    else:
                        continue
                    
                    pg["lr"] = target_lr
                    pg["initial_lr"] = target_lr  # Updates the baseline for the LambdaLR
                
                # 3. CRITICAL: Force the scheduler to accept the new baselines
                if self.sched is not None:
                    self.sched.base_lrs = [pg["initial_lr"] for pg in self.opt.param_groups]
                    
                try:
                    new_wd = float(getattr(self.cfg, "weight_decay", 0.01))
                    for pg in self.opt.param_groups:
                        # Only apply to groups that were originally intended to have decay
                        if pg["weight_decay"] > 0:
                            pg["weight_decay"] = new_wd
                    print(f"[learner] ⚡ Weight Decay Override: Updated to {new_wd}")
                except Exception as e:
                    print(f"[learner] Weight Decay override failed: {e!r}")
                
                print(f"[learner] ⚡ LR Jump: Backbone/Subnets={new_base_lr*mults['backbone']:.2g}, Pi={new_base_lr*mults['pi']:.2g}, V={new_base_lr*mults['v']:.2g}")
            except Exception as e:
                print(f"[learner] LR Jump override failed: {e!r}")

        self.update_idx = int(ckpt.get("update_idx", 0))
        self.total_episodes = int(ckpt.get("total_episodes", 0))
        self.total_steps = int(ckpt.get("total_steps", 0))
        
        # Push resumed temperature to the WeightStore
        new_temp = self.cfg.get_temp(self.total_steps)
        self.weight_store.set_temp.remote(new_temp)

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
            
    async def get_stats(self) -> dict:
        return {
            "update": self.update_idx,
            "episodes": self.total_episodes,
            "steps_in_dataset": len(self.dataset),
            "total_steps": self.total_steps,
        }