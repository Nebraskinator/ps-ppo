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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import RunConfig
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

    def __init__(self, cfg: RunConfig, inference_actor: ray.actor.ActorHandle, weight_store: ray.actor.ActorHandle):
        self.run_cfg = cfg
        self.cfg = cfg.learner
        self.inference_actor = inference_actor
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

    def _init_optimizer(self):
        """
        Initializes the optimizer with specialized parameter groups.
        
        Implements Weight Decay exclusion for 1D parameters (biases, LayerNorms)
        and applies specific learning rate multipliers for the Backbone vs. Heads.
        """
        if self.net is None:
            return

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
            {"params": t_decay, "lr": lr_back, "weight_decay": wd_val, "name": "trunk_wd"},
            {"params": t_no_decay, "lr": lr_back, "weight_decay": 0.0, "name": "trunk_stable"},
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

    @torch.no_grad()
    def _resuscitate_surgical(self, threshold: float = 1e-4, dampening: float = 0.001):
        """
        Revives 'dead' neurons in the network.
        
        Scans Linear and Embedding layers for neurons with near-zero weight 
        magnitudes and resets them using Kaiming initialization to restore 
        gradient flow.
        """
        logger.info(f"Initiating surgical resuscitation (threshold={threshold})")
        for name, module in self.net.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                mags = module.weight.abs().mean(dim=1)
                dead_mask = mags < threshold
                if dead_mask.any():
                    fresh = torch.empty_like(module.weight)
                    if isinstance(module, nn.Linear):
                        nn.init.kaiming_uniform_(fresh, a=np.sqrt(5))
                    else:
                        nn.init.normal_(fresh, std=0.02)
                    
                    module.weight[dead_mask] = fresh[dead_mask] * dampening
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias[dead_mask] = 0.0
                    logger.info(f"Revived {dead_mask.sum().item()} neurons in {name}")

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
            adv_chunks, ret_chunks = [], []
            curr = 0
            for length in lengths.tolist():
                end = curr + int(length)
                adv, ret = gae_from_episode(
                    rew_all[curr:end], val_all[curr:end], done_all[curr:end],
                    gamma=self.cfg.gamma, lam=self.cfg.gae_lambda
                )
                adv_chunks.append(adv)
                ret_chunks.append(ret)
                curr = end

            self.dataset.add_steps(
                obs_all, act_all, torch.from_numpy(logp_cat), val_all,
                torch.cat(adv_chunks), torch.cat(ret_chunks), torch.zeros(len(act_all))
            )
            
            self.total_episodes += len(lengths)
            self.total_steps += len(act_all)

            # Trigger Update
            if len(self.dataset) >= self.cfg.steps_per_update:
                await self._perform_update()

    async def _perform_update(self):
        """Executes a PPO update and synchronizes weights with Inference."""
        try:
            # 1. Prepare training data
            obs_u, act_u, logp_u, val_u, adv_u, ret_u, next_hp_u = self.dataset.swap_out_tensor_cache()
            
            # 2. Advantage Normalization
            adv_u = (adv_u - adv_u.mean()) / (adv_u.std() + 1e-8)
            
            train_ds = AsyncEpisodeDataset(self.run_cfg.env.act_dim, obs_u.device)
            train_ds.add_steps(obs_u, act_u, logp_u, val_u, adv_u, ret_u, next_hp_u)

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
            self.inference_actor.set_temp.remote(new_temp)

            logger.info(f"Update {self.update_idx}: Loss={stats.total_loss:.3f}, KL={stats.approx_kl:.4f}")

            # 5. Checkpointing
            if self.update_idx % self.cfg.save_every_updates == 0:
                self._save_checkpoint(self._ckpt_path_for_update(self.update_idx))
                self.inference_actor.refresh_snapshots_from_disk.remote()

        except Exception as e:
            logger.error(f"PPO Update failed: {e}")
            self.dataset.clear()

    def _save_checkpoint(self, path: str):
        """Serializes model, optimizer, and RNG states to disk."""
        payload = {
            "model": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict() if self.sched else None,
            "update_idx": self.update_idx,
            "total_steps": self.total_steps,
            "run_cfg": self.run_cfg.as_dict()
        }
        torch.save(payload, path + ".tmp")
        os.replace(path + ".tmp", path)

    def _init_if_needed(self):
        """Lazy initialization of the network and optimizer."""
        if self.net is None:
            self.net = self.run_cfg.make_model().to(self.cfg.device).train()
            self._init_optimizer()

    def _on_loop_done(self, task: asyncio.Task):
        """Safety callback to detect and log crashes in the main training loop."""
        if not task.cancelled() and task.exception():
            logger.critical("Learner training loop CRASHED!")
            traceback.print_exception(None, task.exception(), None)