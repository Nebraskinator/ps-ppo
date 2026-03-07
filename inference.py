"""
Inference Server and Model Management for Distributed RL.

This module provides the InferenceActor, which handles large-batch GPU forward 
passes
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Final

import numpy as np
import ray
import torch
from config import RunConfig
from ppo_core import masked_sample

# Setup logging
logger = logging.getLogger(__name__)

# Constants
CKPT_UPDATE_PATTERN: Final[re.Pattern] = re.compile(r"learner_update_(\d+)\.pt$")

# -----------------------------------------------------------------------------
# CHECKPOINT UTILITIES
# -----------------------------------------------------------------------------

def _extract_state_dict(ckpt: dict) -> dict:
    """
    Surgically extracts the model state dictionary from various checkpoint formats.
    
    Args:
        ckpt: A dictionary loaded via torch.load.
        
    Returns:
        dict: The raw state_dict containing model parameters.
        
    Raises:
        RuntimeError: If no valid state_dict structure is detected.
    """
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Expected dict from checkpoint, got {type(ckpt)}")

    # Check common wrapper keys used in RL frameworks
    for k in ("model", "state_dict", "net", "policy", "actor_critic"):
        v = ckpt.get(k)
        if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
            return v
    
    # If no wrapper, check if the dict itself is the state_dict
    if any(isinstance(x, torch.Tensor) for x in ckpt.values()):
        return ckpt
        
    raise RuntimeError("Could not find a model state_dict inside the checkpoint file.")
@dataclass
class InferenceStats:
    """Container for monitoring inference performance and throughput."""
    flushes: int = 0
    total_requests: int = 0
    avg_batch_size: float = 0.0

# -----------------------------------------------------------------------------
# ACTORS
# -----------------------------------------------------------------------------

@ray.remote
class WeightStore:
    """
    A centralized Ray actor that serves as the latest 'truth' for model weights.
    Workers pull from here to synchronize their local GPU models.
    """
    def __init__(self):
        self.weights: Optional[dict] = None
        self.version: int = -1

    def update(self, weights: dict, version: int):
        self.weights = weights
        self.version = version

    def get_state(self) -> Tuple[Optional[dict], int]:
        return self.weights, self.version


class InferenceActor:
    """
    High-throughput inference server responsible for GPU kernel execution.
    
    Supports 'Snapshot Inference'—running multiple different model versions 
    within the same batch by grouping observations by their assigned policy_id.
    """
    def __init__(self, cfg: RunConfig, weight_store: ray.actor.ActorHandle):
        self.cfg = cfg
        self.weight_store = weight_store
        self.device: Final[str] = str(cfg.infer.device)
        
        self.current_version = -1
        self.current_temp = cfg.learner.temp_start
        self.stats = InferenceStats()
        
        # Build main model and league models
        self.net = self.cfg.make_model().to(self.device).eval()
        
        # Initial load
        self.resume_from_disk()

    @torch.no_grad()
    def infer_batch(self, policy_ids_np: np.ndarray, obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes a batch forward pass for a single policy, ignoring legacy policy IDs.
        """
        self._sync_weights()
        
        B = obs_np.shape[0]
        self.stats.flushes += 1
        self.stats.total_requests += B
        self.stats.avg_batch_size = (self.stats.avg_batch_size * 0.95) + (B * 0.05)

        obs = torch.from_numpy(obs_np).to(self.device)

        # Single forward pass for the entire batch
        logits, _, value = self.net(obs)
        
        # Extract Action Mask
        m_start, m_end = self.net.unpacker.offsets["action_mask"]
        mask_sub = obs[:, m_start:m_end].float()
        
        # Sample actions
        act, logp, _ = masked_sample(logits, mask_sub, temp=self.current_temp)

        # Sync before returning to CPU
        torch.cuda.synchronize()
        
        return (
            act.cpu().numpy(),
            logp.cpu().numpy(),
            value.cpu().numpy()
        )
    
    def set_temp(self, temp: float):
        self.current_temp = temp

    def _sync_weights(self):
        """Pulls updated weights from the WeightStore if a new version is available."""
        # Use a lightweight version check before pulling the heavy state_dict
        latest_v = ray.get(self.weight_store.get_state.remote())[1]
        
        if latest_v > self.current_version:
            weights, version = ray.get(self.weight_store.get_state.remote())
            if weights:
                self.net.load_state_dict(weights, strict=True)
                self.current_version = version
                logger.info(f"InferenceActor synced to version {version}")

    def resume_from_disk(self):
        """Initializes the actor with the latest weights found in the checkpoint directory."""
        if not getattr(self.cfg.learner, "resume", False):
            return

        ckpt_dir = self.cfg.learner.ckpt_dir
        if not os.path.exists(ckpt_dir):
            return

        # Find all checkpoints and load the most recent one
        import glob
        paths = sorted(glob.glob(os.path.join(ckpt_dir, "learner_update_*.pt")))
        
        if not paths:
            return
            
        latest_ckpt = paths[-1]
        try:
            st = _extract_state_dict(torch.load(latest_ckpt, map_location="cpu"))
            self.net.load_state_dict(st)
            logger.info(f"InferenceActor resumed from {latest_ckpt}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest_ckpt}: {e}")
            
    def refresh_snapshots_from_disk(self):
        """
        Legacy endpoint for the LearnerActor. 
        In a single-policy setup, dynamic disk reloading is unnecessary because 
        the latest weights are synced continuously via RAM (WeightStore).
        """
        pass
                
    def get_stats(self) -> dict:
        """Returns diagnostic metrics."""
        return {
            "total_requests": self.stats.total_requests,
            "avg_batch_size": round(self.stats.avg_batch_size, 2),
            "model_version": self.current_version
        }