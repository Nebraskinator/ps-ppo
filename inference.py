# inference.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from ppo_core import ActorCriticTransformer, masked_sample
from config import RunConfig

@dataclass
class InferenceStats:
    flushes: int = 0
    total_requests: int = 0
    total_batches: int = 0


class InferenceActor:
    """
    GPU inference server.

    IMPORTANT CHANGE:
      - Workers batch locally.
      - This actor exposes infer_batch(tokens_B, tmask_B, amask_B) -> (act_B, logp_B, val_B)
      - No internal asyncio queue / background loop needed.
    """
    def __init__(self, cfg: RunConfig):
        self.cfg: RunConfig = cfg
        self.device = str(cfg.infer.device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("InferenceActor device=cuda but CUDA not available")

        self.act_dim = int(cfg.env.act_dim)
        self.obs = cfg.obs

        self.net: Optional[ActorCriticTransformer] = None
        self.stats = InferenceStats()
        self._pending_state: Optional[dict] = None

        self._init_net()

    def _init_net(self):
        if self.net is not None:
            return
    
        # Build exactly the same model as the learner expects
        self.net = self.cfg.make_model().to(self.device).eval()
    
        if self._pending_state is not None:
            self.net.load_state_dict(self._pending_state)
            self._pending_state = None

    def set_weights(self, state_dict_cpu: dict):
        if self.net is None:
            self._pending_state = state_dict_cpu
            return
        self.net.load_state_dict(state_dict_cpu, strict=True)


    def get_stats(self) -> dict:
        return {
            "flushes": int(self.stats.flushes),
            "total_requests": int(self.stats.total_requests),
            "total_batches": int(self.stats.total_batches),
        }

    @torch.no_grad()
    def infer_batch(
        self,
        ff_b_np: np.ndarray,      # [B, N, F] float32
        tt_b_np: np.ndarray,      # [B, N] int64
        own_b_np: np.ndarray,     # [B, N] int64
        pos_b_np: np.ndarray,     # [B, N] int64
        sub_b_np: np.ndarray,     # [B, N] int64
        eid_b_np: np.ndarray,     # [B, N] int64
        tmask_b_np: np.ndarray,   # [B, N] float32
        amask_b_np: np.ndarray,   # [B, A] float32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          act   [B] int64
          logp  [B] float32
          value [B] float32
        """
        assert ff_b_np.ndim == 3,   f"ff_b_np must be [B,N,F], got {ff_b_np.shape}"
        assert tt_b_np.ndim == 2,   f"tt_b_np must be [B,N], got {tt_b_np.shape}"
        assert own_b_np.ndim == 2,  f"own_b_np must be [B,N], got {own_b_np.shape}"
        assert pos_b_np.ndim == 2,  f"pos_b_np must be [B,N], got {pos_b_np.shape}"
        assert sub_b_np.ndim == 2,  f"sub_b_np must be [B,N], got {sub_b_np.shape}"
        assert eid_b_np.ndim == 2,  f"eid_b_np must be [B,N], got {eid_b_np.shape}"
        assert tmask_b_np.ndim == 2,f"tmask_b_np must be [B,N], got {tmask_b_np.shape}"
        assert amask_b_np.ndim == 2,f"amask_b_np must be [B,A], got {amask_b_np.shape}"

        B = int(ff_b_np.shape[0])
        
        N = int(ff_b_np.shape[1])
        F = int(ff_b_np.shape[2])
        
        assert N == self.obs.t_max, f"Expected N=t_max={self.obs.t_max}, got N={N}"
        assert F == self.obs.float_dim, f"Expected F=float_dim={self.obs.float_dim}, got F={F}"
        assert int(tt_b_np.shape[1]) == N
        assert int(own_b_np.shape[1]) == N
        assert int(pos_b_np.shape[1]) == N
        assert int(sub_b_np.shape[1]) == N
        assert int(eid_b_np.shape[1]) == N
        assert int(tmask_b_np.shape[1]) == N
        assert int(amask_b_np.shape[1]) == self.act_dim, f"Expected A=act_dim={self.act_dim}, got {amask_b_np.shape[1]}"
        
        self.stats.total_requests += B
        self.stats.total_batches += 1
        self.stats.flushes += 1

        assert self.net is not None

        ff_b   = torch.from_numpy(ff_b_np).to(self.device, dtype=torch.float32, non_blocking=True)
        tt_b   = torch.from_numpy(tt_b_np).to(self.device, dtype=torch.long,   non_blocking=True)
        own_b  = torch.from_numpy(own_b_np).to(self.device, dtype=torch.long,  non_blocking=True)
        pos_b  = torch.from_numpy(pos_b_np).to(self.device, dtype=torch.long,  non_blocking=True)
        sub_b  = torch.from_numpy(sub_b_np).to(self.device, dtype=torch.long,  non_blocking=True)
        eid_b  = torch.from_numpy(eid_b_np).to(self.device, dtype=torch.long,  non_blocking=True)
        tmask_b= torch.from_numpy(tmask_b_np).to(self.device, dtype=torch.float32, non_blocking=True)
        amask_b= torch.from_numpy(amask_b_np).to(self.device, dtype=torch.float32, non_blocking=True)

        logits, value = self.net(ff_b, tt_b, own_b, pos_b, sub_b, eid_b, tmask_b)  # [B,A], [B]


        bad = (amask_b.sum(dim=1) <= 0.0)
        if bad.any():
            amask_b = amask_b.clone()
            amask_b[bad] = 0.0
            amask_b[bad, 0] = 1.0

        act, logp, _ent = masked_sample(logits, amask_b)

        act_cpu   = act.to("cpu").numpy().astype(np.int64, copy=False)
        logp_cpu  = logp.to("cpu").numpy().astype(np.float32, copy=False)
        value_cpu = value.to("cpu").numpy().astype(np.float32, copy=False)
        return act_cpu, logp_cpu, value_cpu
