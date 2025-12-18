# inference.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from ppo_core import ActorCriticTransformer, masked_sample


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
    def __init__(
        self,
        *,
        n_tokens: int,
        model_dim: int,
        n_layers: int,
        n_heads: int,
        act_dim: int,
        device: str = "cuda",
    ):
        self.n_tokens = int(n_tokens)
        self.model_dim = int(model_dim)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.act_dim = int(act_dim)

        self.device = device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("InferenceActor device=cuda but CUDA not available")

        self._token_in_dim: Optional[int] = None
        self.net: Optional[ActorCriticTransformer] = None

        self.stats = InferenceStats()
        self._pending_state: Optional[dict] = None

    def _init_net_if_needed(self, token_in_dim: int):
        if self.net is not None:
            return
        self._token_in_dim = int(token_in_dim)
        self.net = ActorCriticTransformer(
            n_tokens=self.n_tokens,
            token_in_dim=self._token_in_dim,
            act_dim=self.act_dim,
            model_dim=self.model_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
        ).to(self.device).eval()

        # if weights arrived before init
        if self._pending_state is not None:
            self.net.load_state_dict(self._pending_state)
            self._pending_state = None

    def set_weights(self, state_dict_cpu: dict):
        """
        Called by LearnerActor after updates.
        state_dict_cpu must be CPU tensors (Ray-friendly).
        """
        if self.net is None:
            self._pending_state = state_dict_cpu
            return
        self.net.load_state_dict(state_dict_cpu)

    def get_stats(self) -> dict:
        return {
            "flushes": int(self.stats.flushes),
            "total_requests": int(self.stats.total_requests),
            "total_batches": int(self.stats.total_batches),
        }

    @torch.no_grad()
    def infer_batch(
        self,
        tokens_b_np: np.ndarray,  # [B, N, D]
        tmask_b_np: np.ndarray,   # [B, N]
        amask_b_np: np.ndarray,   # [B, A]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          act   [B] int64
          logp  [B] float32
          value [B] float32
        """
        assert tokens_b_np.ndim == 3, f"tokens_b_np must be [B,N,D], got {tokens_b_np.shape}"
        assert tmask_b_np.ndim == 2, f"tmask_b_np must be [B,N], got {tmask_b_np.shape}"
        assert amask_b_np.ndim == 2, f"amask_b_np must be [B,A], got {amask_b_np.shape}"

        B = int(tokens_b_np.shape[0])
        self.stats.total_requests += B
        self.stats.total_batches += 1
        self.stats.flushes += 1

        self._init_net_if_needed(tokens_b_np.shape[-1])
        assert self.net is not None

        tokens_b = torch.from_numpy(tokens_b_np).to(self.device, dtype=torch.float32, non_blocking=True)
        tmask_b  = torch.from_numpy(tmask_b_np).to(self.device, dtype=torch.float32, non_blocking=True)
        amask_b  = torch.from_numpy(amask_b_np).to(self.device, dtype=torch.float32, non_blocking=True)

        logits, value = self.net(tokens_b, tmask_b)  # [B,A], [B]

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
