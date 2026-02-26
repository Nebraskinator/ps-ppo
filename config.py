"""
Configuration module for PokeTransformer Reinforcement Learning.

This module defines the schema and default parameters for observation processing,
neural network architecture, environment interaction, and the PPO learner.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple, Literal, Final
from pathlib import Path

# Setup logging for configuration errors
logger = logging.getLogger(__name__)

# Type Aliases
TrainingMode = Literal["imitation", "warmup", "ppo"]


@dataclass(frozen=True)
class ObsConfig:
    """Configuration for observation parsing and tokenization schema."""
    n_slots: int = 6
    n_move_slots: int = 4
    hp_bins: int = 11  # Categorical bins for HP percentage
    vocab_path: str = "vocab.json"


@dataclass(frozen=True)
class ModelConfig:
    """Parameters for the PokeTransformer/PokeNet architecture."""
    # Dedicated Embedding Sizes
    emb_dims: Dict[str, int] = field(default_factory=lambda: {
        "pokemon": 96,
        "item": 96,
        "ability": 96,
        "move": 96,
        "action": 12,
    })
    
    # Dedicated Subnet Output Sizes
    out_dims: Dict[str, int] = field(default_factory=lambda: {
        "move_vec": 128,
        "ability_vec": 128,
        "pokemon_vec": 1024,
        "global_vec": 128,
        "transition_vec": 128,
    })
    
    # Universal Embedding Bank Sizes
    bank_dims: Dict[str, int] = field(default_factory=lambda: {
        "val_100": 64,  # HP, Level, Acc, PP
        "stat": 64,     # Base Stats, Weight, Height
        "power": 64,    # Move Power
    })
    
    # Vocabulary Safety Caps
    bank_ranges: Dict[str, int] = field(default_factory=lambda: {
        "val_100": 101,
        "stat": 800,
        "power": 251,
    })
    
    dropout: float = 0.0
    n_layers: int = 2
    n_heads: int = 8
    ff_expansion: float = 4.0


@dataclass(frozen=True)
class EnvConfig:
    """Pokemon Showdown environment settings."""
    battle_format: str = "gen9randombattle"
    act_dim: int = 14  # 4 moves + 4 tera moves + 6 switches


@dataclass(frozen=True)
class RolloutConfig:
    """Settings for distributed rollout workers and batching logic."""
    target_concurrent_battles: int = 2048
    rooms_per_pair: int = 32

    # Timeouts
    infer_timeout_s: float = 15.0
    open_timeout: float = 30.0

    # Batching constraints
    infer_min_batch: int = 256
    infer_max_batch: int = 2048
    infer_wait_ms: float = 3.0
    infer_max_pending: int = 20000

    learn_min_episodes: int = 32
    learn_max_episodes: int = 256
    learn_wait_ms: float = 5.0
    learn_max_pending_episodes: int = 4096
    learn_max_pending_batches: int = 4096

    def worker_kwargs(self) -> Dict[str, Any]:
        """Returns a dictionary suitable for RolloutWorker initialization."""
        return asdict(self)


@dataclass(frozen=True)
class InferenceConfig:
    """Config for the centralized inference server/actor."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    server_min_batch: int = 0
    server_max_wait: float = 0

    def kwargs(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "server_min_batch": self.server_min_batch,
            "server_max_wait": self.server_max_wait,
        }

@dataclass(frozen=True)
class LearnerConfig:
    """Core PPO and Hyperparameter settings."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mode: TrainingMode = "ppo"
    
    # Reinforcement Learning Math
    gamma: float = 0.9999
    gae_lambda: float = 0.75
    
    # Distributional Value (Two-Hot Encoding)
    use_twohot_value: bool = True
    v_min: float = -1.6
    v_max: float = 1.6
    v_bins: int = 51
    
    # Schedules
    temp_start: float = 1.0
    temp_end: float = 0.9
    temp_total_steps: int = 500_000
    
    # Optimizer settings
    lr: float = 1e-4
    lr_warmup_steps: int = 1_000
    lr_hold_steps: int = 20_000
    lr_total_steps: int = 500_000
    weight_decay: float = 1e-2
    
    # Layer-specific LR multipliers (initialized in __post_init__)
    lr_backbone_mult: float = field(init=False)
    lr_pi_mult: float = field(init=False)
    lr_v_mult: float = field(init=False)
    
    # PPO Specifics
    update_epochs: int = 3
    minibatch_size: int = 4096
    clip_coef: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    clip_vloss: bool = False
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.02
    steps_per_update: int = 32768

    # Checkpointing
    ckpt_dir: str = "checkpoints"
    save_every_updates: int = 25
    keep_last: int = 500
    resume: bool = True
    
    # Auxiliary Tasks
    use_hp_aux: bool = False
    hp_aux_coef: float = 0.4

    def __post_init__(self):
        """Sets gradient multipliers based on the current TrainingMode."""
        # Use object.__setattr__ because the dataclass is frozen
        multipliers = {
            "imitation": (1.0, 1.0, 0.0),
            "warmup": (0.0, 0.0, 1.0),
            "ppo": (1.0, 1.0, 1.0),
        }
        backbone, pi, v = multipliers.get(self.mode, (1.0, 1.0, 1.0))
        
        object.__setattr__(self, "lr_backbone_mult", backbone)
        object.__setattr__(self, "lr_pi_mult", pi)
        object.__setattr__(self, "lr_v_mult", v)

    def get_temp(self, global_step: int) -> float:
        """Calculates linear annealing of the action temperature."""
        if global_step >= self.temp_total_steps:
            return self.temp_end
        frac = global_step / self.temp_total_steps
        return self.temp_start + frac * (self.temp_end - self.temp_start)

    def ppo_kwargs(self) -> Dict[str, Any]:
        """Extracts PPO-specific hyperparameters for the optimizer."""
        keys = ["update_epochs", "minibatch_size", "clip_coef", "ent_coef", 
                "vf_coef", "clip_vloss", "max_grad_norm", "target_kl"]
        return {k: getattr(self, k) for k in keys}


@dataclass(frozen=True)
class RewardConfig:
    """Defines the reward signal for the agent."""
    terminal_win: float = 1.0
    terminal_loss: float = -1.0
    use_faint_reward: bool = True
    faint_self: float = -0.1
    faint_opp: float = +0.1


@dataclass(frozen=True)
class RunConfig:
    """The master configuration object coordinating all sub-configs."""
    obs: ObsConfig
    model: ModelConfig
    env: EnvConfig
    rollout: RolloutConfig
    infer: InferenceConfig
    learner: LearnerConfig
    reward: RewardConfig

    @classmethod
    def default(cls) -> RunConfig:
        """Factory method for default settings."""
        return cls(
            obs=ObsConfig(),
            model=ModelConfig(),
            env=EnvConfig(),
            rollout=RolloutConfig(),
            infer=InferenceConfig(),
            learner=LearnerConfig(),
            reward=RewardConfig(),
        )

    def make_model(self) -> nn.Module:
        """
        Instantiates the PokeTransformer model based on current configuration.
        
        Returns:
            nn.Module: An initialized model ready for training or inference.
        
        Raises:
            ImportError: If required core modules are missing.
        """
        try:
            from obs_assembler import ObservationAssembler
            from ppo_core import PokeTransformer
        except ImportError as e:
            logger.error(f"Failed to import model components: {e}")
            raise

        # Derive schema metadata (Source of Truth for input sizes)
        assembler = ObservationAssembler()
        meta = assembler.get_schema_metadata()
        meta["offsets"] = assembler.offsets

        return PokeTransformer(
            act_dim=int(self.env.act_dim),
            meta=meta,
            emb_dims=self.model.emb_dims,
            out_dims=self.model.out_dims,
            bank_dims=self.model.bank_dims,
            bank_ranges=self.model.bank_ranges,
            n_heads=self.model.n_heads,
            n_layers=self.model.n_layers,
            v_bins=int(self.learner.v_bins),
            ff_expansion=self.model.ff_expansion,
            dropout=self.model.dropout,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Converts the entire configuration tree to a dictionary."""
        return asdict(self)