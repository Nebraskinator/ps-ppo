# config.py

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional
import torch


# -------------------------
# Observation / token schema
# -------------------------

@dataclass(frozen=True)
class ObsConfig:
    # -------------------------
    # Token length / float schema
    # -------------------------
    t_max: int = 128
    float_dim: int = 112

    # ---- Core flags
    F_PRESENT: int = 0
    F_KNOWN: int = 1
    F_FAINTED: int = 2

    # ---- New: HP as 20-bin one-hot (3..22)
    F_HP_BIN0: int = 3          # 20 dims: 3..22 inclusive

    # ---- Boosts (7 dims)
    F_BOOST_ATK: int = 23
    F_BOOST_DEF: int = 24
    F_BOOST_SPA: int = 25
    F_BOOST_SPD: int = 26
    F_BOOST_SPE: int = 27
    F_BOOST_ACC: int = 28
    F_BOOST_EVAS: int = 29

    # ---- Move scalars
    F_BP: int = 30
    F_ACC: int = 31
    F_PRIO: int = 32
    F_PP_FRAC: int = 33

    # ---- Types
    F_POK_TYPE_MH0: int = 34          # 34..51 (18 dims)
    F_POK_TERA_TYPE_MH0: int = 52     # 52..69 (18 dims)
    F_MOVE_TYPE_MH0: int = 70         # 70..87 (18 dims)
    F_MOVE_CAT_OH0: int = 88          # 88..90 (3 dims)

    # ---- Combat derived
    F_STAB: int = 91
    F_EFF_LOG2: int = 92
    F_EFF_UNKNOWN: int = 93

    # ---- Effect details
    F_STAGE_NORM: int = 94
    F_TURNS_NORM: int = 95
    F_COUNTER_NORM: int = 96

    # ---- active flag
    F_IS_ACTIVE: int = 97

    # ---- Optional unknown flags (keep if you still want them)
    F_POK_TYPE_UNKNOWN: int = 98
    F_MOVE_TYPE_UNKNOWN: int = 99
    F_CAN_TERA: int = 100
    F_IS_TERA: int = 101

    # 100..111 reserved for future use (12 dims of slack)

    EFF_MAX_LAYERS: float = 3.0
    EFF_MAX_TURNS: float = 8.0
    EFF_MAX_COUNTER: float = 10.0
    # -------------------------
    # Closed-vocab entity table sizes (fixed hash/id buckets)
    # -------------------------
    max_species: int = 1500
    max_moves: int = 1000
    max_items: int = 600
    max_abilities: int = 350

    # Hashed bins for open vocab
    h_effect: int = 120

    # -------------------------
    # Token type vocab
    # -------------------------
    token_type_vocab: tuple[str, ...] = (
        "cls",
        "pokemon",
        "move",
        "item",
        "ability",
        "battlefield",
        "effect",
        "side_condition",
        "field",
    )
    _tt_map: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_tt_map", {name: i for i, name in enumerate(self.token_type_vocab)})

    # Derived: categorical vocab sizes
    @property
    def n_tok_types(self) -> int:
        return len(self.token_type_vocab)
    
    def tt(self, name: str) -> int:
        i = self._tt_map.get(name, None)
        if i is None:
            raise KeyError(f"Unknown token type name: {name!r}. Valid: {self.token_type_vocab}")
        return int(i)

    # Token type IDs (derived from order above)
    @property
    def TT_CLS(self) -> int: return self.tt("cls")
    @property
    def TT_POK(self) -> int: return self.tt("pokemon")
    @property
    def TT_MOVE(self) -> int: return self.tt("move")
    @property
    def TT_ITEM(self) -> int: return self.tt("item")
    @property
    def TT_ABILITY(self) -> int: return self.tt("ability")
    @property
    def TT_BF(self) -> int: return self.tt("battlefield")
    @property
    def TT_EFF(self) -> int: return self.tt("effect")
    @property
    def TT_SC(self) -> int: return self.tt("side_condition")
    @property
    def TT_FIELD(self) -> int: return self.tt("field")


    # -------------------------
    # Owners / positions
    # -------------------------
    OWNER_OPP: int = 0
    OWNER_SELF: int = 1
    OWNER_NONE: int = 2
    n_owner: int = 3

    n_slots: int = 6
    POS_NA: int = 6           # 0..5 valid, 6 means NA
    n_pos: int = 7            # 0..5 + POS_NA

    n_move_slots: int = 4
    SUBPOS_NA: int = 4        # 0..3 valid, 4 means NA
    n_subpos: int = 5         # 0..3 + SUBPOS_NA

    # -------------------------
    # Shared entity table layout
    # -------------------------
    ENTITY_NONE: int = 0
    
    @property
    def SPECIES_UNK(self) -> int:
        return self.SPECIES_OFFSET
    
    @property
    def MOVES_UNK(self) -> int:
        return self.MOVES_OFFSET
    
    @property
    def ITEMS_UNK(self) -> int:
        return self.ITEMS_OFFSET
    
    @property
    def ABIL_UNK(self) -> int:
        return self.ABIL_OFFSET
    
    @property
    def EFFECT_UNK(self) -> int:
        return self.EFFECT_OFFSET


    @property
    def SPECIES_OFFSET(self) -> int:
        return 1  # reserve 0 for ENTITY_NONE globally
    
    @property
    def MOVES_OFFSET(self) -> int:
        return self.SPECIES_OFFSET + (1 + self.max_species)
    
    @property
    def ITEMS_OFFSET(self) -> int:
        return self.MOVES_OFFSET + (1 + self.max_moves)
    
    @property
    def ABIL_OFFSET(self) -> int:
        return self.ITEMS_OFFSET + (1 + self.max_items)
    
    @property
    def EFFECT_OFFSET(self) -> int:
        return self.ABIL_OFFSET + (1 + self.max_abilities)
    
    @property
    def n_entity(self) -> int:
        # +1 for UNK, +2*h_effect for (known effects + hashed effects)
        return self.EFFECT_OFFSET + (1 + 2 * self.h_effect)
    


    def model_kwargs(self) -> Dict[str, Any]:
        return dict(
            t_max=self.t_max,
            float_dim=self.float_dim,
            n_tok_types=self.n_tok_types,
            n_owner=self.n_owner,
            n_pos=self.n_pos,
            n_subpos=self.n_subpos,
            n_entity=self.n_entity,
        )



# -------------------------
# Model config
# -------------------------
@dataclass(frozen=True)
class ModelConfig:
    model_dim: int = 64
    n_layers: int = 4
    n_heads: int = 1
    ff_mult: int = 4
    dropout: float = 0.0

    def kwargs(self) -> Dict[str, Any]:
        return dict(
            model_dim=self.model_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            ff_mult=self.ff_mult,
            dropout=self.dropout,
        )


# -------------------------
# Environment / action space
# -------------------------
@dataclass(frozen=True)
class EnvConfig:
    battle_format: str = "gen9randombattle"
    act_dim: int = 14  # 4 moves + 4 tera moves + 6 switches


# -------------------------
# Rollout worker batching + timeouts
# -------------------------
@dataclass(frozen=True)
class RolloutConfig:
    
    target_concurrent_battles: int = 3072
    rooms_per_pair: int = 16

    infer_timeout_s: float = 15.0
    open_timeout: float = 30.0
    ping_interval: float = 20.0
    ping_timeout: float = 20.0

    infer_min_batch: int = 32
    infer_max_batch: int = 1024
    infer_wait_ms: float = 1.0
    infer_max_pending: int = 20000

    learn_min_episodes: int = 32
    learn_max_episodes: int = 256
    learn_wait_ms: float = 5.0
    learn_max_pending_episodes: int = 2000

    def worker_kwargs(self) -> Dict[str, Any]:
        # pass straight into your RolloutWorker/RayBatchedPlayer cfg
        return asdict(self)


# -------------------------
# Inference actor config
# -------------------------
@dataclass(frozen=True)
class InferenceConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def kwargs(self) -> Dict[str, Any]:
        return dict(device=self.device)



# -------------------------
# Learner / PPO config (reuse your PPOConfig if you want)
# -------------------------
@dataclass(frozen=True)
class LearnerConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 3072
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip_vloss: bool = True
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.02

    # update trigger
    steps_per_update: int = 32768

    # checkpointing
    ckpt_dir: str = "checkpoints"
    save_every_updates: int = 25
    keep_last: int = 500
    resume: bool = True

    def ppo_kwargs(self) -> Dict[str, Any]:
        return dict(
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            clip_vloss=self.clip_vloss,
            max_grad_norm=self.max_grad_norm,
            target_kl=self.target_kl,
        )


# -------------------------
# Top-level run config
# -------------------------
@dataclass(frozen=True)
class RunConfig:
    obs: ObsConfig
    model: ModelConfig
    env: EnvConfig
    rollout: RolloutConfig
    infer: InferenceConfig
    learner: LearnerConfig

    @staticmethod
    def default() -> "RunConfig":
        return RunConfig(
            obs=ObsConfig(),
            model=ModelConfig(),
            env=EnvConfig(),
            rollout=RolloutConfig(),
            infer=InferenceConfig(),
            learner=LearnerConfig(),
        )

    # ----- factories / adapters -----

    def make_model(self):
        # centralized place to build ActorCriticTransformer
        from ppo_core import ActorCriticTransformer
        return ActorCriticTransformer(
            act_dim=self.env.act_dim,
            **self.model.kwargs(),
            **self.obs.model_kwargs(),
        )

    def inference_actor_kwargs(self) -> Dict[str, Any]:
        # kwargs for InferenceActor(...)
        return dict(
            act_dim=self.env.act_dim,
            **self.model.kwargs(),
            **self.obs.model_kwargs(),
            **self.infer.kwargs(),
        )

    def rollout_worker_kwargs(self) -> Dict[str, Any]:
        # kwargs used when constructing RolloutWorker(cfg=..., ...)
        return dict(
            battle_format=self.env.battle_format,
            act_dim=self.env.act_dim,
            **self.rollout.worker_kwargs(),
        )
    
    def learner_actor_kwargs(self) -> Dict[str, Any]:
        # kwargs for LearnerActor(cfg=..., inference_actor=...)
        # You’ll likely pass the whole RunConfig anyway; this is here if you prefer slices.
        return dict()


    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
