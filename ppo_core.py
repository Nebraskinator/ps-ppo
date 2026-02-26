"""
Core Neural Network and Proximal Policy Optimization (PPO) Utilities.

This module provides the PokeTransformer architecture, customized for the complex 
structured observation spaces of Pokemon Showdown, along with high-performance 
sampling, GAE calculation, and PPO update logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Iterator, Final

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Setup logger for core model events
logger = logging.getLogger(__name__)

# ----------------------------
# Numerical & Distributional Utils
# ----------------------------

def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a boolean mask to logits, setting invalid actions to a large negative value.
    
    Includes a safety valve to prevent NaN outputs if an entirely zero mask is provided
    by forcing at least one valid index.
    """
    mask_sum = mask.sum(dim=-1)
    if (mask_sum == 0).any():
        bad_indices = (mask_sum == 0).nonzero(as_tuple=True)[0]
        mask = mask.clone()
        mask[bad_indices, 0] = 1.0
        logger.warning(f"Zero mask detected at batch indices {bad_indices.tolist()}. Forced index 0.")

    m = (mask > 0.5).to(torch.bool)
    return logits.masked_fill(~m, -1e4)


def twohot_targets(x: torch.Tensor, *, v_min: float, v_max: float, v_bins: int) -> torch.Tensor:
    """
    Encodes scalar values into a two-hot distribution over uniform bins.
    
    Used for distributional value prediction to reduce variance and improve 
    learning stability in reinforcement learning.
    """
    x = x.clamp(v_min, v_max)
    scale = (v_bins - 1) / (v_max - v_min)
    f = (x - v_min) * scale
    i0 = torch.floor(f).long()
    i1 = torch.clamp(i0 + 1, max=v_bins - 1)

    w1 = (f - i0.float())
    w0 = 1.0 - w1

    # Edge case: exactly at the maximum bin
    w0 = torch.where(i0 == i1, torch.ones_like(w0), w0)
    w1 = torch.where(i0 == i1, torch.zeros_like(w1), w1)

    t = torch.zeros((x.shape[0], v_bins), device=x.device, dtype=torch.float32)
    t.scatter_add_(1, i0.view(-1, 1), w0.view(-1, 1))
    t.scatter_add_(1, i1.view(-1, 1), w1.view(-1, 1))
    return t


def dist_value_loss(v_logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """Computes cross-entropy loss between predicted value logits and target distributions."""
    logp = torch.log_softmax(v_logits, dim=-1)
    return -(target_dist * logp).sum(dim=-1).mean()


@torch.no_grad()
def masked_sample(
    logits: torch.Tensor, 
    mask: torch.Tensor, 
    greedy: bool = False, 
    temp: float = 1.0,
    top_p: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples actions using temperature scaling and optional Nucleus (Top-P) filtering.
    
    Args:
        logits: Raw action scores.
        mask: Valid action mask.
        greedy: If True, returns argmax.
        temp: Temperature (>1.0 increases entropy, <1.0 decreases it).
        top_p: Nucleus sampling threshold.
    """
    ml_pure = masked_logits(logits, mask)
    dist_pure = Categorical(logits=ml_pure)
    
    # Scale logits for exploration
    ml_explore = masked_logits(logits / max(temp, 1e-4), mask)
     
    if greedy:
        a = torch.argmax(ml_pure, dim=-1)
        return a, dist_pure.log_prob(a), torch.zeros_like(a)
    
    if 0.0 < top_p < 1.0:
        probs = torch.softmax(ml_explore, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens outside the nucleus
        sorted_to_remove = cumulative_probs > top_p
        sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
        sorted_to_remove[..., 0] = False
        
        indices_to_remove = sorted_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_to_remove)
        ml_explore = ml_explore.masked_fill(indices_to_remove, float('-inf'))
    
    dist_explore = Categorical(logits=ml_explore)
    a = dist_explore.sample()
    return a, dist_pure.log_prob(a), dist_pure.entropy()


def masked_logprob_entropy(
    logits: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates log probabilities and entropy for a specific set of actions under a mask."""
    ml = masked_logits(logits, mask)
    dist = Categorical(logits=ml)
    return dist.log_prob(actions), dist.entropy()


# ----------------------------
# Transformer Architecture
# ----------------------------

class ObservationUnpacker(nn.Module):
    """
    Slices and reshapes flat observation tensors into structured components.
    
    Expects metadata containing 'offsets' and dynamic dimensions from the 
    ObservationAssembler.
    """
    def __init__(self, meta: dict):
        super().__init__()
        self.meta = meta
        self.offsets = meta["offsets"]
        self.dim_body = meta["dim_pokemon_body"]
        self.dim_move_sc = meta["dim_move_scalars"] // 4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {name: x[:, start:end] for name, (start, end) in self.offsets.items()}
        
        # Reshape structured fields
        out["pokemon_body"] = out["pokemon_body"].reshape(-1, 12, self.dim_body)
        out["pokemon_ids"]  = out["pokemon_ids"].reshape(-1, 12, 2).to(torch.long)
        out["ability_ids"]  = out["ability_ids"].reshape(-1, 12, 4).to(torch.long)
        out["move_ids"]     = out["move_ids"].reshape(-1, 12, 4).to(torch.long)
        out["move_scalars"] = out["move_scalars"].reshape(-1, 12, 4, self.dim_move_sc)
        out["transition_move_ids"] = out["transition_move_ids"].reshape(-1, 2).to(torch.long)
        out["transition_scalars"]  = out["transition_scalars"].reshape(-1, 10)
        
        return out

class PokeTransformer(nn.Module):
    """
    Advanced Transformer-based policy and value network for Pokémon Showdown.
    
    This model uses a "Decision-Token" architecture where Actor and Critic tokens
    attend to a pool of Field and Pokémon state tokens.
    """
    def __init__(
        self,
        act_dim: int,
        meta: dict,
        emb_dims: Dict[str, int],
        out_dims: Dict[str, int],
        bank_dims: Dict[str, int],
        bank_ranges: Dict[str, int],
        n_heads: int = 8,
        n_layers: int = 4,
        ff_expansion: float = 2.0,
        v_bins: int = 51,
        dropout: float = 0.0
    ):
        super().__init__()
        self.meta = meta
        self.f_map = meta["feature_map"]
        self.d_model = out_dims["pokemon_vec"]
        self.n_pok = meta["n_pokemon_slots"]
        
        # 1. Identity Embeddings
        self.pokemon_id_emb = nn.Embedding(meta["vocab_pokemon"], emb_dims["pokemon"])
        self.item_emb       = nn.Embedding(meta["vocab_item"], emb_dims["item"])
        self.ability_emb    = nn.Embedding(meta["vocab_ability"], emb_dims["ability"])
        self.move_emb       = nn.Embedding(meta["vocab_move"], emb_dims["move"])
        
        self.val_100_emb = nn.Embedding(bank_ranges["val_100"], bank_dims["val_100"])
        self.stat_emb    = nn.Embedding(bank_ranges["stat"], bank_dims["stat"])
        self.power_emb   = nn.Embedding(bank_ranges["power"], bank_dims["power"])
        
        # 2. Subnet Configuration
        self.move_net = self._build_subnet(self._calc_move_in(emb_dims, bank_dims), out_dims["move_vec"], ff_expansion)
        self.ability_net = self._build_subnet(emb_dims["ability"] * meta["n_ability_slots"], out_dims["ability_vec"], ff_expansion)
        
        pok_in_dim = self._calc_pok_in(emb_dims, bank_dims, out_dims)
        self.pokemon_net = self._build_subnet(pok_in_dim, self.d_model, ff_expansion)

        field_in_dim = bank_dims["val_100"] + (meta["dim_global_scalars"] - 1) + (emb_dims["move"] * 2) + meta["dim_transition_scalars"]
        self.field_net = self._build_subnet(field_in_dim, self.d_model, ff_expansion)

        # 3. Transformer Block
        self.actor_tok = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.critic_tok = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads, dim_feedforward=int(self.d_model * ff_expansion),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Attention and Masking
        self.total_tokens = 2 + 1 + self.n_pok 
        self.register_buffer("attn_mask", self._build_poke_mask())
        
        # Cross-Attention Readout
        self.readout_mha = nn.MultiheadAttention(self.d_model, n_heads, dropout=dropout, batch_first=True)
        self.readout_norm_attn = nn.LayerNorm(self.d_model)
        self.readout_norm_ff = nn.LayerNorm(self.d_model)
        self.readout_net = nn.Sequential(
            nn.Linear(self.d_model, int(self.d_model * ff_expansion)),
            nn.GELU(),
            nn.Linear(int(self.d_model * ff_expansion), self.d_model)
        )

        # 5. Output Heads
        self.pi_head = nn.Linear(self.d_model, act_dim)
        self.v_head = nn.Linear(self.d_model, v_bins)
        self.unpacker = ObservationUnpacker(meta)
        self.register_buffer("v_support", torch.linspace(-1.6, 1.6, v_bins))
        
        self._reset_parameters()

    def _calc_move_in(self, emb, bank):
        m = self.f_map["move"]
        return emb["move"] + (bank["val_100"] * 2) + bank["power"] + (m["type_raw"][1] - m["onehots_raw"][0])

    def _calc_pok_in(self, emb, bank, out):
        b = self.f_map["body"]
        raw_body_slice_len = (b["boosts_raw"][1] - b["boosts_raw"][0]) + (self.meta["dim_pokemon_body"] - b["flags_raw"][0])
        return (emb["pokemon"] + emb["item"] + bank["val_100"] * 2 + bank["stat"] * 8 + 
                out["ability_vec"] + (self.meta["n_move_slots"] * out["move_vec"]) + raw_body_slice_len)

    def _build_subnet(self, in_d, out_d, ff_expansion):
        return nn.Sequential(
            nn.Linear(in_d, int(out_d * ff_expansion)), nn.GELU(),
            nn.Linear(int(out_d * ff_expansion), out_d), nn.LayerNorm(out_d)
        )

    def _build_poke_mask(self) -> torch.Tensor:
        """Creates an attention mask: State tokens cannot see Decision tokens."""
        mask = torch.zeros(self.total_tokens, self.total_tokens)
        mask[2:, 0:2] = float('-inf')  # State cannot attend to Actor/Critic
        mask[0, 1] = float('-inf')     # Actor cannot see Critic
        mask[1, 0] = float('-inf')     # Critic cannot see Actor
        return mask

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.actor_tok, std=0.02)
        nn.init.normal_(self.critic_tok, std=0.02)

    def forward(self, obs_flat: torch.Tensor, **kwargs):
        """Standard forward pass returning (action_logits, value_logits, expected_value)."""
        obs = self.unpacker(obs_flat.float())
        b_map, m_map, g_map = self.f_map["body"], self.f_map["move"], self.f_map["global"]
        B = obs_flat.shape[0]

        # 1. Feature Extraction (Moves -> Abilities -> Pokemon)
        m_sc = obs["move_scalars"]
        m_combined = torch.cat([
            self.move_emb(obs["move_ids"]),
            self.val_100_emb(m_sc[..., m_map["acc_int"]].long()),
            self.power_emb(m_sc[..., m_map["pwr_int"]].long()),
            self.val_100_emb(m_sc[..., m_map["pp_int"]].long()),
            m_sc[..., m_map["onehots_raw"][0] : m_map["type_raw"][1]]
        ], dim=-1)
        m_vecs = self.move_net(m_combined.view(B * self.n_pok * 4, -1)).view(B, self.n_pok, -1)
        a_vecs = self.ability_net(self.ability_emb(obs["ability_ids"]).view(B, self.n_pok, -1))

        p_body = obs["pokemon_body"]
        s_idx, m_idx = b_map["stats_int"], b_map["weight_int"]
        p_in = torch.cat([
            self.pokemon_id_emb(obs["pokemon_ids"][:, :, 0]),
            self.item_emb(obs["pokemon_ids"][:, :, 1]),
            self.val_100_emb(p_body[:, :, b_map["hp_int"]].long()),
            self.stat_emb(p_body[:, :, s_idx[0] : s_idx[1]].long()).flatten(2),
            self.val_100_emb(p_body[:, :, b_map["level_int"]].long()),
            self.stat_emb(p_body[:, :, m_idx : m_idx+2].long()).flatten(2),
            a_vecs, m_vecs,
            p_body[:, :, b_map["boosts_raw"][0] : b_map["boosts_raw"][1]],
            p_body[:, :, b_map["flags_raw"][0] : ]
        ], dim=-1)
        p_tokens = self.pokemon_net(p_in.view(B * self.n_pok, -1)).view(B, self.n_pok, -1)

        # 2. Field & Sequence
        turn_emb = self.val_100_emb(torch.clamp(obs["global_scalars"][:, g_map["turn_int"]] * 100, 0, 100).long())
        field_in = torch.cat([
            turn_emb, obs["global_scalars"][:, g_map["remainder_raw"][0] : ],
            self.move_emb(obs["transition_move_ids"]).view(B, -1), obs["transition_scalars"]
        ], dim=-1)
        field_token = self.field_net(field_in).unsqueeze(1)

        full_seq = torch.cat([self.actor_tok.expand(B, -1, -1), self.critic_tok.expand(B, -1, -1), field_token, p_tokens], dim=1)
        transformed = self.transformer(full_seq, mask=self.attn_mask)
        
        # 3. Readout (MHA over full sequence context)
        q = self.readout_norm_attn(transformed[:, 0:2, :])
        kv = self.readout_norm_attn(transformed)
        attended, _ = self.readout_mha(query=q, key=kv, value=kv, attn_mask=self.attn_mask[0:2, :])
        q_out = transformed[:, 0:2, :] + attended
        q_out = q_out + self.readout_net(self.readout_norm_ff(q_out))

        # 4. Heads
        v_logits = self.v_head(q_out[:, 1, :])
        v_exp = (torch.softmax(v_logits, dim=-1) * self.v_support).sum(dim=-1)
        return self.pi_head(q_out[:, 0, :]), v_logits, v_exp

# ----------------------------
# Training Components
# ----------------------------

@dataclass
class PPOUpdateStats:
    """Statistics container for a single PPO update batch."""
    approx_kl: float
    clip_frac: float
    entropy: float
    v_loss: float
    pg_loss: float
    hp_loss: float
    total_loss: float
    n_mb: int

class AsyncEpisodeDataset:
    """
    Buffer for storing and tensorizing transition data from asynchronous workers.
    """
    def __init__(self, act_dim: int, device: str):
        self.act_dim, self.device = act_dim, device
        self.clear()

    def __len__(self): return int(self.n_steps)

    def clear(self):
        self.obs, self.act, self.logp, self.val, self.adv, self.ret, self.next_hp = [], [], [], [], [], [], []
        self.n_steps = 0
        self._tensor_cache = None

    def add_steps(self, obs_td, act, logp, val, adv, ret, next_hp):
        def dev(x): return x.detach().to(self.device)
        self.obs.append(obs_td.detach().to("cpu", non_blocking=True))
        self.act.append(dev(act)); self.logp.append(dev(logp))
        self.val.append(dev(val)); self.adv.append(dev(adv))
        self.ret.append(dev(ret)); self.next_hp.append(dev(next_hp))
        self.n_steps += int(act.shape[0])

    def tensorize(self) -> Tuple:
        if self._tensor_cache: return self._tensor_cache
        self._tensor_cache = (
            torch.cat(self.obs, dim=0), torch.cat(self.act, dim=0), 
            torch.cat(self.logp, dim=0).float(), torch.cat(self.val, dim=0).float(), 
            torch.cat(self.adv, dim=0).float(), torch.cat(self.ret, dim=0).float(), 
            torch.cat(self.next_hp, dim=0).float()
        )
        return self._tensor_cache

    def iter_minibatches(self, mb_size: int) -> Iterator[Tuple]:
        obs, act, logp, val, adv, ret, next_hp = self.tensorize()
        n = act.shape[0]
        idx = torch.randperm(n, device=act.device)
        for start in range(0, (n // mb_size) * mb_size, mb_size):
            mb = idx[start : start + mb_size]
            yield obs[mb], act[mb], logp[mb], val[mb], adv[mb], ret[mb], next_hp[mb]

def ppo_update(
    net: nn.Module,
    opt: optim.Optimizer,
    dataset: AsyncEpisodeDataset,
    mode: str = "ppo",
    **cfg
) -> PPOUpdateStats:
    """
    Executes a PPO update across multiple epochs and minibatches.
    
    Includes early stopping based on Target KL to prevent policy collapse.
    """
    dev = next(net.parameters()).device
    stats = {k: 0.0 for k in ["kl", "clip", "ent", "v", "pg", "total"]}
    n_mb = 0
    
    for epoch in range(cfg.get("update_epochs", 1)):
        epoch_kl, epoch_mb = 0.0, 0
        for (mb_obs, mb_act, mb_logp_old, _, mb_adv, mb_ret, _) in dataset.iter_minibatches(cfg["minibatch_size"]):
            mb_obs, mb_act, mb_logp_old, mb_adv, mb_ret = (t.to(dev) for t in [mb_obs, mb_act, mb_logp_old, mb_adv, mb_ret])
            
            m_start, m_end = net.unpacker.offsets["action_mask"]
            mb_mask = mb_obs[:, m_start:m_end]
            
            logits, v_logits, _ = net(mb_obs)

            if mode == "imitation":
                loss = nn.functional.cross_entropy(logits, mb_act, label_smoothing=0.1)
                pg_loss = loss; v_loss = ent_loss = approx_kl = clip_frac = torch.tensor(0.0, device=dev)
            elif mode == "warmup":
                target_dist = twohot_targets(mb_ret, v_min=cfg["v_min"], v_max=cfg["v_max"], v_bins=cfg["v_bins"])
                loss = dist_value_loss(v_logits, target_dist)
                v_loss = loss; pg_loss = ent_loss = approx_kl = clip_frac = torch.tensor(0.0, device=dev)
            else:
                logp, ent = masked_logprob_entropy(logits, mb_mask, mb_act)
                ratio = (logp - mb_logp_old).exp()
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1-cfg["clip_coef"], 1+cfg["clip_coef"])).mean()
                
                target_dist = twohot_targets(mb_ret, v_min=cfg["v_min"], v_max=cfg["v_max"], v_bins=cfg["v_bins"])
                v_loss = dist_value_loss(v_logits, target_dist)
                ent_loss = ent.mean()
                
                loss = pg_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * ent_loss
                approx_kl = (mb_logp_old - logp).mean()
                clip_frac = ((ratio - 1.0).abs() > cfg["clip_coef"]).float().mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
            opt.step()

            # Stats update
            n_mb += 1; epoch_mb += 1
            stats["kl"] += approx_kl.item(); epoch_kl += approx_kl.item()
            stats["clip"] += clip_frac.item(); stats["ent"] += ent_loss.item()
            stats["v"] += v_loss.item(); stats["pg"] += pg_loss.item(); stats["total"] += loss.item()

            if mode == "ppo" and cfg.get("target_kl") and (epoch_kl / epoch_mb) > cfg["target_kl"] * 1.5:
                logger.info(f"Early stop at Epoch {epoch} due to KL {epoch_kl/epoch_mb:.4f}")
                return PPOUpdateStats(**{f"{k}_loss" if k in ["v", "pg", "hp"] else k: v/n_mb for k, v in stats.items()}, n_mb=n_mb, hp_loss=0.0)

    return PPOUpdateStats(
        approx_kl=stats["kl"]/n_mb, clip_frac=stats["clip"]/n_mb, entropy=stats["ent"]/n_mb,
        v_loss=stats["v"]/n_mb, pg_loss=stats["pg"]/n_mb, hp_loss=0.0, total_loss=stats["total"]/n_mb, n_mb=n_mb
    )