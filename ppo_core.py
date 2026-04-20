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
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
# Setup logger for core model events
logger = logging.getLogger(__name__)

# ----------------------------
# Numerical & Distributional Utils
# ----------------------------

TOKENS_PER_TURN = 15
FIELD_IDX = 0
ACTOR_IDX = 13
CRITIC_IDX = 14

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
        return a, dist_pure.log_prob(a), torch.zeros_like(a, dtype=torch.float32)
    
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

def logit_soft_cap(score, b, h, q_idx, kv_idx):
    """
    Applies soft-clipping to attention logits: 
    C * tanh(score / C)
    """
    cap = 6.0
    return torch.tanh(score / cap) * cap

# ----------------------------------------
# THE MASKS
# ----------------------------------------

def pokemon_episodic_mask(b, h, q_idx, kv_idx, episode_ids, n_prev_fields: int):
    same_ep = episode_ids[q_idx] == episode_ids[kv_idx]

    q_t = q_idx // TOKENS_PER_TURN
    kv_t = kv_idx // TOKENS_PER_TURN
    kv_i = kv_idx % TOKENS_PER_TURN

    same_turn = (q_t == kv_t)
    prev_field = (
        (kv_i == FIELD_IDX)
        & (kv_t < q_t)
        & ((q_t - kv_t) <= n_prev_fields)
    )

    return same_ep & (same_turn | prev_field)

def pokemon_batched_episodic_mask(b, h, q_idx, kv_idx, expanded_ids, n_prev_fields: int):
    same_ep = expanded_ids[b, q_idx] == expanded_ids[b, kv_idx]

    q_t = q_idx // TOKENS_PER_TURN
    kv_t = kv_idx // TOKENS_PER_TURN
    kv_i = kv_idx % TOKENS_PER_TURN

    same_turn = (q_t == kv_t)
    prev_field = (
        (kv_i == FIELD_IDX)
        & (kv_t < q_t)
        & ((q_t - kv_t) <= n_prev_fields)
    )

    return same_ep & (same_turn | prev_field)

def pokemon_inference_mask(b, h, q_idx, kv_idx, hist_len: torch.Tensor, n_prev_fields: int):
    q_is_current = q_idx < TOKENS_PER_TURN
    kv_is_current = kv_idx < TOKENS_PER_TURN

    kv_is_hist = (kv_idx >= TOKENS_PER_TURN) & (kv_idx < TOKENS_PER_TURN + n_prev_fields)
    hist_slot = kv_idx - TOKENS_PER_TURN

    # Right-aligned valid history:
    # valid slots are [K - hist_len[b], ..., K-1]
    kv_hist_valid = kv_is_hist & (hist_slot >= (n_prev_fields - hist_len[b]))

    # Current queries can read current tokens + valid history fields.
    current_rule = q_is_current & (kv_is_current | kv_hist_valid)

    # History queries are inert/self-only. This also keeps padded queries safe.
    history_rule = (q_idx >= TOKENS_PER_TURN) & (q_idx == kv_idx)

    return current_rule | history_rule

# ----------------------------------------
# 2. FLEX MODULES (Hardware Optimized)
# ----------------------------------------

class FlexEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_expansion=2.0, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model * ff_expansion)),
            nn.GELU(),
            nn.Linear(int(d_model * ff_expansion), d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, block_mask):
        res = x
        x = self.norm1(x)
        B, S, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, S, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        # Triton-optimized kernel call
        attn_out = flex_attention(
            qkv[0], qkv[1], qkv[2],
            score_mod=logit_soft_cap,
            block_mask=block_mask,
        )
        x = res + self.dropout(self.out_proj(attn_out.transpose(1, 2).reshape(B, S, D)))
        return x + self.ff(self.norm2(x))

class FlexReadout(nn.Module):
    """
    Per-turn readout attention.

    Query tokens:
      0 = actor
      1 = critic

    Key/value tokens:
      full 15-token current turn
    """
    def __init__(self, d_model, n_heads, ff_expansion=2.0, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model * ff_expansion)),
            nn.GELU(),
            nn.Linear(int(d_model * ff_expansion), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor) -> torch.Tensor:
        """
        q_x:  [B, 2, D]   actor/critic query tokens
        kv_x: [B, 15, D]  full current-turn tokens
        """
        B, Q, D = q_x.shape
        K = kv_x.shape[1]

        q = self.q_proj(self.norm_q(q_x)).view(B, Q, self.n_heads, -1).transpose(1, 2)
        k = self.k_proj(self.norm_kv(kv_x)).view(B, K, self.n_heads, -1).transpose(1, 2)
        v = self.v_proj(self.norm_kv(kv_x)).view(B, K, self.n_heads, -1).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        q_out = q_x + self.out_proj(attn_out.transpose(1, 2).reshape(B, Q, D))
        return q_out + self.ff(self.norm_ff(q_out))

# ----------------------------------------
# 3. POKE TRANSFORMER
# ----------------------------------------

class PokeTransformer(nn.Module):
    def __init__(self, act_dim, meta, emb_dims, out_dims, bank_dims, bank_ranges, n_heads=8, n_layers=4, **kwargs):
        super().__init__()
        self.meta, self.f_map = meta, meta["feature_map"]
        self.d_model = out_dims["pokemon_vec"]
        self.n_heads, self.n_layers = n_heads, n_layers
        
        self.kv_cache_len = kwargs.get("kv_cache_len", 64)
        
        # Identity Embeddings
        self.pokemon_id_emb = nn.Embedding(meta["vocab_pokemon"], emb_dims["pokemon"])
        self.item_emb = nn.Embedding(meta["vocab_item"], emb_dims["item"])
        self.ability_emb = nn.Embedding(meta["vocab_ability"], emb_dims["ability"])
        self.move_emb = nn.Embedding(meta["vocab_move"], emb_dims["move"])
        self.val_100_emb = nn.Embedding(bank_ranges["val_100"], bank_dims["val_100"])
        self.stat_emb = nn.Embedding(bank_ranges["stat"], bank_dims["stat"])
        self.power_emb = nn.Embedding(bank_ranges["power"], bank_dims["power"])
        
        # Subnets
        self.move_net = self._build_subnet(self._calc_move_in(emb_dims, bank_dims), out_dims["move_vec"])
        self.ability_net = self._build_subnet(emb_dims["ability"] * meta["n_ability_slots"], out_dims["ability_vec"])
        self.pokemon_net = self._build_subnet(self._calc_pok_in(emb_dims, bank_dims, out_dims), self.d_model)
        self.field_net = self._build_subnet(
                bank_dims["power"]
                + (meta["dim_global_scalars"] - 1)
                + (emb_dims["move"] * 2)
                + (emb_dims["pokemon"] * 4)
                + (emb_dims["ability"] * 2)
                + (emb_dims["item"] * 2)
                + meta["dim_transition_scalars"],
                self.d_model,
            )
        # Decision Tokens
        self.actor_tok = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.critic_tok = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        self.transformer = nn.ModuleList([FlexEncoderLayer(self.d_model, n_heads) for _ in range(n_layers)])
        self.readout = FlexReadout(self.d_model, n_heads)

        self.pi_head = nn.Linear(self.d_model, act_dim)
        self.v_head = nn.Linear(self.d_model, kwargs.get("v_bins", 51))
        self.unpacker = ObservationUnpacker(meta)
        self.register_buffer("v_support", torch.linspace(-1.6, 1.6, kwargs.get("v_bins", 51)))
        self._reset_parameters()

    def _build_subnet(self, in_d, out_d):
        return nn.Sequential(nn.Linear(in_d, out_d * 2), nn.GELU(), nn.Linear(out_d * 2, out_d), nn.LayerNorm(out_d))

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): 
                nn.init.normal_(m.weight, std=0.02)
                if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
        # Depth-scaled initialization for residual projections
        # Scale by 1 / sqrt(2 * n_layers) because there are 2 residual additions per layer (Attn + FFN)
        std_proj = 0.02 / (2 * self.n_layers) ** 0.5
        
        for layer in self.transformer:
            # Scale the Attention output projection
            nn.init.normal_(layer.out_proj.weight, std=std_proj)
            # Scale the final FFN linear layer (which is at index 2 in your nn.Sequential)
            nn.init.normal_(layer.ff[2].weight, std=std_proj)
            
        # Also scale the Readout module's residual projections
        nn.init.normal_(self.readout.out_proj.weight, std=std_proj)
        nn.init.normal_(self.readout.ff[2].weight, std=std_proj)

    def _calc_move_in(self, emb, bank):
        m = self.f_map["move"]
        return emb["move"] + (bank["val_100"] * 2) + bank["power"] + (m["target_raw"][1] - m["onehots_raw"][0])

    def _calc_pok_in(self, emb, bank, out):
        b = self.f_map["body"]
        raw_body_slice_len = (b["boosts_raw"][1] - b["boosts_raw"][0]) + (self.meta["dim_pokemon_body"] - b["flags_raw"][0])
        return (emb["pokemon"] + emb["item"] + bank["val_100"] * 2 + bank["stat"] * 8 + out["ability_vec"] + (4 * out["move_vec"]) + raw_body_slice_len)
    
    def enable_bf16_recurrent_path(self):
        bf16_modules = [
            self.pokemon_id_emb,
            self.item_emb,
            self.ability_emb,
            self.move_emb,
            self.val_100_emb,
            self.stat_emb,
            self.power_emb,
            self.move_net,
            self.ability_net,
            self.pokemon_net,
            self.field_net,
            self.transformer,
            self.readout,
            self.pi_head,
            self.v_head,
        ]
    
        for mod in bf16_modules:
            mod.bfloat16()
    
        self.actor_tok.data = self.actor_tok.data.bfloat16()
        self.critic_tok.data = self.critic_tok.data.bfloat16()
    
    def recurrent_dtype(self) -> torch.dtype:
        return self.actor_tok.dtype
        
    def encode_features(self, obs_flat):
        """Vectorized feature extraction for Field and Pokemon."""
        obs = self.unpacker(obs_flat)
        B = obs_flat.shape[0]
        rec_dtype = self.recurrent_dtype()
        
        # Moves -> Ability -> Pokemon
        m_sc = obs["move_scalars"]
        m_combined = torch.cat([
            self.move_emb(obs["move_ids"]),
            self.val_100_emb(m_sc[..., self.f_map["move"]["acc_int"]].long()),
            self.power_emb(m_sc[..., self.f_map["move"]["pwr_int"]].long()),
            self.val_100_emb(m_sc[..., self.f_map["move"]["pp_int"]].long()),
            # UPDATED: Replaced type_raw with target_raw to include Category and Target
            m_sc[..., self.f_map["move"]["onehots_raw"][0] : self.f_map["move"]["target_raw"][1]].to(rec_dtype)
        ], dim=-1)
        m_vecs = self.move_net(m_combined.view(-1, m_combined.shape[-1])).view(B, 12, -1)
        a_vecs = self.ability_net(self.ability_emb(obs["ability_ids"]).view(B, 12, -1))

        p_body = obs["pokemon_body"]
        p_in = torch.cat([
            self.pokemon_id_emb(obs["pokemon_ids"][:, :, 0]),
            self.item_emb(obs["pokemon_ids"][:, :, 1]),
            self.val_100_emb(p_body[:, :, self.f_map["body"]["hp_int"]].long()),
            self.stat_emb(p_body[:, :, self.f_map["body"]["stats_int"][0] : self.f_map["body"]["stats_int"][1]].long()).flatten(2),
            self.val_100_emb(p_body[:, :, self.f_map["body"]["level_int"]].long()),
            self.stat_emb(p_body[:, :, self.f_map["body"]["weight_int"] : self.f_map["body"]["weight_int"]+2].long()).flatten(2),
            a_vecs, m_vecs,
            p_body[:, :, self.f_map["body"]["boosts_raw"][0] : self.f_map["body"]["boosts_raw"][1]].to(rec_dtype),
            p_body[:, :, self.f_map["body"]["flags_raw"][0] : ].to(rec_dtype)
        ], dim=-1)
        p_tokens = self.pokemon_net(p_in.view(-1, p_in.shape[-1])).view(B, 12, -1)

        field_in = torch.cat([
            self.power_emb(
                obs["global_scalars"][:, self.f_map["global"]["turn_int"]].long()
            ),
            obs["global_scalars"][:, self.f_map["global"]["remainder_raw"][0]:].to(rec_dtype),
        
            self.move_emb(obs["transition_move_ids"]).view(B, -1),
            self.pokemon_id_emb(obs["transition_pokemon_ids"]).view(B, -1),
            self.ability_emb(obs["transition_ability_ids"]).view(B, -1),
            self.item_emb(obs["transition_item_ids"]).view(B, -1),
        
            obs["transition_scalars"].to(rec_dtype),
        ], dim=-1)
        field_token = self.field_net(field_in).unsqueeze(1)
        return field_token, p_tokens

    def forward(self, obs_flat: torch.Tensor, 
                episode_ids: torch.Tensor = None, 
                kv_cache: list[torch.Tensor] = None,
                kv_cache_len: torch.Tensor = None,
                batch_seq_len: int = 0,):
        B = obs_flat.shape[0]
        device = obs_flat.device
        K = self.kv_cache_len
        
        field_tok, p_toks = self.encode_features(obs_flat)
        
        current_turn = torch.cat([
                field_tok,                          # 0
                p_toks,                             # 1-12
                self.actor_tok.expand(B, 1, -1),    # 13
                self.critic_tok.expand(B, 1, -1),   # 14
            ], dim=1)  # [B, 15, D]

        # --- PACKED TRAINING MODE ---
        if episode_ids is not None:
            n_seq = B // batch_seq_len
            s_tok = batch_seq_len * TOKENS_PER_TURN
            x = current_turn.view(n_seq, batch_seq_len, TOKENS_PER_TURN, self.d_model).reshape(n_seq, s_tok, self.d_model)
            expanded_ids = episode_ids.view(n_seq, batch_seq_len, 1).expand(-1, -1, TOKENS_PER_TURN).reshape(n_seq, s_tok)

            b_mask = create_block_mask(
                lambda b, h, q, k: pokemon_batched_episodic_mask(b, h, q, k, expanded_ids, K),
                B=n_seq,
                H=1,
                Q_LEN=s_tok,
                KV_LEN=s_tok,
                device=device,
            )

            for layer in self.transformer: x = layer(x, b_mask)
            turn_x = x.view(n_seq * batch_seq_len, TOKENS_PER_TURN, self.d_model)
            read_q = turn_x[:, ACTOR_IDX:CRITIC_IDX + 1, :]
            readout = self.readout(read_q, turn_x)
            
            pi = self.pi_head(readout[:, 0, :]).float()
            v  = self.v_head(readout[:, 1, :]).float()
            v_exp = (torch.softmax(v, dim=-1) * self.v_support).sum(dim=-1)
            return pi, v, v_exp.float(), None, None

        # --- BATCHED INFERENCE MODE ---
        if kv_cache_len is None:
            kv_cache_len = torch.zeros(B, dtype=torch.long, device=device)
        else:
            kv_cache_len = kv_cache_len.to(device=device, dtype=torch.long)
        S = TOKENS_PER_TURN + K
        b_mask = create_block_mask(
        lambda b, h, q, k: pokemon_inference_mask(
                    b, h, q, k, kv_cache_len, K
                ),
                B=B,
                H=1,
                Q_LEN=S,
                KV_LEN=S,
                device=device,
            )
        
        hist_pos = torch.arange(K, device=device).unsqueeze(0)                 # [1, K]
        hist_valid = hist_pos >= (K - kv_cache_len.unsqueeze(1)) 
        
        x = current_turn
        new_kv_cache = []
        new_kv_cache_len = torch.clamp(kv_cache_len + 1, max=K)

        for i, layer in enumerate(self.transformer):
            # Cache the INPUT field token for this layer to match packed training semantics
            cur_field_in = x[:, FIELD_IDX:FIELD_IDX + 1, :]   # [B, 1, D]
            
            if kv_cache is not None:
                hist_fields_L = kv_cache[i].to(device=device, dtype=x.dtype, non_blocking=True)
            else:
                hist_fields_L = torch.zeros(B, K, self.d_model, device=device, dtype=x.dtype)
                
            hist_fields_L = hist_fields_L * hist_valid.unsqueeze(-1).to(x.dtype)
            # 3. Process
            x_full  = torch.cat([x, hist_fields_L], dim=1)
            x_full  = layer(x_full , b_mask)
            x = x_full[:, :TOKENS_PER_TURN, :]
            # Update layer cache: right-aligned, keep last K
            updated_hist = torch.cat([hist_fields_L, cur_field_in], dim=1)[:, -K:, :]
            new_kv_cache.append(updated_hist)
        
        read_q = x[:, ACTOR_IDX:CRITIC_IDX + 1, :]      # [B, 2, D]
        readout = self.readout(read_q, x)
        
        pi = self.pi_head(readout[:, 0, :])  # actor
        v  = self.v_head(readout[:, 1, :])   # critic
        v_exp = (torch.softmax(v, dim=-1) * self.v_support).sum(dim=-1)
        
        return pi.float(), v.float(), v_exp.float(), new_kv_cache, new_kv_cache_len
    
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
        self.dim_transition_sc = meta["dim_transition_scalars"]
        self.transition_map = meta["feature_map"]["transition"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {name: x[:, start:end] for name, (start, end) in self.offsets.items()}
        
        # Reshape structured fields
        out["pokemon_body"] = out["pokemon_body"].reshape(-1, 12, self.dim_body)
        out["pokemon_ids"]  = out["pokemon_ids"].reshape(-1, 12, 2).to(torch.long)
        out["ability_ids"]  = out["ability_ids"].reshape(-1, 12, 4).to(torch.long)
        out["move_ids"]     = out["move_ids"].reshape(-1, 12, 4).to(torch.long)
        out["move_scalars"] = out["move_scalars"].reshape(-1, 12, 4, self.dim_move_sc)
        out["transition_ids"] = out["transition_ids"].reshape(-1, self.meta["dim_transition_ids"]).to(torch.long)
        out["transition_scalars"] = out["transition_scalars"].reshape(-1, self.dim_transition_sc)
        
        tmap = self.transition_map
        out["transition_move_ids"] = out["transition_ids"][:, tmap["move_ids"][0]:tmap["move_ids"][1]]
        out["transition_pokemon_ids"] = out["transition_ids"][:, tmap["pokemon_ids"][0]:tmap["pokemon_ids"][1]]
        out["transition_ability_ids"] = out["transition_ids"][:, tmap["ability_ids"][0]:tmap["ability_ids"][1]]
        out["transition_item_ids"] = out["transition_ids"][:, tmap["item_ids"][0]:tmap["item_ids"][1]]
        
        return out

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
    total_loss: float
    n_mb: int

class AsyncEpisodeDataset:
    def __init__(self, act_dim: int, device: str):
        self.act_dim, self.device = act_dim, device
        self.clear()

    def __len__(self): return int(self.n_steps)

    def clear(self):
        # Flattened storage for performance
        self.obs, self.act, self.logp, self.val, self.adv, self.ret, self.next_hp = [], [], [], [], [], [], []
        self.ep_ids = []  # New: tracks boundaries via flat IDs
        self.n_steps = 0
        self.ep_counter = 0 
        self._tensor_cache = None

    def add_steps(self, obs, act, logp, val, adv, ret, next_hp, ep_ids):
        """
        Maintains the flat plumbing. 'ep_ids' should match the length of the steps.
        """
        def dev(x): return x.detach().to(self.device)
        self.obs.append(obs.detach().to("cpu", non_blocking=True))
        self.act.append(dev(act)); self.logp.append(dev(logp))
        self.val.append(dev(val)); self.adv.append(dev(adv))
        self.ret.append(dev(ret)); self.next_hp.append(dev(next_hp))
        self.ep_ids.append(dev(ep_ids))
        self.n_steps += int(act.shape[0])

    def tensorize(self) -> Tuple:
        if self._tensor_cache: return self._tensor_cache
        self._tensor_cache = (
            torch.cat(self.obs, dim=0), torch.cat(self.act, dim=0), 
            torch.cat(self.logp, dim=0).float(), torch.cat(self.val, dim=0).float(), 
            torch.cat(self.adv, dim=0).float(), torch.cat(self.ret, dim=0).float(), 
            torch.cat(self.next_hp, dim=0).float(),
            torch.cat(self.ep_ids, dim=0) # Index 7: Episode Boundaries
        )
        return self._tensor_cache
    
    def swap_out_tensor_cache(self):
        data = self.tensorize(); self.clear(); return data
        
    def _episode_bounds(self, ep_ids: torch.Tensor) -> list[tuple[int, int]]:
        n = int(ep_ids.numel())
        if n == 0:
            return []

        changes = torch.nonzero(ep_ids[1:] != ep_ids[:-1], as_tuple=False).flatten() + 1
        bounds = torch.cat([
            torch.tensor([0], dtype=torch.long, device=ep_ids.device),
            changes,
            torch.tensor([n], dtype=torch.long, device=ep_ids.device),
        ])
        return [(int(bounds[i]), int(bounds[i + 1])) for i in range(bounds.numel() - 1)]

    def iter_minibatches(self, mb_size: int, shuffle_episodes: bool = False) -> Iterator[Tuple]:
        # Sequence-Aware Slicing
        data = self.tensorize()
        ep_ids = data[7]
        n = int(ep_ids.shape[0])
        
        if shuffle_episodes:
            spans = self._episode_bounds(ep_ids)
            perm = torch.randperm(len(spans)).tolist()
            order = torch.cat([
                torch.arange(spans[i][0], spans[i][1], dtype=torch.long)
                for i in perm
            ], dim=0)
        else:
            order = torch.arange(n, dtype=torch.long)
            
        usable = (int(order.numel()) // mb_size) * mb_size
        if usable == 0:
            return
        
        order = order[:usable]
        for start in range(0, usable, mb_size):
            idx = order[start:start + mb_size]
            yield tuple(d[idx] for d in data)

@torch.no_grad()
def gae_from_episode(
    rewards: torch.Tensor,     # [T]
    values: torch.Tensor,      # [T]
    dones: torch.Tensor,       # [T] (1 at terminal step)
    gamma: float,
    lam: float,
    last_value: float = 0.0,   # terminal -> 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes GAE for one episode.
    """
    T = rewards.shape[0]
    adv = torch.zeros((T,), device=rewards.device)
    gae = torch.tensor(0.0, device=rewards.device)
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_value = torch.tensor(last_value, device=rewards.device) if (t == T - 1) else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        adv[t] = gae
    ret = adv + values
    return adv, ret

def ppo_update(
    net: nn.Module,
    opt: optim.Optimizer,
    dataset: AsyncEpisodeDataset,
    scheduler=None,
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
    stop_training = False
    
    grad_accum_steps = cfg.get("grad_accum_steps", 1)
    
    for epoch in range(cfg.get("update_epochs", 1)):
        epoch_kl, epoch_mb = 0.0, 0
        
        opt.zero_grad(set_to_none=True)
        
        for i, (mb_obs, mb_act, mb_logp_old, _, mb_adv, mb_ret, _,mb_ep_ids) in enumerate(dataset.iter_minibatches(cfg["minibatch_size"], shuffle_episodes=True)):
            mb_act, mb_logp_old, mb_adv, mb_ret, mb_ep_ids = (t.to(dev) for t in [mb_act, mb_logp_old, mb_adv, mb_ret, mb_ep_ids])

            mb_obs = mb_obs.to(dev, non_blocking=True)
            m_start, m_end = net.unpacker.offsets["action_mask"]
            mb_mask = mb_obs[:, m_start:m_end]
            
            logits, v_logits, _, _, _ = net(mb_obs, 
                                         mb_ep_ids, 
                                         batch_seq_len=cfg["batch_seq_len"],
                                         )

            if mode == "imitation":
                pg_loss = nn.functional.cross_entropy(logits.float(), mb_act, label_smoothing=0.1)
                ent_loss = approx_kl = clip_frac = torch.tensor(0.0, device=dev)
                
                target_dist = twohot_targets(mb_ret, v_min=cfg["v_min"], v_max=cfg["v_max"], v_bins=cfg["v_bins"])
                v_loss = dist_value_loss(v_logits, target_dist)
                loss = pg_loss + v_loss
            elif mode == "warmup":
                target_dist = twohot_targets(mb_ret, v_min=cfg["v_min"], v_max=cfg["v_max"], v_bins=cfg["v_bins"])
                loss = dist_value_loss(v_logits, target_dist)
                v_loss = loss; pg_loss = ent_loss = approx_kl = clip_frac = torch.tensor(0.0, device=dev)
            else:
                logp_game, ent_game = masked_logprob_entropy(logits.float(), mb_mask, mb_act)

                # --- DECOUPLED POLICY GRADIENTS ---
                # 1. Game Policy Gradient (Safe from memory noise)
                ratio_game = (logp_game - mb_logp_old).exp()
                pg_loss_game = torch.max(-mb_adv * ratio_game, -mb_adv * ratio_game.clamp(1-cfg["clip_coef"], 1+cfg["clip_coef"])).mean()

                # Total PG Loss is the sum of independent branches
                pg_loss = pg_loss_game
                
                target_dist = twohot_targets(mb_ret, v_min=cfg["v_min"], v_max=cfg["v_max"], v_bins=cfg["v_bins"])
                v_loss = dist_value_loss(v_logits, target_dist)

                ent_loss = ent_game.mean()
                
                loss = pg_loss + cfg["vf_coef"] * v_loss - (cfg["ent_coef"] * ent_loss)
                
                # Track KL and Clip Frac for the Game Action specifically so you can monitor true agent health
                approx_kl = (mb_logp_old - logp_game).mean() 
                clip_frac = ((ratio_game - 1.0).abs() > cfg["clip_coef"]).float().mean()
                
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            # Stats update
            n_mb += 1; epoch_mb += 1
            stats["kl"] += approx_kl.item(); epoch_kl += approx_kl.item()
            stats["clip"] += clip_frac.item(); stats["ent"] += ent_loss.item()
            stats["v"] += v_loss.item(); stats["pg"] += pg_loss.item(); stats["total"] += loss.item()

            if epoch > 0 and mode == "ppo" and cfg.get("target_kl") and (epoch_kl / epoch_mb) > cfg["target_kl"] * 1.5:
                logger.info(f"Early stop at Epoch {epoch} due to KL {epoch_kl/epoch_mb:.4f}")
                stop_training = True
                break
        
        if not stop_training and (i + 1) % grad_accum_steps != 0:
            nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        
        if stop_training:
            break

    return PPOUpdateStats(
        approx_kl=stats["kl"]/max(n_mb, 1), 
        clip_frac=stats["clip"]/max(n_mb, 1), 
        entropy=stats["ent"]/max(n_mb, 1),
        v_loss=stats["v"]/max(n_mb, 1), 
        pg_loss=stats["pg"]/max(n_mb, 1), 
        total_loss=stats["total"]/max(n_mb, 1), 
        n_mb=n_mb
    )