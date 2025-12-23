# ppo_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ----------------------------
# Masked categorical utils
# ----------------------------
def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(dtype=torch.bool)
    neg = -1e9 if logits.dtype in (torch.float32, torch.float64) else -1e4
    return logits.masked_fill(~m, neg)


@torch.no_grad()
def masked_sample(logits: torch.Tensor, 
                  mask: torch.Tensor,
                  greedy: bool = False,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ml = masked_logits(logits, mask)
    if greedy:
        # masked argmax
        a = torch.argmax(ml, dim=-1)

        # log-prob under the policy (useful for logging)
        logp = torch.log_softmax(ml, dim=-1).gather(
            1, a.unsqueeze(-1)
        ).squeeze(-1)

        # entropy is not meaningful for deterministic choice
        ent = torch.zeros_like(logp)

        return a, logp, ent
    dist = Categorical(logits=ml)
    a = dist.sample()
    logp = dist.log_prob(a)
    ent = dist.entropy()
    return a, logp, ent


def masked_logprob_entropy(
    logits: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    ml = masked_logits(logits, mask)
    dist = Categorical(logits=ml)
    logp = dist.log_prob(actions)
    ent = dist.entropy()
    return logp, ent


# ----------------------------
# Model
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], act_dim: int, hidden: int = 512):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))

        self.trunk = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

        # Orthogonal init (robust PPO default)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = obs.flatten(1)
        h = self.trunk(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value

class ActorCriticTransformer(nn.Module):
    def __init__(
        self,
        act_dim: int,
        *,
        model_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        # ---- obs schema ----
        t_max: int = 128,
        float_dim: int = 16,
        n_tok_types: int = 8,
        n_owner: int = 3,
        n_pos: int = 7,       # 0..5 plus POS_NA=6
        n_subpos: int = 5,    # 0..3 plus SUBPOS_NA=4
        n_entity: int = 1,    # set to showdown_obs.N_ENTITY
    ):
        super().__init__()
        self.act_dim = act_dim
        self.t_max = t_max
        self.float_dim = float_dim

        # floats -> model_dim
        self.float_proj = nn.Sequential(
            nn.LayerNorm(float_dim),
            nn.Linear(float_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        # categorical embeddings (all summed)
        self.type_emb   = nn.Embedding(n_tok_types, model_dim)
        self.owner_emb  = nn.Embedding(n_owner, model_dim)
        self.pos_emb    = nn.Embedding(n_pos, model_dim)
        self.subpos_emb = nn.Embedding(n_subpos, model_dim)
        self.entity_emb = nn.Embedding(n_entity, model_dim)
        
        self.cls_emb = nn.Parameter(torch.zeros(model_dim))  # learnable CLS bias
        nn.init.normal_(self.cls_emb, mean=0.0, std=0.02)    # optional but recommended

        self.in_ln = nn.LayerNorm(model_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=model_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.pi = nn.Linear(model_dim, act_dim)
        self.v  = nn.Linear(model_dim, 1)

        # init (same style you used)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)

        # make embedding(0) be ~neutral for "NONE"/padding IDs
        nn.init.zeros_(self.entity_emb.weight.data[0])
        nn.init.zeros_(self.owner_emb.weight.data[2]) # OWNER_NONE=2 (optional)
        nn.init.zeros_(self.pos_emb.weight.data[-1])  # POS_NA
        nn.init.zeros_(self.subpos_emb.weight.data[-1])  # SUBPOS_NA

    def forward(
        self,
        float_feats: torch.Tensor,   # [B, T, F]
        tok_type: torch.Tensor,      # [B, T]
        owner: torch.Tensor,         # [B, T]
        pos: torch.Tensor,           # [B, T]
        subpos: torch.Tensor,        # [B, T]
        entity_id: torch.Tensor,     # [B, T]
        token_mask: Optional[torch.Tensor] = None,  # [B, T] 1=keep,0=pad
    ):
        # floats
        x = self.float_proj(float_feats)

        # categorical sums
        x = x \
            + self.type_emb(tok_type) \
            + self.owner_emb(owner) \
            + self.pos_emb(pos) \
            + self.subpos_emb(subpos) \
            + self.entity_emb(entity_id)
            
        # Set learned CLS embedding to token 0
        x[:, 0, :] = x[:, 0, :] + self.cls_emb

        x = self.in_ln(x)

        key_padding_mask = None
        if token_mask is not None:
            key_padding_mask = (token_mask <= 0.5)  # True = PAD

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        cls = h[:, 0, :]  # CLS index 0 by construction
        logits = self.pi(cls)
        value  = self.v(cls).squeeze(-1)
        return logits, value


# ----------------------------
# PPO config
# ----------------------------
@dataclass
class PPOConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    num_envs: int = 64
    rollout_steps: int = 256  # sparse rewards -> longer rollouts help

    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-4
    lr_anneal: bool = True
    max_grad_norm: float = 0.5

    update_epochs: int = 4
    minibatch_size: int = 4096  # tune based on CPU/GPU
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip_vloss: bool = True
    target_kl: Optional[float] = 0.02

    # --- league self-play ---
    league_capacity: int = 16
    league_add_every: int = 25          # add a snapshot every N updates
    league_p_latest: float = 0.50       # prob opponent is the latest learner
    league_p_random: float = 0.00       # optional: pure random opponent fallback
    
    # --- battle logging ---
    log_dir: str = "battle_logs"
    log_every: int = 1          # log one battle every N updates
    watched_env_index: int = 0   # which env to log
    
    ckpt_dir: str = "checkpoints"
    save_every: int = 25        # save every N updates
    keep_last: int = 5          # rolling window
    save_latest: bool = True 
    
    resume_path: str | None = None


# ----------------------------
# Rollout buffer for learner only (player_0)
# ----------------------------
class LearnerRollout:
    def __init__(self, T: int, E: int, obs_shape: Tuple[int, ...], act_dim: int, device: str):
        self.T, self.E = T, E
        self.device = device

        self.obs = torch.zeros((T, E, *obs_shape), device=device)
        self.mask = torch.zeros((T, E, act_dim), device=device)

        self.act = torch.zeros((T, E), dtype=torch.long, device=device)
        self.logp = torch.zeros((T, E), device=device)
        self.val = torch.zeros((T, E), device=device)

        self.rew = torch.zeros((T, E), device=device)
        self.done = torch.zeros((T, E), device=device)   # learner done
        self.live = torch.zeros((T, E), device=device)   # learner acting (should be 1 until dead)

        self.adv = torch.zeros((T, E), device=device)
        self.ret = torch.zeros((T, E), device=device)

        self.t = 0

    def add(self, obs, mask, act, logp, val, rew, done, live):
        t = self.t
        self.obs[t].copy_(obs)
        self.mask[t].copy_(mask)
        self.act[t].copy_(act)
        self.logp[t].copy_(logp)
        self.val[t].copy_(val)
        self.rew[t].copy_(rew)
        self.done[t].copy_(done)
        self.live[t].copy_(live)
        self.t += 1

    @torch.no_grad()
    def compute_gae(self, last_val: torch.Tensor, gamma: float, lam: float):
        """
        last_val: [E]
        """
        gae = torch.zeros((self.E,), device=self.device)
        for t in reversed(range(self.T)):
            next_nonterminal = 1.0 - self.done[t]
            next_value = last_val if t == self.T - 1 else self.val[t + 1]
            delta = self.rew[t] + gamma * next_value * next_nonterminal - self.val[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            self.adv[t] = gae
        self.ret = self.adv + self.val

    def iter_minibatches(self, mb_size: int):
        obs = self.obs.reshape((-1, *self.obs.shape[2:]))
        mask = self.mask.reshape((-1, self.mask.shape[-1]))
        act = self.act.reshape((-1,))
        logp = self.logp.reshape((-1,))
        val = self.val.reshape((-1,))
        adv = self.adv.reshape((-1,))
        ret = self.ret.reshape((-1,))
        live = self.live.reshape((-1,))

        idx = torch.nonzero(live > 0.5, as_tuple=False).squeeze(-1)
        idx = idx[torch.randperm(idx.numel(), device=idx.device)]
        for start in range(0, idx.numel(), mb_size):
            mb = idx[start:start + mb_size]
            yield obs[mb], mask[mb], act[mb], logp[mb], val[mb], adv[mb], ret[mb]


# ----------------------------
# League manager (frozen opponent snapshots)
# ----------------------------
class League:
    def __init__(self, obs_shape: Tuple[int, ...], act_dim: int, device: str, capacity: int):
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.device = device
        self.capacity = capacity
        self.snapshots: List[ActorCritic] = []

    @torch.no_grad()
    def add_snapshot(self, net: ActorCritic):
        snap = ActorCritic(self.obs_shape, self.act_dim).to(self.device)
        snap.load_state_dict({k: v.detach().clone() for k, v in net.state_dict().items()})
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)

        self.snapshots.append(snap)
        if len(self.snapshots) > self.capacity:
            # drop oldest
            self.snapshots.pop(0)

    def sample_opponent(self, latest_net: ActorCritic, p_latest: float, p_random: float = 0.0):
        r = random.random()
        if r < p_random:
            return None  # special-cased random opponent
        if r < p_random + p_latest or len(self.snapshots) == 0:
            return latest_net
        return random.choice(self.snapshots)
    
    @torch.no_grad()
    def add_snapshot_path(self, path: str):
        """
        Load a saved ActorCritic state_dict from disk and add as a frozen snapshot.
        Expects `torch.save(learner.state_dict(), path)` format.
        """
        state = torch.load(path, map_location=self.device)
    
        snap = ActorCritic(self.obs_shape, self.act_dim).to(self.device)
        snap.load_state_dict(state)
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
    
        self.snapshots.append(snap)
        if len(self.snapshots) > self.capacity:
            self.snapshots.pop(0)

# ============================
# Async episodic dataset for Showdown-style training
# ============================
from dataclasses import dataclass

@dataclass
class PPOUpdateStats:
    approx_kl: float
    clip_frac: float
    entropy: float
    v_loss: float
    pg_loss: float
    n_mb: int

class AsyncEpisodeDataset:
    def __init__(self, act_dim: int, device: str):
        self.act_dim = int(act_dim)
        self.device = device
        self.clear()
    
    def __len__(self):
        return int(self.n_steps)
    
    def _maybe_pin(self, x: torch.Tensor) -> torch.Tensor:
        return x.pin_memory() if x.device.type == "cpu" else x
    
    def swap_out_tensor_cache(self):
        data = self.tensorize()
        self.clear()
        return data
    
    def clear(self):
        self._tensor_cache = None
        self.float_feats = []
        self.tok_type = []
        self.owner = []
        self.pos = []
        self.subpos = []
        self.entity_id = []
        self.tmask  = []
        self.amask  = []
        self.act    = []
        self.logp   = []
        self.val    = []
        self.adv    = []
        self.ret    = []
        self.n_steps = 0

    def add_steps(self, float_feats, tok_type, owner, pos, subpos, entity_id, tmask, amask, act, logp, val, adv, ret):
        # float_feats: [T, N, F]
        # tok_type/owner/pos/subpos/entity_id: [T, N]
        # tmask: [T, N], amask: [T, A]
        # act/logp/val/adv/ret: [T]
        def dev(x):
            return x.detach().cpu() if self.device == "cpu" else x.detach().to(self.device)

        self.float_feats.append(dev(float_feats))
        self.tok_type.append(dev(tok_type))
        self.owner.append(dev(owner))
        self.pos.append(dev(pos))
        self.subpos.append(dev(subpos))
        self.entity_id.append(dev(entity_id))
        self.tmask.append(dev(tmask))
        self.amask.append(dev(amask))
        self.act.append(dev(act))
        self.logp.append(dev(logp))
        self.val.append(dev(val))
        self.adv.append(dev(adv))
        self.ret.append(dev(ret))

        self.n_steps += int(act.shape[0])
        self._tensor_cache = None

    def tensorize(self):
        if self._tensor_cache is not None:
            return self._tensor_cache

        float_feats = self._maybe_pin(torch.cat(self.float_feats, dim=0))
        tok_type    = self._maybe_pin(torch.cat(self.tok_type,    dim=0))
        owner       = self._maybe_pin(torch.cat(self.owner,       dim=0))
        pos         = self._maybe_pin(torch.cat(self.pos,         dim=0))
        subpos      = self._maybe_pin(torch.cat(self.subpos,      dim=0))
        entity_id   = self._maybe_pin(torch.cat(self.entity_id,   dim=0))
        tmask       = self._maybe_pin(torch.cat(self.tmask,       dim=0))
    
        amask  = self._maybe_pin(torch.cat(self.amask,  dim=0))
        act    = self._maybe_pin(torch.cat(self.act,    dim=0))
        logp   = self._maybe_pin(torch.cat(self.logp,   dim=0))
        val    = self._maybe_pin(torch.cat(self.val,    dim=0))
        adv    = self._maybe_pin(torch.cat(self.adv,    dim=0))
        ret    = self._maybe_pin(torch.cat(self.ret,    dim=0))

        self._tensor_cache = (float_feats, tok_type, owner, pos, subpos, entity_id, tmask, amask, act, logp, val, adv, ret)
        return self._tensor_cache

    def iter_minibatches(self, mb_size: int, shuffle: bool = True):
        float_feats, tok_type, owner, pos, subpos, entity_id, tmask, amask, act, logp, val, adv, ret = self.tensorize()
        n = act.shape[0]
        idx = torch.arange(n, device=act.device)
        if shuffle:
            idx = idx[torch.randperm(n, device=act.device)]
        for start in range(0, n, mb_size):
            mb = idx[start:start + mb_size]
            yield (
                float_feats[mb],
                tok_type[mb],
                owner[mb],
                pos[mb],
                subpos[mb],
                entity_id[mb],
                tmask[mb],
                amask[mb],
                act[mb],
                logp[mb],
                val[mb],
                adv[mb],
                ret[mb],
            )



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
    *,
    update_epochs: int,
    minibatch_size: int,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    clip_vloss: bool,
    max_grad_norm: float,
    target_kl: float | None,
    scheduler=None,
) -> PPOUpdateStats:
    """
    PPO update on a flat dataset.
    """
    dev = next(net.parameters()).device
    kl_sum = clip_sum = ent_sum = v_sum = pg_sum = 0.0
    n_mb = 0

    for epoch in range(update_epochs):
        for (mb_ff, mb_tt, mb_own, mb_pos, mb_sub, mb_eid, mb_tmask,
             mb_amask, mb_act, mb_logp_old, mb_val_old, mb_adv, mb_ret) in dataset.iter_minibatches(minibatch_size):
            
            mb_ff    = mb_ff.to(dev, non_blocking=True)
            mb_tt    = mb_tt.to(dev, non_blocking=True)
            mb_own   = mb_own.to(dev, non_blocking=True)
            mb_pos   = mb_pos.to(dev, non_blocking=True)
            mb_sub   = mb_sub.to(dev, non_blocking=True)
            mb_eid   = mb_eid.to(dev, non_blocking=True)
            mb_tmask = mb_tmask.to(dev, non_blocking=True)
            mb_amask = mb_amask.to(dev, non_blocking=True)
            mb_act   = mb_act.to(dev, non_blocking=True)
            mb_logp_old = mb_logp_old.to(dev, non_blocking=True)
            mb_val_old  = mb_val_old.to(dev, non_blocking=True)
            mb_adv      = mb_adv.to(dev, non_blocking=True)
            mb_ret      = mb_ret.to(dev, non_blocking=True)
        
            logits, v = net(mb_ff, mb_tt, mb_own, mb_pos, mb_sub, mb_eid, mb_tmask)
            A = logits.shape[-1]
            if mb_act.min().item() < 0 or mb_act.max().item() >= A:
                print(
                    f"[FATAL] action out of range: min={mb_act.min().item()} max={mb_act.max().item()} A={A}"
                )
                # Optional: dump a few bad indices
                bad = (mb_act < 0) | (mb_act >= A)
                bi = bad.nonzero(as_tuple=False).squeeze(-1)[:10]
                print("bad acts:", mb_act[bi].detach().cpu().tolist())
                raise RuntimeError("mb_act out of bounds")
            logp, ent = masked_logprob_entropy(logits, mb_amask, mb_act)

            ratio = (logp - mb_logp_old).exp()
            pg1 = -mb_adv * ratio
            pg2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            pg_loss = torch.max(pg1, pg2).mean()

            if clip_vloss:
                v_clipped = mb_val_old + torch.clamp(v - mb_val_old, -clip_coef, clip_coef)
                v_loss = 0.5 * torch.max((v - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()
            else:
                v_loss = 0.5 * (v - mb_ret).pow(2).mean()

            ent_loss = ent.mean()
            loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()
            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                approx_kl = (mb_logp_old - logp).mean()
                clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean()

            kl_sum   += float(approx_kl.item())
            clip_sum += float(clip_frac.item())
            ent_sum  += float(ent_loss.item())
            v_sum    += float(v_loss.item())
            pg_sum   += float(pg_loss.item())
            n_mb     += 1

        if target_kl is not None and (kl_sum / max(n_mb, 1)) > target_kl:
            break

    return PPOUpdateStats(
        approx_kl=kl_sum / max(n_mb, 1),
        clip_frac=clip_sum / max(n_mb, 1),
        entropy=ent_sum / max(n_mb, 1),
        v_loss=v_sum / max(n_mb, 1),
        pg_loss=pg_sum / max(n_mb, 1),
        n_mb=n_mb,
    )
