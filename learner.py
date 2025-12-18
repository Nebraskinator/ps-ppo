# learner.py
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Optional
import os
import glob
import random
import numpy as np
import torch
import torch.optim as optim

from ppo_core import ActorCriticTransformer, gae_from_episode, AsyncEpisodeDataset, ppo_update


@dataclass
class LearnerConfig:
    # model
    n_tokens: int = 74
    model_dim: int = 64
    n_layers: int = 4
    n_heads: int = 1
    act_dim: int = 10

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 4096
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip_vloss: bool = True
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02

    # update trigger
    steps_per_update: int = 32768

    # devices
    device: str = "cuda"
    
    # ---- checkpointing ----
    ckpt_dir: str = "checkpoints"
    save_every_updates: int = 25          # save every N PPO updates
    keep_last: int = 500                   # rotate: keep last K checkpoints
    resume: bool = True                  # auto-load latest ckpt on startup if present


class LearnerActor:
    """
    Receives completed episodes from workers, trains PPO, and sends weights to InferenceActor.
    """
    def __init__(self, cfg: LearnerConfig, inference_actor):
        self.cfg = cfg
        self.inference_actor = inference_actor

        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("LearnerActor device=cuda but CUDA not available")

        self._token_in_dim: Optional[int] = None
        self.net: Optional[ActorCriticTransformer] = None
        self.opt: Optional[optim.Optimizer] = None

        self.dataset = AsyncEpisodeDataset(act_dim=cfg.act_dim, device="cpu")
        self.update_idx = 0
        self.total_episodes = 0
        self.total_steps = 0

        self._q: asyncio.Queue = asyncio.Queue()
        
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        self._last_save_t = 0.0
        
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop())
        self._startup_task = loop.create_task(self._maybe_resume_latest())

    def _init_if_needed(self, token_in_dim: int):
        if self.net is not None:
            return
        self._token_in_dim = int(token_in_dim)
        self.net = ActorCriticTransformer(
            n_tokens=self.cfg.n_tokens,
            token_in_dim=self._token_in_dim,
            act_dim=self.cfg.act_dim,
            model_dim=self.cfg.model_dim,
            n_layers=self.cfg.n_layers,
            n_heads=self.cfg.n_heads,
        ).to(self.cfg.device).train()
        self.opt = optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        
    def _ckpt_path_for_update(self, update_idx: int) -> str:
        # Windows-safe filename
        return os.path.join(self.cfg.ckpt_dir, f"learner_update_{update_idx:06d}.pt")

    def _latest_ckpt_path(self) -> Optional[str]:
        paths = sorted(glob.glob(os.path.join(self.cfg.ckpt_dir, "learner_update_*.pt")))
        return paths[-1] if paths else None

    def _rotate_checkpoints(self):
        k = int(self.cfg.keep_last)
        if k <= 0:
            return
        paths = sorted(glob.glob(os.path.join(self.cfg.ckpt_dir, "learner_update_*.pt")))
        if len(paths) <= k:
            return
        for p in paths[:-k]:
            try:
                os.remove(p)
            except Exception:
                pass

    async def _maybe_resume_latest(self):
        if not self.cfg.resume:
            return
        path = self._latest_ckpt_path()
        if not path:
            return
        try:
            self._load_checkpoint(path)
            # push weights to inference so rollouts use restored policy
            assert self.net is not None
            sd_cpu = {k: v.detach().to("cpu") for k, v in self.net.state_dict().items()}
            await self.inference_actor.set_weights.remote(sd_cpu)
            print(f"[learner] resumed from {path}")
        except Exception as e:
            print(f"[learner] resume failed from {path}: {e!r}")

    def _save_checkpoint(self, path: str):
        assert self.net is not None and self.opt is not None
        assert self._token_in_dim is not None

        payload = {
            "update_idx": int(self.update_idx),
            "total_episodes": int(self.total_episodes),
            "total_steps": int(self.total_steps),
            "token_in_dim": int(self._token_in_dim),
            "model": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cfg": self.cfg,  # nice to have; dataclass is picklable
        }

        # atomic write (Windows-friendly): write temp -> replace
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
        token_in_dim = ckpt.get("token_in_dim", None)
        if token_in_dim is None:
            raise RuntimeError("Checkpoint missing token_in_dim; cannot init model.")
        self._init_if_needed(int(token_in_dim))
    
        assert self.net is not None and self.opt is not None
        self.net.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
    
        self.update_idx = int(ckpt.get("update_idx", 0))
        self.total_episodes = int(ckpt.get("total_episodes", 0))
        self.total_steps = int(ckpt.get("total_steps", 0))
    
        # RNG restore (must be CPU ByteTensor)
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


    def _should_save_now(self) -> bool:
        if self.net is None or self.opt is None:
            return False

        # save every N updates (including update 1)
        by_update = (self.cfg.save_every_updates > 0) and (self.update_idx % self.cfg.save_every_updates == 0)

        return by_update

    async def save_now(self, path: Optional[str] = None) -> str:
        """
        Manual save hook callable from driver if you want it later (ray call).
        """
        if self.net is None or self.opt is None or self._token_in_dim is None:
            raise RuntimeError("Cannot save: model not initialized yet (no episodes seen).")
        if path is None:
            path = self._ckpt_path_for_update(self.update_idx)
        self._save_checkpoint(path)
        self._rotate_checkpoints()
        return path

    # ----------------------------
    # Public API
    # ----------------------------
    
    async def submit_episode(
        self,
        tokens: np.ndarray,  # [T,N,D]
        tmask: np.ndarray,   # [T,N]
        amask: np.ndarray,   # [T,A]
        act: np.ndarray,     # [T]
        logp: np.ndarray,    # [T]
        val: np.ndarray,     # [T]
        rew: np.ndarray,     # [T]
        done: np.ndarray,    # [T] (1 at terminal)
    ):
        """
        Worker sends finalized episode (already has logp/value from inference).
        """
        self._q.put_nowait((tokens, tmask, amask, act, logp, val, rew, done))

    async def get_stats(self) -> dict:
        return {
            "update": self.update_idx,
            "episodes": self.total_episodes,
            "steps_in_dataset": len(self.dataset),
            "total_steps": self.total_steps,
        }
    
    async def submit_packed_batch(
        self,
        tokens_cat: np.ndarray,  # [S,N,D]
        tmask_cat: np.ndarray,   # [S,N]
        amask_cat: np.ndarray,   # [S,A]
        act_cat: np.ndarray,     # [S]
        logp_cat: np.ndarray,    # [S]
        val_cat: np.ndarray,     # [S]
        rew_cat: np.ndarray,     # [S]
        done_cat: np.ndarray,    # [S]
        lengths: np.ndarray,     # [B] int32 episode lengths summing to S
    ):
        """
        Receive a packed batch of episodes. This avoids creating many Ray objects.
        """
        self._q.put_nowait(("packed", tokens_cat, tmask_cat, amask_cat, act_cat, logp_cat, val_cat, rew_cat, done_cat, lengths))


    # ----------------------------
    # Main training loop
    # ----------------------------
    
    def gae_from_packed(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                    gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        last_adv = 0.0
        last_value = 0.0
        for t in range(T - 1, -1, -1):
            if dones[t] > 0.0:
                next_nonterminal = 0.0
                next_value = 0.0
                last_adv = 0.0
            else:
                next_nonterminal = 1.0
                next_value = last_value
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_adv = delta + gamma * lam * next_nonterminal * last_adv
            adv[t] = last_adv
            last_value = values[t]
        ret = adv + values
        return adv, ret

    
    async def _loop(self):
        while True:
            item = await self._q.get()
            
            if isinstance(item, tuple) and len(item) > 0 and item[0] == "packed":
                _, tokens_cat, tmask_cat, amask_cat, act_cat, logp_cat, val_cat, rew_cat, done_cat, lengths = item

                # init
                self._init_if_needed(tokens_cat.shape[-1])

                # torchify once
                tokens_all = torch.from_numpy(tokens_cat).float()
                tmask_all  = torch.from_numpy(tmask_cat).float()
                amask_all  = torch.from_numpy(amask_cat).float()
                act_all    = torch.from_numpy(act_cat).long()
                logp_all   = torch.from_numpy(logp_cat).float()
                val_all    = torch.from_numpy(val_cat).float()
                rew_all    = torch.from_numpy(rew_cat).float()
                done_all   = torch.from_numpy(done_cat).float()

                adv_all, ret_all = self.gae_from_packed(rew_all, val_all, done_all, self.cfg.gamma, self.cfg.gae_lambda)
                self.dataset.add_steps(tokens_all, tmask_all, amask_all, act_all, logp_all, val_all, adv_all, ret_all)
                self.total_episodes += int(lengths.shape[0])
                self.total_steps += int(act_all.shape[0])
                    
            else:
            
                tokens, tmask, amask, act, logp, val, rew, done = item
    
                self._init_if_needed(tokens.shape[-1])
    
                # torchify on CPU
                tokens_t = torch.from_numpy(tokens).float()
                tmask_t  = torch.from_numpy(tmask).float()
                amask_t  = torch.from_numpy(amask).float()
                act_t    = torch.from_numpy(act).long()
                logp_t   = torch.from_numpy(logp).float()
                val_t    = torch.from_numpy(val).float()
                rew_t    = torch.from_numpy(rew).float()
                done_t   = torch.from_numpy(done).float()
    
                adv_t, ret_t = gae_from_episode(
                    rewards=rew_t,
                    values=val_t,
                    dones=done_t,
                    gamma=self.cfg.gamma,
                    lam=self.cfg.gae_lambda,
                    last_value=0.0,
                )
    
                self.dataset.add_steps(tokens_t, tmask_t, amask_t, act_t, logp_t, val_t, adv_t, ret_t)
    
                self.total_episodes += 1
                self.total_steps += int(act.shape[0])
    
            if len(self.dataset) < self.cfg.steps_per_update:
                continue

            # ---- PPO update ----
            assert self.net is not None and self.opt is not None

            tokens_u, tmask_u, amask_u, act_u, logp_u, val_u, adv_u, ret_u = self.dataset.tensorize()

            # train on GPU
            tokens_u = tokens_u.to(self.cfg.device)
            tmask_u  = tmask_u.to(self.cfg.device)
            amask_u  = amask_u.to(self.cfg.device)
            act_u    = act_u.to(self.cfg.device)
            logp_u   = logp_u.to(self.cfg.device)
            val_u    = val_u.to(self.cfg.device)
            adv_u    = adv_u.to(self.cfg.device)
            ret_u    = ret_u.to(self.cfg.device)

            adv_u = (adv_u - adv_u.mean()) / (adv_u.std().clamp_min(1e-8))

            train_ds = AsyncEpisodeDataset(act_dim=self.cfg.act_dim, device=self.cfg.device)
            train_ds.add_steps(tokens_u, tmask_u, amask_u, act_u, logp_u, val_u, adv_u, ret_u)

            stats = ppo_update(
                net=self.net,
                opt=self.opt,
                dataset=train_ds,
                update_epochs=self.cfg.update_epochs,
                minibatch_size=self.cfg.minibatch_size,
                clip_coef=self.cfg.clip_coef,
                ent_coef=self.cfg.ent_coef,
                vf_coef=self.cfg.vf_coef,
                clip_vloss=self.cfg.clip_vloss,
                max_grad_norm=self.cfg.max_grad_norm,
                target_kl=self.cfg.target_kl,
            )

            self.update_idx += 1

            # push weights to inference actor (CPU tensors for Ray)
            sd_cpu = {k: v.detach().to("cpu") for k, v in self.net.state_dict().items()}
            await self.inference_actor.set_weights.remote(sd_cpu)

            # clear dataset
            self.dataset.clear()

            print(
                f"[learner] upd={self.update_idx} "
                f"kl={stats.approx_kl:.4f} clip={stats.clip_frac:.3f} ent={stats.entropy:.3f} "
                f"vloss={stats.v_loss:.3f} ploss={stats.pg_loss:.3f} n_mb={stats.n_mb}"
            )
            
            # ---- checkpoint ----
            if self._should_save_now():
                try:
                    path = self._ckpt_path_for_update(self.update_idx)
                    self._save_checkpoint(path)
                    self._rotate_checkpoints()
                    print(f"[learner] saved checkpoint: {path}")
                except Exception as e:
                    print(f"[learner] checkpoint save failed: {e!r}")
