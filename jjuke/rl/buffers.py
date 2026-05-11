"""On-policy rollout buffer skeleton for jjuke's RL framework.

`BaseRolloutBuffer` declares the standard slots (obs, actions, mu, log_sigma,
log_prob, value, reward, done, returns, advantages) as fixed-size GPU tensors,
plus the GAE math + a rough episode-length estimator. Subclasses implement
`append(...)` and `get_flat()` and may add their own slots (e.g. distillation
buffers add `expert_mus` + `expert_mask`).

Reference impl:
    `isaaclab/model/rollout_buffer.py:RolloutBuffer` (PPO)
    `isaaclab/model/rollout_buffer.py:DAggerRolloutBuffer`
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Dict

import torch


class BaseRolloutBuffer(metaclass=ABCMeta):
    """Fixed-size on-policy rollout buffer (skeleton).

    Standard slots (declared in `__init__`):
        obs            (horizon, num_envs, obs_dim)
        actions_raw    (horizon, num_envs, action_dim)
        mus, log_sigmas (horizon, num_envs, action_dim)
        log_probs      (horizon, num_envs)
        values         (horizon, num_envs)
        rewards        (horizon, num_envs)
        dones          (horizon, num_envs)
        returns        (horizon, num_envs)  ← filled by `compute_gae`
        advantages     (horizon, num_envs)  ← filled by `compute_gae`

    Subclasses add domain-specific slots (e.g. `expert_mus` + `expert_mask`
    for DAgger; `value_targets_clipped` for value-clipping; etc.) and
    implement:

      * `append(obs, action_raw, mu, log_sigma, log_prob, value, reward, done)`
      * `get_flat() → dict[str, Tensor]`  (flatten `(horizon, num_envs, ...)` →
        `(horizon * num_envs, ...)` for the PPO update)

    `compute_gae` and `episode_length_estimate` are concrete (the math is
    standard).
    """

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device,
    ) -> None:
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device

        zeros = lambda *shape: torch.zeros(*shape, device=device, dtype=torch.float32)

        self.obs = zeros(horizon, num_envs, obs_dim)
        self.actions_raw = zeros(horizon, num_envs, action_dim)
        self.mus = zeros(horizon, num_envs, action_dim)
        self.log_sigmas = zeros(horizon, num_envs, action_dim)
        self.log_probs = zeros(horizon, num_envs)
        self.values = zeros(horizon, num_envs)
        self.rewards = zeros(horizon, num_envs)
        self.dones = zeros(horizon, num_envs)

        self.returns = zeros(horizon, num_envs)
        self.advantages = zeros(horizon, num_envs)

        self._step = 0

    def reset(self) -> None:
        self._step = 0

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def append(
        self,
        obs: torch.Tensor,
        action_raw: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        # if self._step >= self.horizon:
        #     raise IndexError(f"buffer overflow at step={self._step}")
        # t = self._step
        # self.obs[t] = obs
        # self.actions_raw[t] = action_raw
        # self.mus[t] = mu
        # self.log_sigmas[t] = log_sigma
        # self.log_probs[t] = log_prob
        # self.values[t] = value
        # self.rewards[t] = reward
        # self.dones[t] = done
        # self._step += 1
        pass

    @abstractmethod
    def get_flat(self) -> Dict[str, torch.Tensor]:
        # n = self.horizon * self.num_envs
        # return {
        #     "obs": self.obs.reshape(n, self.obs_dim),
        #     "actions_raw": self.actions_raw.reshape(n, self.action_dim),
        #     "mus": self.mus.reshape(n, self.action_dim),
        #     "log_sigmas": self.log_sigmas.reshape(n, self.action_dim),
        #     "log_probs": self.log_probs.reshape(n),
        #     "values": self.values.reshape(n),
        #     "rewards": self.rewards.reshape(n),
        #     "dones": self.dones.reshape(n),
        #     "returns": self.returns.reshape(n),
        #     "advantages": self.advantages.reshape(n),
        # }
        pass

    # ------------------------------------------------------------------
    # Concrete: GAE + episode-length helpers (pure math, generic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_gae(
        self, last_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95
    ) -> None:
        """Standard GAE with done-masking and bootstrap from `last_value`.

        Writes `self.returns` and `self.advantages`. `last_value` shape `(num_envs,)`
        is the bootstrap value `V(s_{horizon})`.
        """
        adv = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        for t in reversed(range(self.horizon)):
            non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.horizon - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            adv = delta + gamma * lam * non_terminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    @torch.no_grad()
    def episode_length_estimate(self) -> float:
        """Approximate mean episode length (steps until first done in this rollout).

        For each env, count steps until the FIRST done within the current
        rollout; envs that never reset contribute the full horizon. The
        result under-counts steady-state episode length when episodes span
        many rollouts (rl_games' `episode_lengths/iter` is more exact), but
        the magnitude is comparable for diagnostics.
        """
        any_done = self.dones.any(dim=0)                # (num_envs,)
        first_done = self.dones.argmax(dim=0).float()   # (num_envs,) — 0 if none
        first_done = torch.where(
            any_done,
            first_done + 1.0,                            # +1: step index → length
            torch.full_like(first_done, float(self.horizon)),
        )
        return float(first_done.mean().item())
