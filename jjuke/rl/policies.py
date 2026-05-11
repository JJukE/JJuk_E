"""Policy networks for jjuke's RL framework.

This module ships ONE skeleton (`BasePolicy`) for diagonal-Gaussian
continuous-action policies, plus the static math helpers a subclass needs.
The math helpers are concrete because they're pure (no model state) — they
match the rl_games convention so checkpoints / metrics round-trip cleanly
between the two stacks.

A concrete subclass implements the actor + critic forward path. See
the reference impl `isaaclab/model/network.py:ActorCritic`
(MLP, separate actor/critic, fixed log-sigma) and
`TransformerActorCritic` (4-token transformer encoder).
"""
from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BasePolicy(nn.Module, metaclass=ABCMeta):
    """Abstract continuous-action policy with diagonal Gaussian distribution.

    Subclass contract:
        * `forward(obs) → (mu, log_sigma, value)` — `mu` and `log_sigma`
          shape `(B, action_dim)`, `value` shape `(B,)`.
        * `act(obs, deterministic=False)` — sample (or return mu) and
          return `(action, log_prob, value, mu, log_sigma)`.

    Free helpers (pure math, no module state — call as
    `BasePolicy.log_prob(...)` etc.):
        * `log_prob(action, mu, log_sigma) → (B,)`
        * `entropy(log_sigma) → (B,) or scalar`
        * `gaussian_kl(mu_old, log_sigma_old, mu_new, log_sigma_new) → (B,)`

    Reference impl:
        `isaaclab/model/network.py:ActorCritic` (MLP)
        `isaaclab/model/network.py:TransformerActorCritic`
    """

    @abstractmethod
    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # actor_feat = self.actor_mlp(obs)
        # critic_feat = self.critic_mlp(obs)
        # mu = self.mu_head(actor_feat)
        # log_sigma = mu * 0.0 + self.log_sigma   # broadcast (B, action_dim)
        # value = self.value_head(critic_feat).squeeze(-1)
        # return mu, log_sigma, value
        pass

    @abstractmethod
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample (or take mu) and return diagnostics. Decorate with `@torch.no_grad()`."""
        # @torch.no_grad()  # decorate the override with this
        # mu, log_sigma, value = self.forward(obs)
        # if deterministic:
        #     action = mu
        # else:
        #     std = log_sigma.exp()
        #     action = mu + std * torch.randn_like(std)
        # log_prob = BasePolicy.log_prob(action, mu, log_sigma)
        # return action, log_prob, value, mu, log_sigma
        pass

    # ------------------------------------------------------------------
    # Pure-math helpers (concrete)
    # ------------------------------------------------------------------

    @staticmethod
    def log_prob(
        action: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor
    ) -> torch.Tensor:
        """Diagonal-Gaussian log-prob, summed over action dims. Returns `(B,)`."""
        var = (2.0 * log_sigma).exp()
        log_prob = -0.5 * (
            (action - mu).pow(2) / var + 2.0 * log_sigma + math.log(2.0 * math.pi)
        )
        return log_prob.sum(dim=-1)

    @staticmethod
    def entropy(log_sigma: torch.Tensor) -> torch.Tensor:
        """Differential entropy of `N(mu, exp(log_sigma))`, summed over action dims."""
        return (log_sigma + 0.5 * math.log(2.0 * math.pi * math.e)).sum(dim=-1)

    @staticmethod
    def gaussian_kl(
        mu_old: torch.Tensor,
        log_sigma_old: torch.Tensor,
        mu_new: torch.Tensor,
        log_sigma_new: torch.Tensor,
    ) -> torch.Tensor:
        """Analytic `KL( N(mu_old, σ_old) || N(mu_new, σ_new) )`, summed over action dims.

        Matches rl_games' `info/kl` convention (signed, magnitude directly
        comparable). Subclasses' PPO update typically logs
        `gaussian_kl(...).mean()` per minibatch.
        """
        var_old = (2.0 * log_sigma_old).exp()
        var_new = (2.0 * log_sigma_new).exp()
        kl = (
            log_sigma_new
            - log_sigma_old
            + (var_old + (mu_old - mu_new).pow(2)) / (2.0 * var_new)
            - 0.5
        )
        return kl.sum(dim=-1)
