"""Observation / input normalizers — generic across DL and RL.

Two surfaces:
    * `RunningMeanStd` (concrete) — Welford parallel algorithm, GPU-resident
      mean/var/count buffers, used to track input statistics during training.
    * `BaseObsNormalizer(BasePreprocessor)` (skeleton) — abstract `__call__`
      that subclasses fill in. Documents the `state_dict` / `load_state_dict`
      resume contract so checkpoints round-trip the running statistics.

Reference impl (RL-side consumer that wires `RunningMeanStd` into the
`BasePreprocessor` plumbing):
    `isaaclab/model/preprocessor.py` — `ObsNormalizer`.
"""
from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn

from .trainer import BasePreprocessor


class RunningMeanStd(nn.Module):
    """Online running mean / variance via Welford's parallel algorithm.

    Buffers (`mean`, `var`, `count`) are registered with `nn.Module` so they
    move with `.to(device)` and round-trip through `state_dict()` /
    `load_state_dict()`. Use `update(x)` to fold a fresh batch of obs into
    the statistics, and `normalize(x)` to apply `(x - mean) / sqrt(var + eps)`.
    """

    def __init__(self, shape, epsilon: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running mean/var with a fresh batch.

        `x` may be either `(*shape)` (single sample) or `(B, *shape)`.
        Implementation follows the parallel Welford merge formulation.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.fill_(float(tot_count.item()))

    def normalize(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + eps)


class BaseObsNormalizer(BasePreprocessor):
    """`BasePreprocessor` skeleton dedicated to observation normalization.

    Subclasses are expected to:
        * Hold a `RunningMeanStd` (or equivalent) at `self.rms`.
        * Implement `__call__(obs, augmentation=False)` to update RMS when
          `augmentation=True` (the rollout-collection phase) and otherwise
          just normalize.
        * Surface `state_dict()` / `load_state_dict(state)` so the trainer's
          save hook can persist running statistics next to the Accelerator
          checkpoint and `_load_state_extras` can restore them on resume.

    Reference impl:
        `isaaclab/model/preprocessor.py:ObsNormalizer`.
    """

    rms: "RunningMeanStd | None"

    @abstractmethod
    def __call__(self, obs, augmentation: bool = False) -> torch.Tensor:
        # obs_t = self._to(obs).float()
        # if self.rms is None:
        #     self._init_rms(obs_t.shape[-1])
        # if augmentation:
        #     self.rms.update(obs_t)
        # return self.rms.normalize(obs_t)
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        # return {f"rms.{k}": v for k, v in self.rms.state_dict().items()}
        pass

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        # rms_state = {k[len("rms."):]: v for k, v in state.items() if k.startswith("rms.")}
        # self.rms.load_state_dict(rms_state)
        pass
