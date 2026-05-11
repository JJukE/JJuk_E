"""jjuke RL framework — algorithm-agnostic env-driven trainer skeletons.

Subpackage contents:
    base_trainer.BaseRLTrainer     — env loop scaffold (subclass for PPO / DAgger / etc.)
    buffers.BaseRolloutBuffer      — fixed-size on-policy buffer + GAE
    policies.BasePolicy            — diagonal-Gaussian policy + log_prob/entropy/KL helpers

All three are SKELETONS in the same sense as `jjuke.core.trainer.BaseTrainer`:
they're substantial scaffolds with abstract methods + commented-out reference
impls. A reference concrete implementation (PPO + DAgger over an Isaac Lab
env) lives in `isaaclab/model/`.
"""

from .base_trainer import BaseRLTrainer
from .buffers import BaseRolloutBuffer
from .policies import BasePolicy

__all__ = ["BaseRLTrainer", "BaseRolloutBuffer", "BasePolicy"]
