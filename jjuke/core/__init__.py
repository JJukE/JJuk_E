"""Cross-cutting base layer of the jjuke framework.

`trainer` ships `BaseTrainer` (Accelerator-wrapped training loop) and
`BasePreprocessor` (batch-to-device + augmentation hook). Both are
shared by `jjuke.dl` consumers (DataLoader-iterating training) and
`jjuke.rl` consumers (env-iterating training, via `BaseRLTrainer`
which subclasses `BaseTrainer`).

`normalizers` ships `RunningMeanStd` (Welford running mean/var) plus
`BaseObsNormalizer` skeleton — used by RL for observation
normalization, but generic enough that DL consumers could reuse them
for input normalization too.
"""

from . import trainer, normalizers

__all__ = ["trainer", "normalizers"]
