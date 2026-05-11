"""DL-specific conveniences for the jjuke framework.

Currently exposes the `dataset` example subpackage (`ExampleDataset` +
`load_dataloaders` factory) — a template for users who train via the
DataLoader-iterating pattern (supervised, self-supervised, generative,
contrastive, diffusion, ...). The shared `BaseTrainer` lives in
`jjuke.core.trainer`.
"""

from . import dataset

__all__ = ["dataset"]
