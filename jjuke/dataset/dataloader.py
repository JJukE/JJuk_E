from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ExampleDataset(Dataset):
    def __init__(
        self,
        ...,
    ):
        pass


    def __len__(self):
        pass
    
    
    def __getitem__(self, idx):
        pass


def load_dataloaders(
    batch_size, num_workers, ..., **kwargs
):
    # [local diffusion, global diffusion, ld finetuning]
    ds_train = ExampleDataset(...)
    ds_valid = ExampleDataset(...)
    # ds_test = ExampleDataset(...)

    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    # dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dl_train, dl_valid #, dl_test


def __test__():
    pass
    

if __name__ == "__main__":
    __test__()
