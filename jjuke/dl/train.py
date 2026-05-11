import argparse
from pathlib import Path

import torch
from easydict import EasyDict
from omegaconf import OmegaConf

from jjuke import logger, options


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_file", type=str)
    
    opt = parser.parse_args()
    args = EasyDict(OmegaConf.load(opt.args_file))
    
    if len(args.gpus) == 1:
        torch.cuda.set_device(args.gpus[0])
    
    logger.basic_config(Path(args.exp_path) / "train.log")
    args.log = logger.get_logger()
    
    trainer = options.instantiate_from_config(args.trainer, args)
    trainer.fit()


if __name__ == "__main__":
    train()