"""Worker skeleton for env-driven (RL) jjuke consumers.

Invoked by `jjuke/rl/main.py` (in-process — no subprocess accelerate
launch). The skeleton order matters for env-bootstrap correctness:
the env-specific bootstrap (e.g. Isaac Sim's `AppLauncher`) MUST run
BEFORE any module that imports the env, otherwise C++ extensions don't
load and env construction fails.

Reference consumer:
    `isaaclab/train.py` fills the env-bootstrap block
    with Isaac Sim's `AppLauncher` and the MJCF-importer extension
    enable for headless mode.

Subclass / fork contract:
    1. Replace the commented "Env-specific bootstrap" block with your
       env's actual init (e.g. `AppLauncher`, `gym.make(...)`,
       `mujoco_py.load_model_from_xml(...)`, etc.).
    2. Keep the rest of the flow as-is — `options.instantiate_from_config`
       + `trainer.fit()` is generic across RL trainers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from easydict import EasyDict
from omegaconf import OmegaConf

# Make `jjuke` importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def train(args_file: str, headless: bool = True) -> None:
    args = EasyDict(OmegaConf.load(args_file))

    if "gpus" in args and len(args.gpus) == 1:
        torch.cuda.set_device(args.gpus[0])

    # ------------------------------------------------------------------
    # Env-specific bootstrap goes HERE — BEFORE importing any env module.
    # ------------------------------------------------------------------
    # Example for Isaac Sim (see isaaclab/train.py for the
    # full reference implementation, including the MJCF-importer
    # extension enable for headless mode):
    #
    #   from isaaclab.app import AppLauncher
    #   app_launcher = AppLauncher({"headless": headless})
    #   simulation_app = app_launcher.app
    #
    # For a Gym env (no Sim bootstrap needed):
    #
    #   import gymnasium as gym  # the env class import is fine here
    #
    # For a MuJoCo env or other physics framework, follow the framework's
    # own initialization order (typically: load model → make env).
    # ------------------------------------------------------------------

    # Now safe to import jjuke + env modules.
    from jjuke import logger, options  # noqa: WPS433

    logger.basic_config(Path(args.exp_path) / "train.log")
    args.log = logger.get_logger()
    args.log.info(f"[jjuke.rl] Loaded args from {args_file}")
    args.log.info(f"[jjuke.rl] exp_path={args.exp_path}, headless={headless}")

    # Build trainer via cfg `target+params` (must subclass jjuke.rl.BaseRLTrainer
    # — or any concrete RL trainer that fills the env-loop scaffolding).
    trainer = options.instantiate_from_config(args.trainer, args)
    args.log.info(f"[jjuke.rl] Trainer instantiated: {type(trainer).__name__}")

    # Run training.
    trainer.fit()

    # ------------------------------------------------------------------
    # Cleanup — close the env / Sim app if the bootstrap created one.
    # ------------------------------------------------------------------
    args.log.info("[jjuke.rl] fit() completed; existing.")
    os._exit(0)


if __name__ == "__main__":
    # Allow direct invocation (skips main.py's get_config). Useful when
    # resuming from a saved args.yaml.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--args_file", type=str, required=True)
    parser.add_argument("--headless", action="store_true")
    opt = parser.parse_args()
    train(args_file=opt.args_file, headless=opt.headless)
