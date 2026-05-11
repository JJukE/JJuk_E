"""In-process launcher skeleton for env-driven (RL) jjuke consumers.

Counterpart to `jjuke/dl/main.py`, which spawns
`accelerate launch <train.py>` as a subprocess for multi-process DDP.
The RL pattern can't use that subprocess hop when the env requires
in-process bootstrap (e.g. Isaac Sim's `AppLauncher` must run in the
same Python process that imports the env modules; see the worker at
`jjuke/rl/train.py`). The RL launcher therefore stays single-process
and dispatches IN-PROCESS to `jjuke.rl.train.train(...)`.

Skeleton style (matches the rest of jjuke): substantial wiring +
commented reference impl. A reference consumer (Isaac Sim Lab + the
InterMimic Teacher / Student stack) lives at
`isaaclab/main.py`.

Minimal CLI: `--config_file --gpus --debug --headless`. Every other
setting (num_envs, learning_rate, wandb toggle, resume_ckpt, …) lives
in the YAML; use `__parent__:` to make variant configs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `jjuke` importable when invoked as `python jjuke/rl/main.py`
# (the canonical invocation goes through Isaac Lab's wrapper, e.g.
# `isaaclab.sh -p jjuke/rl/main.py`).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from jjuke import options  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="jjuke RL launcher (in-process).")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to the project YAML.")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated CUDA device indices "
                             "(single-process: only the first is honored).")
    parser.add_argument("--debug", action="store_true",
                        help="jjuke debug mode: 4 epochs, save_period=2.")
    parser.add_argument("--headless", action="store_true",
                        help="Run env headless (no viewer). Drop this flag if "
                             "your env has no 'headless' concept.")
    opt = parser.parse_args()

    args, yaml_path = options.get_config(
        opt.config_file, opt.gpus, debug=opt.debug, save=True
    )
    print(f"[jjuke.rl] Config saved to {yaml_path}")
    print(f"[jjuke.rl] exp_path = {args.exp_path}")

    # Import after get_config so the timestamped exp dir exists for train.log.
    from jjuke.rl.train import train  # noqa: WPS433
    train(args_file=yaml_path, headless=opt.headless)


if __name__ == "__main__":
    main()
