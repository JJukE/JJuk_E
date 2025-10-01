import os
import argparse
import subprocess

import torch.multiprocessing as mp

from jjuke import options

"""
see the documents for settings:
https://huggingface.co/docs/accelerate/package_reference/cli

Before the first time to execute this command, you should configure with `accelerate config` to use no distributed, 1 gpu as default.

EX)
python main.py --config_file config/MyModel/MyConfig.yaml --gpus 0,1
"""

def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_rtx_4000(gpus_indices):
    try:
        gpu_names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], encoding="utf-8")
        gpu_names = [gpu_names.strip().split("\n")[i] for i in gpus_indices]
        for name in gpu_names:
            if "RTX 40" in name:
                return True
        return False
    except FileNotFoundError:
        print("nvidia-smi not found. Ensure NVIDIA drivers are installed.")
        return False


def format_args(config):
    assert isinstance(config, dict)
    args = []
    for k, v in config.items():
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        elif isinstance(v, list):
            args.append(f"--{k} " + " ".join(map(str, v)))
        elif v is not None:
            args.append(f"--{k} {v}")
    return " ".join(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--gpus", type=str)
    parser.add_argument("--debug", action="store_true")
    
    opt = parser.parse_args()
    
    # DDP setting
    args, yaml_path = options.get_config(opt.config_file, opt.gpus, opt.debug, save=True)
    
    # if is_rtx_4000(args.gpus):
    #     os.environ["NCCL_P2P_DISABLE"] = "1"
    #     os.environ["NCCL_IB_DISABLE"] = "1"
    
    # NOTE: if nccl backend has an issue (infinitely waiting, etc.), try these:
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    
    if len(args.gpus) > 1:
        print(f"Multi gpu training with {opt.gpus}th gpus...")
        accel_configs = {
            "multi_gpu": True,
            "num_machines": 1, # TODO: if you want to train multiple machines, implement it.
            "num_processes": len(args.gpus),
            "num_cpu_threads_per_process": min(args.dataset.params.num_workers, mp.cpu_count()),
            "main_process_port": find_free_port(),
            "gpu_ids": opt.gpus
        }
    else:
        # NOTE: check if accelerate use `No distributed training` as default.
        print(f"Single gpu training with {opt.gpus}th gpu...")
        accel_configs = {
            "multi_gpu": False,
            "num_machines": 1,
            "num_processes": 1,
            "num_cpu_threads_per_process": min(args.dataset.params.num_workers, mp.cpu_count()),
            "main_process_port": find_free_port(),
        }
    
    # with open(Path(args.exp_path) / "accelerate_config.yaml", "w") as f:
    #     yaml.dump(accel_configs, f, sort_keys=False)
    
    script_configs = {"args_file": yaml_path}
    
    train_cmd = ["accelerate", "launch"] \
        + format_args(accel_configs).split() \
        + ["train.py"] \
        + format_args(script_configs).split()
    
    try:
        result = subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\33[101mError occured (exit {e.returncode})\33[0m")
    except KeyboardInterrupt:
        print("\33[101mkill \33[0m")
