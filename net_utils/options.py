"""
Copyright (c) 2022 Kitsunetic, https://github.com/Kitsunetic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

from easydict import EasyDict
from omegaconf import DictConfig, ListConfig, OmegaConf

from net_utils import get_obj_from_str, instantiate_from_config


def _load_yaml_recursive(cfg):
    keys_to_del = []
    for k in cfg.keys():
        if k == "__parent__":
            if isinstance(cfg[k], ListConfig):
                cfg2 = load_yaml(cfg[k][0])
                path = cfg[k][1].split(".")
                for p in path:
                    cfg2 = cfg2[p]
            else:
                cfg2 = load_yaml(cfg[k])

            keys_to_del.append(k)
            cfg = OmegaConf.merge(cfg2, cfg)
        elif isinstance(cfg[k], DictConfig):
            cfg[k] = _load_yaml_recursive(cfg[k])

    for k in keys_to_del:
        del cfg[k]

    return cfg


def _postprocess_yaml_recursive(cfg):
    for k in cfg.keys():
        if k == "__pycall__":
            cfg2 = instantiate_from_config(cfg[k])
            return cfg2
        elif k == "__pyobj__":
            if isinstance(cfg[k], dict):
                cfg2 = get_obj_from_str(cfg[k]["target"])
            else:
                cfg2 = get_obj_from_str(cfg[k])
            return cfg2
        elif isinstance(cfg[k], dict):
            cfg[k] = _postprocess_yaml_recursive(cfg[k])

    return cfg


def load_yaml(path):
    cfg = OmegaConf.load(path)
    cfg = _load_yaml_recursive(cfg)
    # cfg = _postprocess_yaml_recursive(cfg)
    return cfg


def get_config(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--gpus", type=str)
    parser.add_argument("--no_vscode_debug", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--outdir")

    opt, unknown = parser.parse_known_args(argv)

    if not opt.no_vscode_debug:
        opt.config_file = "./configs/DiffuScene/diffuscene.yaml"
        opt.gpus = "0"
        opt.outdir = "/root/hdd1/SGTD/diffuscene_implementation"

    cfg = load_yaml(opt.config_file)
    cli = OmegaConf.from_dotlist(unknown)
    args = OmegaConf.merge(cfg, cli)

    args.gpus = list(map(int, opt.gpus.split(",")))
    args.debug = opt.debug
    args.outdir = opt.outdir

    n = datetime.now()
    # timestr = "{}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(
    #     n.year%100, n.month, n.day, n.hour, n.minute, n.second
    # )
    timestr = "{}{:02d}{:02d}_{:02d}{:02d}".format(
        n.year%100, n.month, n.day, n.hour, n.minute
    )
    # timestr = "{}{:02d}{:02d}".format(
    #     n.year%100, n.month, n.day
    # )
    timestr += "_" + Path(opt.config_file).stem
    if args.memo:
        timestr += "_%s" % args.memo
    if args.debug:
        timestr += "_debug"

    args.exp_path = os.path.join(args["exp_dir"], timestr)
    (Path(args.exp_path) / "samples").mkdir(parents=True, exist_ok=True)
    print("Start on exp_path:", args.exp_path)

    with open(os.path.join(args.exp_path, "args.yaml"), "w") as f:
        OmegaConf.save(args, f)

    print(OmegaConf.to_yaml(args, resolve=True))
    args = OmegaConf.to_container(args, resolve=True)
    args = EasyDict(args)
    args.exp_path = Path(args.exp_path)

    args = _postprocess_yaml_recursive(args)

    return args
