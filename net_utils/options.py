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
import os
import argparse
import importlib
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from easydict import EasyDict
from omegaconf import DictConfig, ListConfig, OmegaConf


def _parse_pyinstance_dict(params: dict):
    out_dict = EasyDict()

    for p, v in params.items():
        if p == "__pyinstance__":
            inst = instantiate_from_config(v)
            return inst
        elif isinstance(v, dict):
            out_dict[p] = _parse_pyinstance_dict(v)
        elif isinstance(v, (list, tuple)):
            out_dict[p] = _parse_pyinstance_list(v)
        else:
            out_dict[p] = v

    return out_dict


def _parse_pyinstance_list(params: list):
    out_list = []

    for v in params:
        if isinstance(v, dict):
            out_list.append(_parse_pyinstance_dict(v))
        elif isinstance(v, (list, tuple)):
            out_list.append(_parse_pyinstance_list(v))
        else:
            out_list.append(v)

    return out_list


def instantiate_from_config(config: dict, *args, **kwargs):
    config = deepcopy(config)

    # https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/util.py#L78
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")

    # parse __pyinstance__
    argums = config.get("argums", list())
    argums = _parse_pyinstance_list(argums)
    params = config.get("params", dict())
    params = _parse_pyinstance_dict(params)

    return get_obj_from_str(config["target"])(*argums, *args, **params, **kwargs)


def get_obj_from_str(string, reload=False):
    # https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/util.py#L88
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--outdir")

    opt, unknown = parser.parse_known_args(argv)

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
    if "memo" in args.keys():
        timestr += "_%s" % args.memo
    
    if args.debug:
        timestr += "_debug"

    if "exp_dir" in args.keys():
        args.exp_path = os.path.join(args["exp_dir"], timestr)
        (Path(args.exp_path) / "samples").mkdir(parents=True, exist_ok=True)
        print("Start on exp_path:", args.exp_path)

        with open(os.path.join(args.exp_path, "args.yaml"), "w") as f:
            OmegaConf.save(args, f)

        print(OmegaConf.to_yaml(args, resolve=True))
        args = OmegaConf.to_container(args, resolve=True)
        args = EasyDict(args)
        args.exp_path = Path(args.exp_path)
    else:
        print("There's no any experiment directory in the config file.")

    args = _postprocess_yaml_recursive(args)

    return args
