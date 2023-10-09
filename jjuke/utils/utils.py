import importlib
import os
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from easydict import EasyDict
from torch import Tensor
from torchvision.utils import make_grid

__all__ = [
    "BlackHole",
    "CheckpointManager",
    "AverageMeter",
    "AverageMeters",
    "seed_everything",
    "find_free_port",
    "get_model_params",
    "get_data_iterator",
    "instantiate_from_config",
    "get_obj_from_str",
    "tensor_to_image",
    "print_model"
]

class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


class CheckpointManager(object):
    """Checkpoint manager
    Checkpoint should be saved as "ckpt_[iteration or epoch].pt(h)"
    """

    def __init__(self, save_dir, isTrain=False, logger=BlackHole()):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

        if not isTrain:
            for f in os.listdir(self.save_dir):
                if f[:4] != 'ckpt':
                    continue
                _, score, it = f.split('_')
                it = it.split('.')[0]
                self.ckpts.append({
                    'score': float(score),
                    'file': f,
                    'iteration': int(it),
                })

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float('-inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] >= worst:
                idx = i
                worst = ckpt['score']
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = float('inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] <= best:
                idx = i
                best = ckpt['score']
        return idx if idx >= 0 else None
        
    def get_latest_ckpt_idx(self):
        idx = -1
        latest_it = -1
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['iteration'] > latest_it:
                idx = i
                latest_it = ckpt['iteration']
        return idx if idx >= 0 else None

    def save(self, model, args, score, others=None, step=None):

        if step is None:
            fname = 'ckpt_{}_.pt'.format(float(score))
        else:
            fname = 'ckpt_{}.pt'.format(int(step))
        path = os.path.join(self.save_dir, fname)

        torch.save({
            'args': args,
            'state_dict': model.state_dict(),
            'others': others
        }, path)

        self.ckpts.append({
            'score': score,
            'file': fname
        })

        return True

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt
    
    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt

    def load_selected(self, file):
        ckpt = torch.load(os.path.join(self.save_dir, file))
        return ckpt


def print_model(model, verbose=False, print_flag=False):
    """Print the total number of parameters in the network and (if verbose) network architecture

    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    message = 'Model info\n'
    message += '---------- Networks initialized -------------\n'
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    if verbose:
        message += repr(model)
    message += '[Network {}] Total number of parameters : {:.3f} M\n'.format(type(model).__name__, num_params / 1e6)
    message += '-----------------------------------------------'

    if print_flag:
        print(message)
    return message


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, val, n=1):
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt

    def get(self):
        return self.avg

    def __call__(self):
        return self.avg


class AverageMeters:
    def __init__(self, *keys) -> None:
        # self.data = OrderedDict({key: AverageMeter() for key in keys})
        self.data = defaultdict(AverageMeter)
        for k in keys:
            self.data[k]

    def __getitem__(self, key):
        return self.data[key]()

    def __getattr__(self, key):
        return self.data[key]()

    def update_dict(self, n, g):
        for k, v in g.items():
            self.data[k].update(v, n)

    def _get(self, k):
        if k in self.data:
            return f"{self.data[k]():.4f}"
        else:
            return "_"

    def to_msg(self, format="%s:%.4f"):
        msgs = []
        for k, v in self.data.items():
            if k == "loss":
                msgs = [format % (k, v())] + msgs
            else:
                msgs.append(format % (k, v()))
        return " ".join(msgs)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def get_model_params(model):
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    return model_size


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


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


def tensor_to_image(images, nrow):
    # images: b 3 h w, [-1, 1]
    grid = make_grid(images, nrow=nrow).permute(1, 2, 0)  # H W 3 [-1, 1]
    # (x+1)/2 * 255 + 0.5 = 127.5x + 128, (반올림이 되게 하기 위해 0.5를 더함, 안 더하면 내림이 됨)
    grid = grid.mul_(127.5).add_(128).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    return grid


def tensor_to_image(x: Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    ### input
    - x: (b 3 h w) or (3 h w)
    """
    xdim = x.dim()
    if xdim == 3:
        x = x[None]

    if not isinstance(mean, Tensor):
        mean = x.new_tensor(mean).view(1, 3, 1, 1)
    else:
        mean = mean.to(x).view(1, 3, 1, 1)
    if not isinstance(std, Tensor):
        std = x.new_tensor(std).view(1, 3, 1, 1)
    else:
        std = std.to(x).view(1, 3, 1, 1)
    x = x * std + mean
    x = x.detach().mul(255).add_(0.5).clamp_(0, 255).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    if xdim == 3:
        x = x[0]

    return x
