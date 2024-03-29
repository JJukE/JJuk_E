import os
import time
import random
from collections import defaultdict

import numpy as np
import torch
from easydict import EasyDict
# from torch import Tensor
# from torchvision.utils import make_grid

__all__ = [
    "AverageMeter",
    "AverageMeters",
    "seed_everything",
    "find_free_port",
    "get_model_params",
    "try_remove_file"
]


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
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True # usually False?
        torch.backends.cudnn.deterministic = False # usually True?


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


# def tensor_to_image(images, nrow):
#     # images: b 3 h w, [-1, 1]
#     grid = make_grid(images, nrow=nrow).permute(1, 2, 0)  # H W 3 [-1, 1]
#     # (x+1)/2 * 255 + 0.5 = 127.5x + 128, (반올림이 되게 하기 위해 0.5를 더함, 안 더하면 내림이 됨)
#     grid = grid.mul_(127.5).add_(128).clamp_(0, 255).to("cpu", torch.uint8).numpy()
#     return grid


# class BlackHole(int):
#     def __setattr__(self, *args, **kwargs):
#         pass

#     def __call__(self, *args, **kwargs):
#         return self

#     def __getattr__(self, *args, **kwargs):
#         return self

#     def __enter__(self, *args, **kwargs):
#         return self

#     def __exit__(self, *args, **kwargs):
#         return self

#     def __getitem__(self, *args, **kwargs):
#         return self


# def tensor_to_image(x: Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#     """
#     ### input
#     - x: (b 3 h w) or (3 h w)
#     """
#     xdim = x.dim()
#     if xdim == 3:
#         x = x[None]

#     if not isinstance(mean, Tensor):
#         mean = x.new_tensor(mean).view(1, 3, 1, 1)
#     else:
#         mean = mean.to(x).view(1, 3, 1, 1)
#     if not isinstance(std, Tensor):
#         std = x.new_tensor(std).view(1, 3, 1, 1)
#     else:
#         std = std.to(x).view(1, 3, 1, 1)
#     x = x * std + mean
#     x = x.detach().mul(255).add_(0.5).clamp_(0, 255).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

#     if xdim == 3:
#         x = x[0]

#     return x


def try_remove_file(file):
    for _ in range(10):
        try:
            os.remove(file)
            break
        except:
            print("Warn: Failed to remove", file)
            time.sleep(0.1)
