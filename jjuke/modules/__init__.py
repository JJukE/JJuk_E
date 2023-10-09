from functools import wraps
from inspect import isfunction

import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from jjuke.utils.resize_right import resize

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        # return itself when it is already tuple
        return t
    # return converted tuple given element and length
    return ((t,) * length)


def resize_feature_to(data, target_size, clamp_range = None, nearest = False, **kwargs):
    """ Resize 1D features or 2D feature maps to given target size. """
    orig_size = data.shape[-1] # 2D -> h or w // 1D -> n
    if orig_size == target_size:
        return data
    
    if not nearest:
        scale_factors = target_size / orig_size
        out = resize(data, scale_factors=scale_factors, **kwargs)
    else:
        out = F.interpolate(data, target_size, mode="nearest")
    
    if clamp_range is not None:
        out = out.clamp(*clamp_range)
    
    return out


# def maybe(fn):
#     @wraps(fn)
#     def inner(x, *args, **kwargs):
#         if x is None:
#             return x
#         return fn(x, *args, **kwargs)
#     return inner


# def make_checkpointable(fn, **kwargs):
#     if isinstance(fn, nn.ModuleList):
#         return [maybe(make_checkpointable)(el, **kwargs) for el in fn]

#     condition = kwargs.pop('condition', None)

#     if condition is not None and not condition(fn):
#         return fn

#     @wraps(fn)
#     def inner(*args):
#         input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

#         if not input_needs_grad:
#             return fn(*args)

#         return checkpoint(fn, *args)

#     return inner