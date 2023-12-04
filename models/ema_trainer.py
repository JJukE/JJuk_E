import math
from collections import OrderedDict
from typing import Union
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from functools import reduce

import torch
import torch.distributed as dist
from torch import nn

from ..net_utils import try_remove_file
from ..net_utils.dist import reduce_dict
from .trainer import BaseTrainer, StepTrainer


def ema(source: Union[OrderedDict, nn.Module], target: Union[OrderedDict, nn.Module], decay: float):
    if isinstance(source, nn.Module):
        source = source.state_dict()
    if isinstance(target, nn.Module):
        target = target.state_dict()
    for key in source.keys():
        target[key].data.copy_(target[key].data * decay + source[key].data * (1 - decay))


class BaseTrainerEMA(BaseTrainer):
    def __init__(self, *args, ema_decay: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay

        self._ema_state = False

    @contextmanager
    def ema_state(self, activate=True):
        previous = self._ema_state
        self._ema_state = activate
        yield
        self._ema_state = previous

    def build_network(self):
        super().build_network()
        self.model_ema = deepcopy(self.model_src)
        self.model_ema.load_state_dict(self.model_src.state_dict())
        self.model_ema.eval().requires_grad_(False)

    def load_checkpoint(self, ckpt):
        super().load_checkpoint(ckpt)
        if "model_ema" in ckpt:
            self.model_ema.load_state_dict(ckpt["model_ema"])

    def on_train_batch_end(self, s):
        super().on_train_batch_end(s)
        ema(self.model_src, self.model_ema, self.ema_decay)

    def save(self, out_path):
        data = {
            "optim": self.optim.state_dict(),
            "model": self.model_src.state_dict(),
            "model_ema": self.model_ema.state_dict(),
            "epoch": self.epoch,
        }
        torch.save(data, str(out_path))


class StepTrainerEMA(StepTrainer):
    def __init__(self, *args, ema_decay: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay

        self._ema_state = False
        self.best_ema_epoch = -1
        self.best_ema = math.inf if self.small_is_better else -math.inf

    @contextmanager
    def ema_state(self, activate=True):
        previous = self._ema_state
        self._ema_state = activate
        yield
        self._ema_state = previous

    def build_network(self):
        super().build_network()
        self.model_ema = deepcopy(self.model_src)
        self.model_ema.load_state_dict(self.model_src.state_dict())
        self.model_ema.requires_grad_(False)

    def load_checkpoint(self, ckpt):
        super().load_checkpoint(ckpt)
        if "model_ema" in ckpt:
            self.model_ema.load_state_dict(ckpt["model_ema"])

    def on_train_batch_end(self, s):
        super().on_train_batch_end(s)
        ema(self.model_src, self.model_ema, self.ema_decay)

    def save(self, out_path):
        data = {
            "optim": self.optim.state_dict(),
            "model": self.model.state_dict(),
            "epoch": self.epoch,
        }
        torch.save(data, str(out_path))

    @torch.no_grad()
    def evaluation_ema(self, *o_lst):
        # self.step_sched(o_lst[0][self.monitor], is_on_epoch=True)

        improved = False
        if self.rankzero:  # scores are not calculated in other nodes
            flag = ""
            _c1 = self.small_is_better and o_lst[0][self.monitor] < self.best_ema
            _c2 = not self.small_is_better and o_lst[0][self.monitor] > self.best_ema
            _c3 = (
                self.sample_at_least_per_epochs is not None
                and (self.epoch - self.best_ema_epoch) >= self.sample_at_least_per_epochs
            )

            if _c1 or _c2 or _c3:
                if _c1:
                    self.best_ema = o_lst[0][self.monitor]
                elif _c2:
                    self.best_ema = max(self.best_ema, o_lst[0][self.monitor])

                improved = True

                self.best_ema_epoch = self.epoch
                self.save(self.args.exp_path / "best_ema_ep{:06d}.pth".format(self.epoch))
                saved_files = sorted(list(self.args.exp_path.glob("best_ema_ep*.pth")))
                if len(saved_files) > self.num_saves:
                    to_deletes = saved_files[: len(saved_files) - self.num_saves]
                    for to_delete in to_deletes:
                        try_remove_file(str(to_delete))

                flag = "*"
                improved = self.epoch > self.epochs_to_save or self.args.debug or not self.save_only_improved

            msg = f"Step-EMA[%06d/%06d]" % (self.epoch, self.args.epochs)
            msg += f" {self.monitor}[" + ";".join([o._get(self.monitor) for o in o_lst]) + "]"
            msg += " (best:%.4f%s)" % (self.best_ema, flag)

            keys = reduce(lambda x, o: x | set(o.data.keys()), o_lst, set())
            keys = sorted(list(filter(lambda x: x != self.monitor, keys)))

            for k in keys:
                msg += f" {k}[" + ";".join([o._get(k) for o in o_lst]) + "]"

            self.log.info(msg)
            self.log.flush()

        # share improved condition with other nodes
        if self.ddp:
            improved = torch.tensor([improved], device="cuda")
            dist.broadcast(improved, 0)

        return improved

    def sample(self, is_ema: bool):
        pass

    @torch.no_grad()
    def stage_eval(self, o_train):
        o_valid, losses_dict = self.valid_epoch(self.dl_valid)
        
        # wandb logging for each step
        if self.rankzero and self.args.logging.use_wandb:
            loss_reduced = reduce_dict(losses_dict)
            losses_dict = {k: v.mean().item() if hasattr(v, "mean") else v for k, v in loss_reduced.items()}
            self.log_wandb(losses_dict, "valid", epoch=self.epoch)
        
        improved = self.evaluation(o_valid, o_train)
        
        if improved:
            self.sample(is_ema=False)

        with self.ema_state():
            o_valid_ema, losses_dict_ema = self.valid_epoch(self.dl_valid)
            
            # wandb logging for each step
            if self.rankzero and self.args.logging.use_wandb:
                loss_reduced = reduce_dict(losses_dict)
                losses_dict = {k: v.mean().item() if hasattr(v, "mean") else v for k, v in loss_reduced.items()}
                self.log_wandb(losses_dict, "valid_ema", epoch=self.epoch)
            
            improved = self.evaluation_ema(o_valid_ema, o_train)
            
            if improved:
                self.sample(is_ema=True)