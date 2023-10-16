import os
import urllib
from glob import glob

import torch
import torch.distributed as dist
from torch import nn
from torch.utils import model_zoo

from jjuke.network_utils.distributed import is_dist_avail_and_initialized


class CheckpointIO(object):
    '''
    load, save, resume network weights.
    '''
    def __init__(self, cfg, is_master, **kwargs):
        '''
        initialize model and optimizer.
        '''
        self.cfg = cfg
        self.is_master = is_master
        self.log_path = cfg.config.log.log_dir
        self._module_dict = kwargs
        self._module_dict.update({'epoch': 0, 'min_loss': 1e8})
        self._saved_filename = 'model_last'

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def saved_filename(self):
        return self._saved_filename

    @staticmethod
    def is_url(url):
        scheme = urllib.parse.urlparse(url).scheme
        return scheme in ('http', 'https')

    def get(self, key):
        return self._module_dict.get(key, None)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self._module_dict.update(kwargs)

    def save(self, suffix=None, **kwargs):
        '''
        save the current module dictionary.
        :param kwargs:
        :return:
        '''
        if not self.is_master:
            return

        outdict = kwargs
        for k, v in self._module_dict.items():
            if k in ['net', 'optimizer', 'scheduler']:
                outdict[k] = dict()
                for k1, v1 in v.items():
                    outdict[k][k1] = v1.state_dict()
            else:
                outdict[k] = v

        if not suffix:
            filename = self.saved_filename
        else:
            filename = self.saved_filename.replace('last', suffix)

        torch.save(outdict, os.path.join(self.log_path, filename + '.pth'))

    def load(self, filename, device='cuda', *domain):
        '''
        load a module dictionary from local file or url.
        :param filename (str): name of saved module dictionary
        :return:
        '''
        if self.cfg.config.distributed.num_gpus > 1:
            # Use a barrier() to make sure that process 1 loads the model after process
            # 0 saves it.
            dist.barrier()
        if self.is_url(filename):
            return self.load_url(filename, device, *domain)
        else:
            return self.load_file(filename, device, *domain)

    def parse_checkpoint(self, device='cuda'):
        '''
        check if resume or finetune from existing checkpoint.
        :return:
        '''
        if self.cfg.config.resume:
            # resume everything including net weights, optimizer, last epoch, last loss.
            self.cfg.info('Begin to resume from the last checkpoint.')
            self.resume(device)
        elif self.cfg.config.finetune:
            # only load net weights.
            self.cfg.info('Begin to finetune from the existing weight.')
            self.finetune(device)
        else:
            self.cfg.info('Begin to train from scratch.')

    def finetune(self, device='cuda'):
        '''
        finetune fron existing checkpoint
        :return:
        '''
        if isinstance(self.cfg.config.weight, str):
            weight_paths = [self.cfg.config.weight]
        else:
            weight_paths = self.cfg.config.weight

        for weight_path in weight_paths:
            weight_path = os.path.join(self.cfg.config.root_dir, weight_path)
            if not os.path.exists(weight_path):
                self.cfg.info('Warning: finetune failed: the weight path %s is invalid. Begin to train from scratch.' % (weight_path))
            else:
                self.load(weight_path, device, 'net')
                self.cfg.info('Weights for finetuning loaded.')

    def resume(self, device='cuda'):
        '''
        resume the lastest checkpoint
        :return:
        '''
        saved_file_names = glob(os.path.join(os.path.dirname(os.path.dirname(self.log_path)), '*-*-*/*-*-*/%s*.pth' % (self.saved_filename)))
        saved_file_names.sort(reverse=True)

        for last_checkpoint in saved_file_names:
            if not os.path.exists(last_checkpoint):
                continue
            else:
                self.load(last_checkpoint, device)
                self.cfg.info('Last checkpoint resumed.')
                return

        self.cfg.info('Warning: resume failed: No checkpoint available. Begin to train from scratch.')

    def load_file(self, filename, device='cuda', *domain):
        '''
        load a module dictionary from file.
        :param filename: name of saved module dictionary
        :return:
        '''

        if os.path.exists(filename):
            self.cfg.info('Loading checkpoint from %s.' % (filename))
            checkpoint = torch.load(filename, map_location=device)
            scalars = self.parse_state_dict(checkpoint, *domain)
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url, device='cuda', *domain):
        '''
        load a module dictionary from url.
        :param url: url to a saved model
        :return:
        '''
        self.cfg.info('Loading checkpoint from %s.' % (url))
        state_dict = model_zoo.load_url(url, progress=True, map_location=device)
        scalars = self.parse_state_dict(state_dict, domain)
        return scalars

    def parse_state_dict(self, checkpoint, *domain):
        '''
        parse state_dict of model and return scalars
        :param checkpoint: state_dict of model
        :return:
        '''
        for key, value in self._module_dict.items():

            # only load specific key names.
            if domain and (key not in domain):
                continue

            if key in checkpoint:
                if key in ['net', 'optimizer', 'scheduler']:
                    for subkey, subvalue in value.items():
                        if key in ['net']:
                            subvalue.module.load_weight(checkpoint[key][subkey])
                        else:
                            subvalue.load_state_dict(checkpoint[key][subkey])
                else:
                    self._module_dict.update({key: checkpoint[key]})
            else:
                self.cfg.info('Warning: Could not find %s in checkpoint!' % key)

        if not domain:
            # remaining weights in state_dict that not found in the models.
            scalars = {k:v for k,v in checkpoint.items() if k not in self._module_dict}
            if scalars:
                self.cfg.info('Warning: the remaining modules %s in checkpoint are not found in the current setting.' % (scalars.keys()))
        else:
            scalars = {}

        return scalars


class AverageMeter(object):
    '''
    Computes ans stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val # current value
        if not isinstance(val, list):
            self.sum += val * n # accumulated sum, n = batch_size
            self.count += n # accumulated count
        else:
            self.sum += sum(val)
            self.count += len(val)
        self.avg = self.sum / self.count # current average value

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]


class LossRecorder(object):
    def __init__(self, batch_size=1):
        '''
        Log loss data
        :param config: configuration file.
        :param phase: train, validation or eval.
        '''
        self._batch_size = batch_size
        self._loss_recorder = {}

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def loss_recorder(self):
        return self._loss_recorder

    def update_loss(self, loss_dict):
        for key, item in loss_dict.items():
            if key not in self._loss_recorder:
                self._loss_recorder[key] = AverageMeter()
            self._loss_recorder[key].update(item, self._batch_size)

    def synchronize_between_processes(self):
        for meter in self._loss_recorder.values():
            meter.synchronize_between_processes()