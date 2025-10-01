import math
import shutil
from pathlib import Path
from packaging import version
from abc import ABCMeta, abstractmethod
from typing import Sequence

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import Dataset

import transformers
import diffusers
import huggingface_hub
from accelerate import Accelerator
from accelerate.utils import (set_seed, ProjectConfiguration, DataLoaderConfiguration,
                              DeepSpeedPlugin, DistributedDataParallelKwargs)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from jjuke.util import load_yaml, instantiate_from_config, logger, ProgressBar, model_summary

diffusers.utils.check_min_version("0.26.0.dev0") # Will error if the minimal version of diffusers is not installed. Remove at your own risks.

class BasePreprocessor(metaclass=ABCMeta):
    def __init__(self, device) -> None:
        self.device = device

    def to(self, *xs):
        ys = []
        for x in xs:
            y = self._to(x, self.device)
            ys.append(y)

        if len(ys) == 1:
            return ys[0]
        else:
            return ys

    def _to(self, x):
        if isinstance(x, torch.Tensor):
            x = x.to(self.device, non_blocking=True)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device, non_blocking=True)
        elif isinstance(x, (list, tuple, dict)):
            x = self.batch_to_device(x)
        return x

    def batch_to_device(self, batch):
        if isinstance(batch, list):
            return [self._to(x) for x in batch]
        elif isinstance(batch, tuple):
            return (self._to(x) for x in batch)
        elif isinstance(batch, dict):
            return {k: self._to(batch[k]) for k in batch}
        else:
            return self._to(batch)

    @abstractmethod
    def __call__(self, batch, augmentation=False):
        # b = EasyDict(log={})
        # b.motion = self._to(batch["motion"]) # (B, N, D)
        # b.motion = normalizer.normalize(b.motion) # NOTE
        # b.music = self._to(batch["music"]) # (B, N, d)
        # b.genre = batch["genre"]
        # b.wav = batch["wav"]
        # if augmentation:
        #     pass
        # return b
        pass


class BaseTrainer(metaclass=ABCMeta):
    def __init__(
        self,
        args,
        num_saves: int = None,
        save_period: int = 50000,
        valid_period: int = 50000,
        mixed_precision: str = "no",
        clip_grad: float = 0.,
        grad_acc_steps: int = 1,
        # input_data_name: str = "input", # for printing model info
        **kwargs
    ) -> None:
        self.args = args
        assert hasattr(self.args, "train_steps") ^ hasattr(self.args, "train_epochs"), "Either `train_steps` (for step training) or `train_epochs` (for epoch training) must be given."
        
        flag = False
        if hasattr(self.args, "resume_ckpt"):
            resume_path = Path(self.args.resume_ckpt).parent
            resume_args = load_yaml(resume_path / "args.yaml")
            new_path = Path(self.args.exp_path)
            try:
                new_args = load_yaml(new_path / "args.yaml")
            except FileNotFoundError as e:
                new_path.parts[-1]
                print(e)
                flag = True
                prev_path = new_path
                new_path = new_path.parent / "_".join([new_path.stem.split("_")[0], str(int(new_path.stem.split("_")[1]) - 1), *new_path.stem.split("_")[2:]])
                print("Find another new path: ", new_path)
                new_args = load_yaml(new_path / "args.yaml")
            new_timestr = "_".join(new_path.stem.split("_")[:2])
            
            # move config, arg, log files into the resume_dir # TODO: should the file name be overwritten?
            new_args.exp_dir = resume_args.exp_dir
            new_args.exp_path = resume_args.exp_path
            
            with (resume_path / f"args_{new_timestr}.yaml").open("w") as f:
                OmegaConf.save(new_args, f)
            
            del self.args.log
            logger.basic_config(resume_path / f"train_{new_timestr}.log")
            self.args.log = logger.get_logger()
            
            self.args.exp_dir = str(Path(self.args.resume_ckpt).parent.parent)
            self.args.exp_path = str(Path(self.args.resume_ckpt).parent)
        
        self.num_saves = num_saves # save only latest `num_saves` checkpoints
        self.save_period = save_period # save every `save_period` steps
        self.valid_period = valid_period # validation every `valid_period` steps
        if mixed_precision == False:
            mixed_precision = "no"
        elif mixed_precision == "fp16": # TODO: debug
            print(f"learning rate is fixed as 0.0 when using fp16. Need to debug.")
        assert mixed_precision in ["no", "fp16", "bf16"] # , "fp8"]
        self.mixed_precision = mixed_precision
        self.clip_grad = clip_grad
        self.grad_acc_steps = grad_acc_steps
        
        self.build_accelerator()
        
        self.accel.wait_for_everyone()
        if hasattr(self.args, "resume_ckpt") and self.accel.is_main_process:
            shutil.rmtree(str(new_path), ignore_errors=True)
            if flag:
                shutil.rmtree(str(prev_path), ignore_errors=True)
        
        if self.accel.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        
        # set the training seed
        if args.seed is not None:
            if self.accel.is_main_process:
                self.log.info(f"Setting seed to {args.seed}")
            set_seed(args.seed)
        
        # repository creation
        if self.accel.is_main_process:
            if args.logging.push_to_hub:
                # NOTE: use either push_to_hub or wandb
                repo_id = huggingface_hub.create_repo(
                    repo_id=args.logging.push_to_hub.hub_model_id or Path(args.exp_path).name,
                    exist_ok=True,
                    token=args.logging.push_to_hub.hub_token
                ).repo_id
        
        # build network
        self.build_network(**kwargs)
        self.build_dataset()
        self.build_preprocessor()
        self.prepare_accelerator()
        # if hasattr(self.args, "debug") and self.accel.is_main_process:
        #     data = next(iter(self.dl_train))
        #     t = torch.randint(0, 1000, (data["motion"].shape[0],), dtype=torch.int64, device=self.device)
        #     self.log.info(f"**************************** Model summarization *********************************")
        #     self.log.info(model_summary(self.model, input_data=(data["motion"].to(self.device), t, data["music"].to(self.device), 0.25, data["genre"].to(self.device))))
        #     self.log.info(f"**********************************************************************************")
        self.accel.wait_for_everyone()
    
    @property
    def trainer_type(self):
        return "EpochTrainer" if hasattr(self.args, "train_epochs") else "StepTrainer"
    
    @property
    def dtype(self):
        """
        NOTE: mixed precision is automatically applied with `self.accel.backward -> self.accel.clip_grad_norm_ -> self.optim.step`
        In other words, this `self.dtype` is not needed for mixed precision setting.
        If a model is set with `self.model.to(..., dtype=self.dtype)` explicitly, `Attempting to unscale FP16 gradients.` occurs.
        Refer to https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372/16
        """
        return self.weight_dtype
    
    @property
    def device(self):
        return self.accel.device

    @property
    def ddp(self):
        return self.accel.use_distributed

    @property
    def log(self) -> logger.CustomLogger:
        return self.args.log
    
    @property
    def model_params(self):
        model_size = 0
        for param in self.model.parameters():
            if param.requires_grad:
                model_size += param.data.nelement()
        return model_size
            
    #============================================================
    # Initialization
    #============================================================
    def build_accelerator(self):
        dl_cfg = None
        if hasattr(self.args.accel, "dl_cfg"):
            dl_cfg: DataLoaderConfiguration = instantiate_from_config(self.args.accel.dl_cfg)
        
        deepspeed_plugin = None
        if hasattr(self.args.accel, "deepspeed"):
            deepspeed_plugin: DeepSpeedPlugin = instantiate_from_config(self.args.accel.deepspeed)
        
        kwargs = []
        if hasattr(self.args.accel, "ddp_kwargs"):
            ddp_kwargs: DistributedDataParallelKwargs = instantiate_from_config(self.args.accel.ddp_kwargs)
            kwargs.append(ddp_kwargs)
        
        proj_cfg = ProjectConfiguration(project_dir=self.args.exp_path)
        
        self.accel = Accelerator(
            gradient_accumulation_steps=self.grad_acc_steps,
            mixed_precision=self.mixed_precision,
            dataloader_config=dl_cfg,
            deepspeed_plugin=deepspeed_plugin,
            log_with="wandb" if (self.args.logging.use_wandb and not hasattr(self.args, "debug")) else None,
            project_config=proj_cfg,
            kwargs_handlers=kwargs if len(kwargs) > 0 else None
        )
        self.log.info("Accelerator State\n" + str(self.accel.state))
    
    @abstractmethod
    def build_network(self, **kwargs):
        # self.model: DanceDecoder = instantiate_from_config(self.args.diffusion.denoiser).to(self.device)
        # self.scheduler: SDE = instantiate_from_config(self.args.diffusion.scheduler)
        # self.score_model = ScoreModel(self.model, pred="x0_pred", sde=self.scheduler, guidance_weight=self.guidance_weight)
        
        # self.smpl_model: Union[SMPLSkeleton, SMPLX_Skeleton] = instantiate_from_config(self.args.smpl_model, self.device)
        # self.log.info("Model Params: %.2fM" % (self.model_params / 1e6))
        
        # self.optim = instantiate_from_config(self.args.optim, self.model.parameters())
        
        # self.config_network(**kwargs)
        pass
    
    def config_network(self, **kwargs):
        self.use_ema = kwargs.get("use_ema", False)
        enable_mem_eff_att = kwargs.get("enable_mem_eff_att", False)
        gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
        allow_tf32 = kwargs.get("allow_tf32", False)
        scale_lr = kwargs.get("scale_lr", False)
        
        # Create EMA for the model
        if self.use_ema:
            self.ema_model: EMAModel = instantiate_from_config(self.args.trainer.ema, self.model.parameters())
        
        if enable_mem_eff_att:
            if diffusers.utils.import_utils.is_xformers_available():
                import xformers
                
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.log.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly.")
        
        # self.accelerate_save_setting(self.model)
        
        if gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
        
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        
        if scale_lr:
            self.args.optim.params.lr = (
                self.args.optim.params.lr * self.grad_acc_steps * self.args.dataset.batch_size * self.accelerator.num_processes
            )
    
    # @abstractmethod
    # def accelerate_save_setting(self, model_cls): # TODO: this might be a function for customizing save
    #     # custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    #     def save_model_hook(models, weights, output_dir):
    #         if self.accel.is_main_process:
    #             i = len(weights) - 1

    #             while len(weights) > 0:
    #                 weights.pop()
    #                 model = models[i]

    #                 sub_dir = "controlnet"
    #                 model.save_pretrained(Path(output_dir) / sub_dir)

    #                 i -= 1

    #     def load_model_hook(models, input_dir):
    #         while len(models) > 0:
    #             # pop models so that they are not loaded again
    #             model = models.pop()

    #             # load diffusers style into model
    #             load_model = model_cls.from_pretrained(input_dir, subfolder="controlnet") # ex. model_cls -> ControlNetModel
    #             model.register_to_config(**load_model.config)

    #             model.load_state_dict(load_model.state_dict())
    #             del load_model

    #     self.accel.register_save_state_pre_hook(save_model_hook)
    #     self.accel.register_load_state_pre_hook(load_model_hook)
    
    def build_dataset(self):
        dataloaders: Sequence[Dataset] = instantiate_from_config(self.args.dataset)
        if len(dataloaders) == 3:
            self.dl_train, self.dl_valid, self.dl_test = dataloaders
            l1, l2, l3 = len(self.dl_train.dataset), len(self.dl_valid.dataset), len(self.dl_test.dataset)
            if self.accel.is_main_process:
                self.log.info(f"Load {l1} train, {l2} valid, {l3} test items")
        elif len(dataloaders) == 2:
            self.dl_train, self.dl_valid = dataloaders
            l1, l2 = len(self.dl_train.dataset), len(self.dl_valid.dataset)
            if self.accel.is_main_process:
                self.log.info(f"Load {l1} train, {l2} valid items")
        else:
            raise NotImplementedError
    
    def build_preprocessor(self):
        self.preprocessor: BasePreprocessor = instantiate_from_config(self.args.preprocessor, device=self.device)
    
    @abstractmethod
    def prepare_objects(self):
        # # not prepare dl_valid
        # self.model, self.ddpm_trainer, self.ddpm_sampler, self.optim, self.dl_train, self.sched = self.accel.prepare(
        #     self.model, self.ddpm_trainer, self.ddpm_sampler, self.optim, self.dl_train, self.sched
        # )
        # # prepare dl_valid as well
        # self.model, self.ddpm_trainer, self.ddpm_sampler, self.optim, self.dl_train, self.dl_valid, self.sched = self.accel.prepare(
        #     self.model, self.ddpm_trainer, self.ddpm_sampler, self.optim, self.dl_train, self.dl_valid, self.sched
        # )
        pass
    
    def prepare_accelerator(self):
        # scheduler and training steps
        if self.trainer_type == "EpochTrainer":
            self.total_epochs = self.args.train_epochs
            num_sharded_dl_train = math.ceil(len(self.dl_train) / self.accel.num_processes)
            self.steps_per_epoch = math.ceil(num_sharded_dl_train / self.grad_acc_steps)
        else:
            self.total_steps = self.args.train_steps
        
        if hasattr(self.args, "sched"):
            if "get_scheduler" in self.args.sched.target:
                self.warmup_steps_for_sched = self.args.sched.params.num_warmup_steps * self.accel.num_processes
                if self.trainer_type == "EpochTrainer": # self.sched.step() must be after each epoch
                    self.train_steps_for_sched = self.total_epochs * self.accel.num_processes
                else: # StepTrainer # self.sched.step() must be after each step (self.optim.step())
                    self.warmup_steps_for_sched = self.args.sched.params.num_warmup_steps * self.accel.num_processes
                    self.train_steps_for_sched = self.total_steps * self.accel.num_processes
                self.sched = get_scheduler(
                    name=self.args.sched.params.name,
                    optimizer=self.optim,
                    num_warmup_steps=self.warmup_steps_for_sched,
                    num_training_steps=self.train_steps_for_sched,
                    num_cycles=self.args.sched.params.num_cycles if hasattr(self.args.sched.params, "num_cycles") else 1,
                    power=self.args.sched.params.power if hasattr(self.args.sched.params, "power") else 1.,
                )
            else:
                self.sched = instantiate_from_config(self.args.sched, self.optim)
        
        self.prepare_objects()
        
        if self.use_ema:
            if self.args.trainer.ema.offload:
                self.ema_model.pin_memory()
            else:
                self.ema_model.to(self.device)

        # mixed precision
        self.weight_dtype = torch.float32
        if self.accel.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accel.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        
        # recalculate training steps and epochs as the size of the training dataloader may have changed
        self.steps_per_epoch = math.ceil(len(self.dl_train) / self.grad_acc_steps)
        if self.trainer_type == "EpochTrainer":
            self.total_steps = self.total_epochs * self.steps_per_epoch
        else:
            self.total_epochs = math.ceil(self.total_steps / self.steps_per_epoch)
        
        # trackers for storing the configuration
        if self.args.logging.use_wandb and self.accel.is_main_process:
            self.accel.init_trackers(project_name=self.args.logging.project_name, config=self.args,
                                     init_kwargs={"wandb": {
                                         "name": "_".join(Path(self.args.exp_path).stem.split("_")[2:]),
                                         "dir": Path(self.args.exp_path),
                                         "id": Path(self.args.exp_path).stem,
                                         "resume": "allow",
                                        }})

    #============================================================
    # Training utilities
    #============================================================
    def unwrap_model(self):
        model = self.accel.unwrap_model(self.model)
        if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
            # if model is compiled with torch.compile()
            model = model._orig_mod
        else:
            model = model
        return model
    
    def save(self):
        self.accel.wait_for_everyone()
        if self.accel.is_main_process:
            if self.num_saves is not None:
                ckpts = [d.name for d in Path(self.args.exp_path).iterdir() if d.name.startswith("checkpoint")]
                ckpts = sorted(ckpts, key=lambda x: int(str(x).split("-")[1]))
                
                if len(ckpts) >= self.num_saves:
                    num_remove = len(ckpts) - self.num_saves + 1
                    removing_ckpts = ckpts[0:num_remove]
                    self.log.info(f"{len(ckpts)} checkpoints already exist, removing {len(removing_ckpts)} checkpoints")
                    self.log.info(f"removing checkpoints: {', '.join(removing_ckpts)}")
                    
                    for removing_ckpt in removing_ckpts:
                        removing_ckpt = Path(self.args.exp_path) / removing_ckpt
                        shutil.rmtree(removing_ckpt)
            
            save_path = Path(self.args.exp_path) / \
                f"checkpoint-{self.global_epoch if self.trainer_type == 'EpochTrainer' else self.global_step}"
            
            self.accel.save_state(str(save_path), safe_serialization=False)
            if self.use_ema:
                torch.save(self.ema_model.state_dict(), str(save_path / "ema_state_dict.pth"))
            self.log.info(f"Saved state to {str(save_path)}")
    
    def get_total_loss(self, losses_dict, verbose=False):
        """ Note that all losses should be meant (and weighted) """
        total_loss = 0.
        for loss_name, loss in losses_dict.items():
            # max_value = torch.finfo(loss.dtype).max
            # clamped_loss = torch.clamp(loss, min=0, max=max_value)
            # if loss > max_value and self.accel.is_main_process:
            #     self.log.info(f"Clamp {loss_name}: {loss.item()} -> {max_value}")
            if verbose:
                self.log.info(f"{loss_name}:\t{loss:.4f}")
            total_loss = total_loss + loss # clamped_loss # NOTE: all losses should be meant
        return total_loss
    
    def gather_loss(self, batch_size, losses_dict):
        """ Gather the losses across all processes for logging (during validation). """
        out_dict = {}
        for loss_name, loss in losses_dict.items():
            out_dict[loss_name] = self.accel.gather(loss.repeat(batch_size)).mean().detach().cpu()
        out_dict["valid_loss"] = self.get_total_loss(out_dict).item()
        return out_dict

    def clip_gradient(self, model):
        # TODO: check if it works when bf16
        if self.clip_grad > 0.:
            self.accel.clip_grad_norm_(model.parameters(), self.clip_grad)
    
    def ema_step(self):
        if self.args.trainer.ema.offload:
            self.ema_model.to(device="cuda", non_blocking=True)
        self.ema_model.step(self.model.parameters())
        if self.args.trainer.ema.offload:
            self.ema_model.to(device="cpu", non_blocking=True)
            torch.cuda.empty_cache()  # TODO: check if it works (free unused memory)
    
    def log_loss(self, data_dict, phase, images=None, caption=None):
        """
        Log data during training or validation.
        
        Args:
            data_dict (Dict): Dictionary of losses to log
            phase (str): `train` or `valid`
            images (np.ndarray): vstacked numpy array, e.g., top will be model output and bottom will be Ground Truth.
            caption (str): caption for the images
        
        Example:
        ```python
        import numpy as np
        import torch
        from jjuke.util.vis import get_wandb_image
        
        losses_dict = {"example_loss": torch.nn.L1Loss(pred, gt)}
        losses_dict.update({"images": get_wandb_image(pred_images, gt_images})
        
        self.log_loss(losses_dict, "train", images=images_fig, caption="Top: Model output / Bottom: Ground Truth")
        ```
        """
        assert phase == "train" or phase == "valid", "argument `phase` should be either `train` or `valid`."
        assert (images is not None and caption is not None) or (images is None and caption is None)
        dict_ = dict()
        for k, v in data_dict.items():
            dict_[phase + "/" + k] = v
        if phase == "train":
            self.accel.log(dict_, step=self.global_epoch if self.trainer_type == "EpochTrainer" else self.global_step)
        else:
            if self.trainer_type == "EpochTrainer":
                self.accel.log(dict_, step=self.global_epoch)
                
                msg = f"Epoch[{self.global_epoch:04}/{self.total_epochs:04}]"
                for k, v in data_dict.items():
                    if isinstance(v, (torch.Tensor, int, float)):
                        msg += f" {k}[" + ";".join([f"{v.item() if isinstance(v, torch.Tensor) else v:.4f}"]) + "]"
                
                print(flush=True)
                self.log.info(msg)
                self.log.flush()
            else: # StepTrainer
                self.accel.log(dict_, step=self.global_step)
                
                msg = f"Step[{self.global_step:08}/{self.total_steps:08}]"
                for k, v in data_dict.items():
                    if isinstance(v, (torch.Tensor, int, float)):
                        msg += f" {k}[" + ";".join([f"{v.item() if isinstance(v, torch.Tensor) else v:.4f}"]) + "]"
                
                print(flush=True)
                self.log.info(msg)
                self.log.flush()
            
            if images is not None and caption is not None:
                for tracker in self.accel.trackers:
                    if tracker.name == "wandb":
                        wandb.Image(images, caption=caption)
    
    #============================================================
    # Training
    #============================================================
    def prepare_train(self):
        total_batch_size = self.args.dataset.params.batch_size * self.accel.num_processes * self.grad_acc_steps
        if hasattr(self.args, "debug"):
            if self.accel.is_main_process:
                self.log.info("*********************** Running training (Debugging Mode) ***********************")
                self.log.info(f"*  Num examples: {len(self.dl_train)}")
                if self.trainer_type == "EpochTrainer":
                    self.log.info(f"*  Num total epochs: {self.total_epochs}")
                else: # StepTrainer
                    self.log.info(f"*  Num total steps: {self.total_steps}")
                self.log.info(f"*  Instantaneous batch size per device: {self.args.dataset.params.batch_size}")
                if self.grad_acc_steps > 1:
                    self.log.info(f"*  Total train batch size (w/ parallel, distributed & accumulation): {total_batch_size}")
                    self.log.info(f"*  Gradient Accumulation steps = {self.grad_acc_steps}")
                else:
                    self.log.info(f"*  Total train batch size (w/ parallel, distributed): {total_batch_size}")
                self.log.info("*********************************************************************************")
                self.log.flush()
        else:
            if self.accel.is_main_process:
                self.log.info("**************************** Running training ****************************")
                self.log.info(f"*  Num examples: {len(self.dl_train)}")
                if self.trainer_type == "EpochTrainer":
                    self.log.info(f"*  Num total epochs: {self.total_epochs}")
                else: # StepTrainer
                    self.log.info(f"*  Num total steps: {self.total_steps}")
                self.log.info(f"*  Instantaneous batch size per device: {self.args.dataset.params.batch_size}")
                if self.grad_acc_steps > 1:
                    self.log.info(f"*  Total train batch size (w/ parallel, distributed & accumulation): {total_batch_size}")
                    self.log.info(f"*  Gradient Accumulation steps = {self.grad_acc_steps}")
                else:
                    self.log.info(f"*  Total train batch size (w/ parallel, distributed): {total_batch_size}")
                self.log.info("**************************************************************************")
                self.log.flush()
        global_step = 0
        global_epoch = 0
        
        # potentially load in the weights and states from a previous save
        if hasattr(self.args, "resume_ckpt"):
            ckpt_name = Path(self.args.resume_ckpt).name
            if self.accel.is_main_process:
                self.log.info(f"Resuming from checkpoint {ckpt_name}")
            self.accel.load_state(self.args.resume_ckpt)
            if self.use_ema:
                ema_state_dict = torch.load(Path(self.args.resume_ckpt) / "ema_state_dict.pth", map_location="cpu")
                self.ema_model.load_state_dict(ema_state_dict)
                self.ema_model.to(self.device)
            if self.trainer_type == "EpochTrainer":
                global_epoch = int(ckpt_name.split("-")[1])
                global_step = global_epoch * self.steps_per_epoch
            else:
                global_step = int(ckpt_name.split("-")[1])
                global_epoch = global_step // self.steps_per_epoch
        
        if hasattr(self.args, "debug"):
            if self.trainer_type == "EpochTrainer":
                self.save_period, self.valid_period = 2, 2
                self.total_epochs = global_epoch + 4 if hasattr(self.args, "resume_ckpt") else 4
            else:
                self.save_period, self.valid_period = 5, 5
                self.total_steps = global_step + 10 if hasattr(self.args, "resume_ckpt") else 10
        
        self.pbar = ProgressBar(self.accel.is_main_process, trainer_type=self.trainer_type)
        return global_step, global_epoch

    def train_epoch(self):
        # self.model.train()
        # avg_train_loss = 0.
        # avg_recon_loss, avg_vel_loss, avg_fk_loss, avg_foot_loss = 0., 0., 0., 0.
        # for step, batch in enumerate(self.dl_train):
        #     batch = self.preprocessor(batch, self.normalizer, augmentation=True)
        #     motion, music, genre, wav = batch.motion, batch.music, batch.genre, batch.wav
        #     B = motion.shape[0]
            
        #     kwargs = {
        #         "smpl_model": self.smpl_model,
        #         "normalizer": self.normalizer if isinstance(self.smpl_model, SMPLX_Skeleton) else None,
        #         "p2_loss_weight": None,
        #         "recon_loss_weight": self.recon_loss_weight,
        #         "vel_loss_weight": self.vel_loss_weight,
        #         "fk_loss_weight": self.fk_loss_weight,
        #         "foot_loss_weight": self.foot_loss_weight,
        #     }
        #     losses_dict = self.ddpm_trainer(self.model, motion, c=music, **kwargs)
        #     train_loss = self.get_total_loss(losses_dict)
            
        #     # backpropagation
        #     self.optim.zero_grad()
        #     self.accel.backward(train_loss)
        #     if self.clip_grad > 0.:
        #         self.accel.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
        #     self.optim.step()
            
        #     avg_recon_loss += losses_dict["recon_loss"].detach().cpu().numpy() / B
        #     avg_vel_loss += losses_dict["vel_loss"].detach().cpu().numpy() / B
        #     avg_fk_loss += losses_dict["fk_loss"].detach().cpu().numpy() / B
        #     avg_foot_loss += losses_dict["foot_loss"].detach().cpu().numpy() / B
        #     avg_train_loss += (self.get_total_loss(losses_dict)).detach().cpu().numpy() / B
            
        #     if self.use_ema:
        #         if self.args.trainer.ema.offload:
        #             self.ema_model.to(device="cuda", non_blocking=True)
        #         self.ema_model.step(self.model.parameters())
        #         if self.args.trainer.ema.offload:
        #             self.ema_model.to(device="cpu", non_blocking=True)
        #     self.global_step += 1
        
        # self.sched.step() # NOTE: Discriminate if this scehduler is either step-based, epoch-based, or validation metric-based
        # self.pbar.update(1)
        # self.global_epoch += 1
        # losses_dict.update({
        #     "train_loss": avg_train_loss,
        #     "lr": self.sched.get_last_lr()[0],
        # })
        # self.log_loss(losses_dict, "train")
        # self.pbar.set_postfix(**{"train_loss": avg_train_loss})
        pass
    
    def train_step(self):
        # self.model.train()
        # batch = self.preprocessor(batch, self.normalizer, augmentation=True)
        # motion, music, genre, wav = batch.motion, batch.music, batch.genre, batch.wav
        # B = motion.shape[0]
        
        # kwargs = {
        #     "smpl_model": self.smpl_model,
        #     "normalizer": self.normalizer if isinstance(self.smpl_model, SMPLX_Skeleton) else None,
        #     "p2_loss_weight": None,
        #     "recon_loss_weight": self.recon_loss_weight,
        #     "vel_loss_weight": self.vel_loss_weight,
        #     "fk_loss_weight": self.fk_loss_weight,
        #     "foot_loss_weight": self.foot_loss_weight,
        # }
        # losses_dict = self.ddpm_trainer(self.model, motion, c=music, **kwargs)
        # train_loss = self.get_total_loss(losses_dict)
        
        # # backpropagation
        # self.optim.zero_grad()
        # self.accel.backward(train_loss)
        # if self.clip_grad > 0.:
        #     self.accel.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        
        # self.optim.step()
        # self.sched.step() # NOTE: Discriminate if this scehduler is either step-based, epoch-based, or validation metric-based
        
        # if self.use_ema:
        #     if self.args.trainer.ema.offload:
        #         self.ema_model.to(device="cuda", non_blocking=True)
        #     self.ema_model.step(self.model.parameters())
        #     if self.args.trainer.ema.offload:
        #         self.ema_model.to(device="cpu", non_blocking=True)
        # self.pbar.update(1)
        # self.global_step += 1
        # losses_dict.update({
        #     "train_loss": train_loss,
        #     "lr": self.sched.get_last_lr()[0],
        # })
        # self.log_loss(losses_dict, "train")
        # self.pbar.set_postfix(**{"train_loss": train_loss})
        pass

    @torch.no_grad()
    def valid_step(self):
        # # TODO: self.dl_valid should be also prepared for validation with multi gpus in some cases.
        # self.model.eval()
        # pbar = tqdm(range(0, len(self.dl_valid)), ncols=128, desc="Valid", disable=not self.accel.is_local_main_process)
        # if self.use_ema:
        #     # store the model params temporarily and load the EMA params to perform inference
        #     self.ema_model.store(self.model.parameters())
        #     self.ema_model.copy_to(self.model.parameters())
        
        # avg_valid_loss = 0.
        # avg_recon_loss, avg_vel_loss, avg_fk_loss, avg_foot_loss = 0., 0., 0., 0.
        # for step, batch in enumerate(self.dl_valid):
        #     batch = self.preprocessor(batch, self.normalizer, augmentation=False)
        #     motion, music, genre, wav = batch.motion, batch.music, batch.genre, batch.wav
        #     B = motion.shape[0]
            
        #     # DDPM
        #     kwargs = {
        #         "smpl_model": self.smpl_model,
        #         "normalizer": self.normalizer if isinstance(self.smpl_model, SMPLX_Skeleton) else None,
        #         "p2_loss_weight": None,
        #         "recon_loss_weight": self.recon_loss_weight,
        #         "vel_loss_weight": self.vel_loss_weight,
        #         "fk_loss_weight": self.fk_loss_weight,
        #         "foot_loss_weight": self.foot_loss_weight,
        #     }
        #     samples, losses_dict = self.ddpm_sampler(self.model, motion.shape, target=motion, c=music, **kwargs)
            
        #     # gather loss
        #     losses_dict = self.gather_loss(B, losses_dict) # NOTE: gather is necessary for validation for multiple gpus
        #     avg_recon_loss += losses_dict["recon_loss"].detach().cpu().numpy()
        #     avg_vel_loss += losses_dict["vel_loss"].detach().cpu().numpy()
        #     avg_fk_loss += losses_dict["fk_loss"].detach().cpu().numpy()
        #     avg_foot_loss += losses_dict["foot_loss"].detach().cpu().numpy()
        #     avg_valid_loss += (self.get_total_loss(losses_dict)).detach().cpu().numpy()
            
        #     pbar.update(1)
        #     pbar.set_postfix(**{"valid_loss": avg_valid_loss})
        
        # if self.accel.is_main_process:
        #     # log loss
        #     val_losses_dict = {
        #         "valid_loss": avg_valid_loss.item(),
        #         "recon_loss": avg_recon_loss.item(),
        #         "vel_loss": avg_vel_loss.item(),
        #         "fk_loss": avg_fk_loss.item(),
        #         "foot_loss": avg_foot_loss.item(),
        #     }
        #     self.log_loss(val_losses_dict, "valid")
        pass
    
    def save_pipeline(self):
        pass
    
    def final_process(self):
        pass

    def fit(self):
        self.global_step, self.global_epoch = self.prepare_train()
        self.accel.wait_for_everyone()
        if self.trainer_type == "EpochTrainer":
            for epoch in range(self.global_epoch, self.total_epochs):
                self.pbar.start(total_epochs=self.total_epochs, current_epoch=self.global_epoch+1,
                                steps_per_epoch=self.steps_per_epoch, msg="Train")
                with self.accel.autocast():
                    self.train_epoch()
                self.global_epoch += 1
                self.pbar.stop()
                
                if self.global_epoch % self.valid_period == 0:
                    with self.accel.autocast():
                        self.valid_step()
                if self.global_epoch % self.save_period == 0:
                    self.save()
                self.accel.wait_for_everyone()
        else: # StepTrainer
            self.pbar.start(total_steps=self.total_steps, current_step=self.global_step, msg="Train")
            for epoch in range(self.global_epoch, self.total_epochs):
                for step, batch in enumerate(self.dl_train):
                    with self.accel.autocast():
                        self.train_step(batch)

                    if self.global_step % self.valid_period == 0:
                        with self.accel.autocast():
                            self.valid_step()
                    if self.global_step % self.save_period == 0:
                        self.save()
                    
                    self.accel.wait_for_everyone()
                    if self.global_step >= self.total_steps:
                        break
            self.pbar.stop()
        
        # last process
        # self.valid_step()
        # self.save()
        self.accel.wait_for_everyone()
        if self.accel.is_main_process:
            model = self.unwrap_model()
            if self.use_ema:
                self.ema_model.copy_to(model.parameters())
            
            self.save_pipeline()
            self.final_process()
        
        self.accel.end_training()
