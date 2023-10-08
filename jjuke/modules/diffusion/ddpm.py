from functools import partial
from typing import Callable, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, Size
from einops import rearrange

from jjuke.modules import default
from jjuke.modules.diffusion import normal_kl, discretized_gaussian_log_likelihood
from jjuke.modules.diffusion.diffusion_base import DiffusionBase


class DDPMTrainer(DiffusionBase):
    def __init__(
            self,
            betas: np.ndarray,
            model_mean_type="eps",
            model_var_type="fixed_small",
            loss_type="l2",
            p2_loss_weight_gamma=0.0,
            p2_loss_weight_k=1.0,
            clip_denoised=True
    ):
        """ Diffusion DDPM Trainer

        Args:
            betas (np.ndarray): Noise schedule
            Hyperparameters for Diffusion
            model_mean_type (str, optional): Model mean type for calculating output. Defaults to "eps".
            model_var_type (str, optional): Model variation type for calculating output. Defaults to "fixed_small".
            clip_denoised (bool, optional): Clamping the output or not. Defaults to True.
            Hyperparameters for loss function
                loss_type, p2_loss_weight_gamma, p2_loss_weight_k
        """
        super().__init__(betas, model_mean_type, model_var_type, clip_denoised)

        assert loss_type in "l2|rescaled_l2|l1|rescaled_l1|kl|rescaled_kl".split("|")
        self.loss_type = loss_type
        self.do_p2 = p2_loss_weight_gamma > 0.0

        with self.register_diffusion_parameters():
            self.p2_loss_weight = (p2_loss_weight_k + self.alphas_cumprod / (1 - self.alphas_cumprod)) ** -p2_loss_weight_gamma
    

    def loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = {
            "l2": partial(F.mse_loss, reduction="none"),
            "rescaled_ls": partial(F.mse_loss, reduction="none"),
            "l1": partial(F.l1_loss, reduction="none"),
            "rescaled_l1": partial(F.l1_loss, reduction="none")
        }[self.loss_type](pred, target)
        loss = rearrange(loss, "dim_0 ... -> dim_0 (...)").mean(dim=1) # flatten -> mean
        return loss
    

    def get_vlb(self, denoise_fn, x_start, x_t, t):
        # variational lower bound
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_start, x_t, t)
        out = self.p_mean_variance(denoise_fn, x_t, t)
        model_mean, model_log_var = out["model_mean"], out["model_log_var"]
        kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var)
        kl = rearrange(kl, "dim_0 ... -> dim_0 (...)").mean(dim=1)

        decoder_nll = -discretized_gaussian_log_likelihood(x_start, model_mean, 0.5 * model_log_var)
        decoder_nll = rearrange(decoder_nll, "dim_0 ... -> dim_0 (...)").mean(dim=1)

        out["vlb"] = torch.where(t == 0, decoder_nll, kl)
        return out
    
    
    def forward(self, denoise_fn: Callable[[Tensor, Tensor], Tensor],
                x_start: Tensor, t: Tensor = None, noise: Tensor = None, return_info=False):
        batch_size = x_start.size(0)
        t = default(t, lambda: torch.randint(0, self.num_timesteps, (batch_size, ), dtype=torch.long, device=self.device))
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start, t, noise)

        losses = {}
        info = {"t": t, "eps": noise, "x_t": x_t, "x_start": x_start}
        
        if self.loss_type in ("kl", "rescaled_kl"):
            # \mathcal{L}_\text{kl}
            info.update(self.get_vlb(denoise_fn=denoise_fn, x_start=x_start, x_t=x_t, t=t))
            losses["loss"] = info["vlb"]
        
        elif self.loss_type in ("l2", "rescaled_l2", "l1", "rescaled_l1"):
            # \mathcal{L}_\text{vlb}
            model_out = denoise_fn(x_t, t)
            if self.model_var_type in ["learned", "learned_range"]:
                assert model_out.size(1) == 2 * x_t.size(1), "Output channel must be double of input channel for {}".format(self.model_var_type)
                model_out, tmp = torch.chunk(model_out, chunks=2, dim=1)
                frozen_out = torch.cat([model_out.detach(), tmp], dim=1)
                info.update(self.get_vlb(lambda *_: frozen_out, x_start, x_t, t)) # return the values of the variables regardless of its input
                losses["vlb"] = info["vlb"]
                if self.loss_type in ("rescaled_l1", "rescaled_l2"):
                    losses["vlb"] = losses["vlb"] * self.num_timesteps / 1000.
            # NOTE: vlb is not calculated when its fixed variance setting, but the fixed variances are used during sampling

            # \mathcal{L}_\text{simple}
            x_prev = self.q_posterior_mean_variance(x_start, x_t, t)[0]
            info["x_prev"] = x_prev
            
            target = {"eps": noise, "x_start": x_start, "x_prev": x_prev}[self.model_mean_type]
            info["target"] = target
            losses["recon"] = self.loss_fn(model_out, target)

            # \mathcal{L} = \mathcal{L}_\text{simple} * \lambda_\text{p2}
            if self.do_p2:
                losses["loss"] = losses["recon"] * self.p2_loss_weight[t]
            else:
                losses["loss"] = losses["recon"]
            
            # \mathcal{L}_\text{hybrid}
            if "vlb" in losses:
                losses["loss"] = losses["loss"] + losses["vlb"]
            
        return (losses, info) if return_info else losses


class DDPMSampler(DiffusionBase):
    def __init__(
            self,
            betas: np.ndarray,
            model_mean_type="eps",
            model_var_type="fixed_small",
            clip_denoised=True
    ):
        super().__init__(betas, model_mean_type, model_var_type, clip_denoised)
    

    def p_sample(self, denoise_fn: Callable[[Tensor, Tensor], Tensor],
                 x_t: Tensor, t: Tensor, condition=None, condition_cross=None):
        """ Sample x_{t-1} from p_theta(x_{t-1} | x_t) """
        out = self.p_mean_variance(denoise_fn=denoise_fn, x_t=x_t, t=t, condition=condition, condition_cross=condition_cross)
        noise = torch.randn_like(x_t)
        
        nonzero_mask = self.unsqueeze_as((t!=0), x_t)
        sample = out["model_mean"] + nonzero_mask * torch.exp(0.5 * out["model_log_var"]) * noise
        return sample
    

    def sample_progressive(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Union[Sequence, Size],
                           noise: Tensor = None, condition=None, condition_cross=None) -> Tensor:
        """
        Generate samples

        Returns:
            Tensor: (num_samples, *shape)
        """
        batch_size = shape[0]
        sample = default(noise, lambda: torch.randn(shape, device=self.device))
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices:
            t = torch.full((batch_size, ), i, dtype=torch.long, device=self.device)
            sample = self.p_sample(denoise_fn, sample, t, condition, condition_cross)
            yield sample
    

    def forward(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Union[Sequence, Size],
                noise: Tensor = None, condition=None, condition_cross=None, num_samples=1) -> Tensor:
        """
        Generate samples, returning intermediate samples when num_samples > 1

        Returns:
            Tensor: (num_samples, *shape) or (*shape, )
        """
        sample = default(noise, lambda: torch.randn(shape, device=self.device))
        
        # variables for filtering intermediations
        sample_indices = np.linspace(0, self.num_timesteps, num_samples, dtype=np.int64).tolist()
        sample_list = []

        if num_samples > 1:
            for i, sample in enumerate(self.sample_progressive(
                denoise_fn, shape, sample, condition, condition_cross)):
                if i in sample_indices:
                    sample_list.append(sample)
            sample_list.append(sample)
            return torch.stack(sample_list)
        else:
            return next(self.sample_progressive(denoise_fn, shape, sample, condition, condition_cross))
        