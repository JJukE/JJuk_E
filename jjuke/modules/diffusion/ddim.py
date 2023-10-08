from typing import Callable, Union, Sequence

import numpy as np
import torch
from torch import Tensor, Size

from jjuke.modules import default
from jjuke.modules.diffusion import get_betas
from jjuke.modules.diffusion.diffusion_base import DiffusionBase

class DDIMSampler(DiffusionBase):
    def __init__(
            self,
            betas: np.ndarray,
            ddim_num_timesteps: int = 1000,
            discretize_method: str = "uniform",
            eta: float = 0.,
            model_mean_type: str = "eps",
            model_var_type: str = "fixed_small",
            clip_denoised: bool = True
    ):
        """ Diffusion DDIM Sampler

        Args:
            betas (np.ndarray): Noise schedule
            ddim_num_timesteps (int, optional): Number of time steps for sampling.
                                                Defaults to 1000 (Equals to DDPM's).
            Hyperparameters for DDIM Sampler
                discretize_method, eta
            Hyperparameters for Diffusion
                model_mean_type, model_var_type, clip_denoised
        """
        super().__init__(betas, model_mean_type, model_var_type, clip_denoised)
        
        self.disc_method = discretize_method
        self.eta = eta # stochasticity (0: DDIM, 1: DDPM)
        self.num_ddim_timesteps = ddim_num_timesteps # number of time steps of subsequence (<= self.num_timesteps)

        # Noise schedule for DDIM
        if self.disc_method == "uniform":
            c = self.num_timesteps // self.num_ddim_timesteps
            ddim_timesteps = np.asarray(list(range(0, self.num_timesteps, c)))
        elif self.disc_method == "quad":
            ddim_timesteps = ((np.linspace(0, np.sqrt(self.num_timesteps * .8), self.num_ddim_timesteps)) ** 2).astype(int)
        else:
            raise NotImplementedError("There is no ddim discretization method called '{}'.".format(self.disc_method))
        
        # assert ddim_timesteps.shape[0] == self.num_ddim_timesteps, "Invalid number of ddim timesteps. It should be divisor of the number of ddpm timesteps."
        # Add one to get the final alpha values right (ones from first scale to data during sampling)
        ddim_timesteps += 1
        if self.num_timesteps in ddim_timesteps:
            # If there is an index over number of ddpm time steps, delete it
            # TODO: Doesn't it make any problem in practice?
            print("{} causes an indexing error because it is same to the number of DDPM time steps,".format(ddim_timesteps[-1])
                  + " so deleted and the result number of time steps of DDIM is {}".format(len(ddim_timesteps)))
            ddim_timesteps = np.delete(ddim_timesteps, -1)
        self.ddim_timesteps = ddim_timesteps

        # DDIM parameters
        with self.register_diffusion_parameters():
            # parameters that should be calculated in ddim timestep
            self.ddim_alphas_cumprod = self.alphas_cumprod[self.ddim_timesteps]
            self.ddim_alphas_cumprod_prev = np.asarray(
                [self.ddim_alphas_cumprod[0]] + self.alphas_cumprod[self.ddim_timesteps[:-1]].tolist()
            )
            self.ddim_sqrt_alphas_cumprod_prev = np.sqrt(self.ddim_alphas_cumprod_prev)
            self.ddim_sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.ddim_alphas_cumprod)
            self.ddim_sigmas = self.eta * np.sqrt(
                (1. - self.ddim_alphas_cumprod_prev) / (1. - self.ddim_alphas_cumprod)
                * (1. - self.ddim_alphas_cumprod / self.ddim_alphas_cumprod_prev)
            )
            self.ddim_dir = np.sqrt(1. - self.ddim_alphas_cumprod_prev - self.ddim_sigmas ** 2)
        
    
    def ddim_sample(self, denoise_fn: Callable[[Tensor, Tensor], Tensor],
                    x_t: Tensor, t: Tensor, s: Tensor, condition=None, condition_cross=None): # TODO: maybe s is also tensor?
        """ Sample x_{t-1} from the model using DDIM. (same usage as p_sample() in DDPM) """
        out = self.p_mean_variance(denoise_fn=denoise_fn, x_t=x_t, t=t, condition=condition, condition_cross=condition_cross)
        eps = default(out.get("pred_eps", None), lambda: self._predict_eps_from_xstart(x_t, t, out["pred_x_start"]))

        ddim_sqrt_alphas_cumprod_prev = self._extract(self.ddim_sqrt_alphas_cumprod_prev.to(x_t.device), s, x_t.shape)
        ddim_sigmas = self._extract(self.ddim_sigmas.to(x_t.device), s, x_t.shape)
        ddim_dir = self._extract(self.ddim_dir.to(x_t.device), s, x_t.shape)

        mean_pred = out["pred_x_start"] * ddim_sqrt_alphas_cumprod_prev + ddim_dir * eps
        nonzero_mask = self.unsqueeze_as((t!=0), x_t)
        noise = torch.randn_like(x_t)
        sample = mean_pred + nonzero_mask * ddim_sigmas * noise
        return sample
    
    
    def ddim_reverse_sample(self, denoise_fn: Callable[[Tensor, Tensor], Tensor],
                            x_t: Tensor, t: Tensor, condition=None, condition_cross=None):
        """ Sample x_{t+1} from the model using DDIM reverse ODE. """
        if self.eta != 0.0:
            print("Warn! Reverse ODE only for deterministic path, but 'eta={}'".format(self.eta))
        
        sqrt_recip_alphas_cumprod = self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape)
        sqrt_recip_m1_alphas_cumprod = self._extract(self.sqrt_recip_m1_alphas_cumprod.to(x_t.device), t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_next = self._extract(self.sqrt_one_minus_alphas_cumprod_next.to(x_t.device), t, x_t.shape)
        sqrt_alphas_cumprod_prev = self._extract(self.sqrt_alphas_cumprod_prev.to(x_t.device), t, x_t.shape)

        out = self.p_mean_variance(denoise_fn=denoise_fn, x_t=x_t, t=t, condition=condition, condition_cross=condition_cross)
        if self.model_mean_type == "x_prev":
            out["pred_x_start"] = self._predict_xstart_from_xprev(x_t, t, out["model_mean"])
        eps = (sqrt_recip_alphas_cumprod * x_t - out["pred_x_start"]) / sqrt_recip_m1_alphas_cumprod
        sample = out["pred_x_start"] * sqrt_alphas_cumprod_prev + sqrt_one_minus_alphas_cumprod_next * eps
        # return dict(sample=sample, eps=eps, **out)
        return sample
    

    def sample_progressive(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Union[Sequence, Size],
                           noise: Tensor = None, condition = None, condition_cross=None) -> Tensor:
        """
        Generate samples

        Returns:
            Tensor: (num_samples, *shape)
        """
        batch_size = shape[0]
        sample = default(noise, lambda: torch.randn(shape, device=self.device))
        indices = np.flip(self.ddim_timesteps) # [..., 3c+1, 2c+1, c+1, 1]

        for i, step in enumerate(indices):
            s = torch.full((batch_size, ), len(indices) - i - 1, dtype=torch.long, device=self.device)
            t = torch.full((batch_size, ), step, dtype=torch.long, device=self.device)
            sample = self.ddim_sample(denoise_fn, sample, t, s, condition, condition_cross)
            yield sample
    

    def forward(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Union[Sequence, Size],
                noise: Tensor = None, condition=None, condition_cross=None, num_samples=1) -> Tensor:
        """
        Generate samples, returning intermediate samples when num_samples > 1

        Returns:
            Tensor: (num_samples, *shape)
        """
        # variables for sampling intermedians like in DDPM
        sample_indices = np.linspace(0, len(self.ddim_timesteps), num_samples, dtype=np.int64).tolist()
        sample_list = []

        if num_samples > 1:
            for i, sample in enumerate(self.sample_progressive(
                denoise_fn, shape, noise, condition, condition_cross)):
                if i in sample_indices:
                    sample_list.append(sample)
            sample_list.append(sample)
            return torch.stack(sample_list)
        else:
            return next(self.sample_progressive(denoise_fn, shape, noise, condition, condition_cross))


def __test__():
    betas = get_betas("linear", 10)
    diffusion_sampler = DDIMSampler(betas, ddim_num_timesteps=5, ddim_eta=0.0, model_var_type="learned_range")
    model = lambda x, t: torch.cat([x, x], dim=1)  # any model that doubles the channel (only when learnt model variance)
    shape = (2, 3, 4, 5, 6, 7, 8, 9)  # arbitrary shape
    out = diffusion_sampler(model, shape, num_samples=5)
    print(out.shape)
    # torch.Size([5, 2, 3, 4, 5, 6, 7, 8, 9])
    out = diffusion_sampler(model, shape, num_samples=1)
    print(out.shape)
    # torch.Size([2, 3, 4, 5, 6, 7, 8, 9])


if __name__ == "__main__":
    __test__()