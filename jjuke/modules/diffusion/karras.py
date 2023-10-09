from typing import Callable

import numpy as np
import torch
from torch import Tensor, Size
from scipy.interpolate import interp1d
from einops import rearrange

from jjuke.modules import default
from jjuke.modules.diffusion import get_betas
from jjuke.modules.diffusion.diffusion_base import DiffusionBase


class KarrasSampler(DiffusionBase):
    def __init__(
            self,
            betas: np.ndarray,
            karras_num_timesteps: int,
            karras_sampler: str = "heun", # ["heun|dpm|ancestral"]
            sigma_min: float = 0.002,
            sigma_max: int = 80, # higher for high resolutions?
            rho: float = 7.,
            s_churn: float = 0.,
            s_tmin:float = 0.,
            s_tmax: float = float("inf"),
            s_noise:float = 1.,
            model_mean_type: str = "eps",
            model_var_type: str = "fixed_small",
            clip_denoised: bool = True
    ):
        """ Diffusion Karras Sampler

        Args:
            betas (np.ndarray): Noise schedule
            karras_num_timesteps (int): Number of time steps for sampling
            karras_sampler (str, optional): Sampling function. Defaults to "heun".
            Hyperparameters for Karras Sampler
                sigma_max, s_churn, s_tmin, s_tmax, s_noise
            Hyperparameters for Diffusion
                model_mean_type, model_var_type, clip_denoised
        """
        super().__init__(betas, model_mean_type, model_var_type, clip_denoised)
        assert karras_sampler in "heun|dpm|ancestral".split("|"), "{} is not in 'heun, dpm, ancestral'.".format(karras_sampler)

        self.karras_num_timesteps = karras_num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.karras_sampler = karras_sampler
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        
        # Noise schedule for Karras sampler
        ramp = np.linspace(0, 1, self.karras_num_timesteps)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

        # Karras sampler parameters
        with self.register_diffusion_parameters():
            self.sigmas = np.concatenate([sigmas, np.zeros((1,), dtype=sigmas.dtype)]) # append zero
            self.gammas = np.zeros_like(self.sigmas[:-1])
            self.gammas[s_tmin <= self.sigmas[:-1]] = min(self.s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1)

            sigmas_alphas_cumprod = 1. / (self.sigmas[:-1] ** 2 + 1)
            alphas_cumprod_to_t = interp1d(self.alphas_cumprod, np.arange(0, self.num_timesteps))
            c1 = sigmas_alphas_cumprod > self.alphas_cumprod[0]
            c2 = sigmas_alphas_cumprod <= self.alphas_cumprod[-1]
            c3 = ~(c1 | c2)
            self.karras_t_sigmas = np.zeros_like(sigmas_alphas_cumprod)
            self.karras_t_sigmas[c2] = self.num_timesteps - 1
            self.karras_t_sigmas[c3] = alphas_cumprod_to_t(sigmas_alphas_cumprod[c3])

            self.sigmas_hat = self.sigmas[:-1] * (self.gammas + 1)
            sigmas_hat_alphas_cumprod = 1. / (self.sigmas_hat ** 2 + 1)
            c1 = sigmas_hat_alphas_cumprod > self.alphas_cumprod[0]
            c2 = sigmas_hat_alphas_cumprod <= self.alphas_cumprod[-1]
            c3 = ~(c1 | c2)
            self.karras_t_sigmas_hat = np.zeros_like(sigmas_hat_alphas_cumprod)
            self.karras_t_sigmas_hat[c2] = self.num_timesteps - 1
            self.karras_t_sigmas_hat[c3] = alphas_cumprod_to_t(sigmas_hat_alphas_cumprod[c3])

            self.sigmas_mid = ((self.sigmas_hat ** (1/3) + self.sigmas[1:] ** (1/3)) / 2) ** 3
            sigmas_mid_alphas_cumprod = 1. / (self.sigmas_mid ** 2 + 1)
            c1 = sigmas_mid_alphas_cumprod > self.alphas_cumprod[0]
            c2 = sigmas_mid_alphas_cumprod <= self.alphas_cumprod[-1]
            c3 = ~(c1 | c2)
            self.karras_t_sigmas_mid = np.zeros_like(sigmas_mid_alphas_cumprod)
            self.karras_t_sigmas_mid[c2] = self.num_timesteps - 1
            self.karras_t_sigmas_mid[c3] = alphas_cumprod_to_t(sigmas_mid_alphas_cumprod[c3])

    @staticmethod
    def to_d(x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        dims_to_append = x.ndim - sigma.ndim
        if dims_to_append < 0:
            raise ValueError("Input has {} dims, but target dimension is {}, which is less".format(sigma.ndim, x.ndim))
        appended = rearrange(sigma, "... -> ... " + " ".join(["1"] * dims_to_append)) # append dimensions to the end of the sigma as x dimensions
        return (x - denoised) / appended
    
    @staticmethod
    def get_ancestral_step(sigma_from, sigma_to):
        """Calculates the noise level (sigma_down) to step down to and the amount
        of noise to add (sigma_up) when doing an ancestral sampling step."""
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
        return sigma_down, sigma_up
    
    def sample_heun(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Size):
        """
        Implements Algorithm 2 (Heun steps) from Karras et al. (2022).

        ### yield:
        - x: current denoised sample (x_t)
        - x_0_pred: prediction of x_0 of current step
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device) * self.sigma_max
        indices = range(len(self.sigmas) - 1)

        for i in indices:
            gamma = self.gammas[i]
            sigma_hat = self.sigmas_hat[i]
            eps = torch.randn_like(x) * self.s_noise
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - self.sigmas[i] ** 2) ** 0.5
            
            t = torch.full((batch_size,), self.karras_t_sigmas_hat[i], device=self.device, dtype=torch.long)
            c_in = self.unsqueeze_as(1. / (sigma_hat ** 2 + 1), x)
            denoised = self.p_mean_variance(denoise_fn, x * c_in, t)["pred_x_start"]
            d = self.to_d(x, sigma_hat, denoised)
            yield x, denoised

            dt = self.sigmas[i+1] - sigma_hat
            if self.sigmas[i+1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                t_2 = torch.full((batch_size,), self.karras_t_sigmas[i+1], device=self.device, dtype=torch.long)
                c_in_2 = self.unsqueeze_as(1.0 / (self.sigmas[i+1] ** 2 + 1) ** 0.5, x_2)
                denoised_2 = self.p_mean_variance(denoise_fn, x_2 * c_in_2, t_2)["pred_x_start"]
                d_2 = self.to_d(x_2, self.sigmas[i+1], denoised_2)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        yield x, denoised
    
    def sample_dpm(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Size):
        """
        A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022).

        ### yield:
        - x: current denoised sample (x_t)
        - x_0_pred: prediction of x_0 of current step
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)
        indices = range(len(self.sigmas) - 1)

        for i in indices:
            gamma = self.gammas[i]
            eps = torch.randn_like(x) * self.s_noise
            sigma_hat = self.sigmas_hat[i]
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - self.sigmas[i] ** 2) ** 0.5
            
            t = torch.full((batch_size,), self.karras_t_sigmas_hat[i], device=self.device, dtype=torch.long)
            c_in = self.unsqueeze_as(1. / (sigma_hat ** 2 + 1) ** 0.5, x)
            denoised = self.p_mean_variance(denoise_fn, x * c_in, t)["pred_x_start"]
            d = self.to_d(x, sigma_hat, denoised)
            yield x, denoised # "x": x, "pred_x_start": denoised

            # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
            sigma_mid = self.sigmas_mid[i]
            dt_1 = sigma_mid - sigma_hat
            dt_2 = self.sigmas[i+1] - sigma_hat

            x_2 = x + d * dt_1
            t_2 = torch.full((batch_size,), self.karras_t_sigmas_mid[i], device=self.device, dtype=torch.long)
            c_in_2 = self.unsqueeze_as(1. / (sigma_mid ** 2 + 1) ** 0.5, x)
            denoised_2 = self.p_mean_variance(denoise_fn, x_2 * c_in_2, t_2)["pred_x_start"]
            d_2 = self.to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        yield x, denoised # "x": x, "pred_x_start": denoised
    
    def sample_euler_ancestral(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Size):
        """
        Ancestral sampling with Euler method steps.

        ### yield:
        - x: current denoised sample (x_t)
        - x_0_pred: prediction of x_0 of current step
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)
        indices = range(len(self.sigmas) - 1)

        for i in indices:
            t = torch.full((batch_size,), self.karras_t_sigmas[i], device=self.device, dtype=torch.long)
            c_in = self.unsqueeze_as(1. / (self.sigmas[i] ** 2 + 1) ** 0.5, x)
            denoised = self.p_mean_variance(denoise_fn, x * c_in, t)["pred_x_start"]
            yield x, denoised # "x": x, ..., "pred_x_start": denoised

            # Euler method
            sigma_down, sigma_up = self.get_ancestral_step(self.sigmas[i], self.sigmas[i+1])
            d = self.to_d(x, self.sigmas[i], denoised)
            dt = sigma_down - self.sigmas[i]
            x = x + d * dt
            x = x + torch.randn_like(x) * sigma_up
        yield x, x # "x": x, "pred_x_start": x
    
    def forward(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Size, num_samples=1):
        sampler_fn = {
            "heun": self.sample_heun,
            "dpm": self.sample_dpm, 
            "ancestral": self.sample_euler_ancestral
        }[self.karras_sampler]

        # variables for sampling intermediations too
        sample_indices = np.linspace(0, self.karras_num_timesteps, num_samples, dtype=np.int64).tolist()
        sample_list = []

        if num_samples > 1:
            for i, (x_t, pred_x_start) in enumerate(sampler_fn(denoise_fn, shape)):
                if i in sample_indices:
                    sample_list.append(x_t)
            sample_list.append(pred_x_start)
            return torch.stack(sample_list)
        else:
            return next(sampler_fn(denoise_fn, shape))[1] # pred_x_start


def __test__():
    betas = get_betas("linear", 1000)
    model = lambda x, t: torch.cat([x, x], dim=1)  # any model that doubles the channel (only when learnt model variance)
    shape = (2, 3, 4, 5, 6, 7, 8, 9)  # arbitrary shape

    diffusion_sampler = KarrasSampler(betas, karras_sampler="heun", model_var_type="learned_range", karras_num_timesteps=200)
    out = diffusion_sampler(model, shape, num_samples=5)
    print(out.shape)
    # torch.Size([6, 2, 3, 4, 5, 6, 7, 8, 9])
    out = diffusion_sampler(model, shape, num_samples=1)
    print(out.shape)
    # torch.Size([2, 3, 4, 5, 6, 7, 8, 9])

    diffusion_sampler = KarrasSampler(betas, karras_sampler="dpm", model_var_type="learned_range", karras_num_timesteps=200)
    out = diffusion_sampler(model, shape, num_samples=5)
    print(out.shape)
    # torch.Size([6, 2, 3, 4, 5, 6, 7, 8, 9])

    diffusion_sampler = KarrasSampler(betas, karras_sampler="ancestral", model_var_type="learned_range", karras_num_timesteps=200)
    out = diffusion_sampler(model, shape, num_samples=5)
    print(out.shape)
    # torch.Size([6, 2, 3, 4, 5, 6, 7, 8, 9])


if __name__ == "__main__":
    __test__()