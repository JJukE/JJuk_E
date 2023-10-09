from contextlib import contextmanager
from inspect import isfunction
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch import nn
from einops import rearrange


class PostInitModule(nn.Module):
    def __init_subclass__(cls, **kwargs):
        def init_decorator(previous_init):
            def new_init(self, *args, **kwargs):
                previous_init(self, *args, **kwargs)
                if type(self) == cls:
                    self.__post_init__()
            return new_init
        cls.__init__ = init_decorator(cls.__init__)
    def __post_init__(self):
        pass


class DiffusionBase(PostInitModule):
    def __init__(
            self,
            betas: np.ndarray,
            model_mean_type="eps",
            model_var_type="fixed_small",
            clip_denoised=True
    ) -> None:
        """ Base class for Diffusion Models

        Args:
            betas (np.ndarray): Noise schedule
            model_mean_type (str, optional): Model mean type for calculating output. Defaults to "eps".
            model_var_type (str, optional): Model variation type for calculating output. Defaults to "fixed_small".
            clip_denoised (bool, optional): Clamping the output or not. Defaults to True.
        """
        assert model_mean_type in "eps|x_start|x_prev".split("|")
        assert model_var_type in "fixed_small|fixed_large|learned|learned_range".split("|")
        super().__init__()

        self.num_timesteps = len(betas)
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.clip_denoised = clip_denoised

        self._param_names = set()

        # noise schedule caches (betas)
        with self.register_diffusion_parameters():
            self.betas = betas.astype(np.float64)
            self.betas_log = np.log(betas)

            self.alphas = 1. - betas
            self.alphas_cumprod = np.cumprod(self.alphas)
            self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])

            # Calculations for diffusion q(x_t | x_{t-1}) and others
            self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
            self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
            self.sqrt_recip_m1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)
            self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
            self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1. - self.alphas_cumprod)
            self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., self.alphas_cumprod))
            self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.)
            self.sqrt_one_minus_alphas_cumprod_next = np.sqrt(1 - self.alphas_cumprod_next)

            # noise schedule caches (variational lower bound calculation)
            self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
            self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
            self.posterior_variance_large = np.append(self.posterior_variance[1], betas[1:])
            self.posterior_log_variance_clipped_large = np.log(np.append(self.posterior_variance[1], betas[1:]))

    def __post_init__(self) -> None:
        for k in self._param_names:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                if v.dtype in (np.float16, np.float32, np.float64, np.float128):
                    v = v.astype(np.float32)
                v = torch.from_numpy(v)
            elif isinstance(v, Tensor):
                if v.dtype == torch.float64:
                    v = v.float()
            else:
                raise ValueError("Unknown type '{}' of param 'self.{}'.".format(type(v), k))
            
            v = torch.clone(v)
            delattr(self, k)
            self.register_buffer(k, v)
    
    @contextmanager
    def register_diffusion_parameters(self):
        prev_dict = [k for k in self.__dict__]
        yield
        post_dict = [k for k in self.__dict__]

        for k in post_dict:
            if k not in prev_dict:
                self._param_names.add(k)

    @property
    def device(self):
        return self.betas.device
    
    @staticmethod
    def _extract(coeff, t, x_shape):
        """ 
        Extract some coefficients at specified timesteps, then reshape to
        (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.

        Args:
            coeff (_type_): Coefficients.
            t (_type_): Timestep.
            x_shape (_type_): Shape of x.
        """

        assert x_shape[0] == t.shape[0]
        out = torch.gather(coeff, 0, t)
        return rearrange(out, "b -> b " + " ".join(["1"] * (len(x_shape) - 1)))
    
    @staticmethod
    def unsqueeze_as(x: Tensor, y: Tensor):
        """
        Unsqueeze x as y.

        Args:
            x (Tensor): Tensor to be unsqueezed
            y (Tensor): Shape of the target Tensor to be unsqueezed
        """
        return rearrange(x, "... -> ..." + " ".join(["()"] * (y.dim() - x.dim())))

    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape, "shape of x_t: {}, shape of eps: {}".format(x_t.shape, eps.shape)
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t
            - self._extract(self.sqrt_recip_m1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )
    
    # def _predict_xstart_from_xprev(self, x_t, t, xprev):
    #     assert x_t.shape == xprev.shape, "x_t={}, xprev={}".format(x_t.shape, xprev.shape)
    #     return ( # (xprev - coef2 * x_t) / coef1
    #         unsqueeze_as(1.0 / self.posterior_mean_coef1[t], x_t.shape) * xprev
    #         - unsqueeze_as(self.posterior_mean_coef2[t] / self.posterior_mean_coef1[t], x_t.shape) * x_t
    #     )
    
    # TODO: check if it works
    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        print("_predict_xstart_from_xprev_diffuscene")
        assert x_t.shape == xprev.shape, "shape of x_t: {}, shape of xprev: {}".format(x_t.shape, xprev.shape)
        return ( # (xprev - coef2 * x_t) / coef1
            self._extract((1.0 / self.posterior_mean_coef1).to(x_t.device), t, x_t.shape) * xprev
            - self._extract((self.posterior_mean_coef2 / self.posterior_mean_coef1).to(x_t.device), t, x_t.shape) * x_t
        )
    
    # def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
    #     return (
    #         unsqueeze_as(self.sqrt_recip_alphas_cumprod[t], x_t.shape) * x_t - pred_xstart
    #         / unsqueeze_as(self.sqrt_recip_m1_alphas_cumprod[t], x_t.shape)
    #     )
    
    
    def _predict_eps_from_xstart(self, x_t, t, x_0):
        print("_predict_eps_from_xstart_diffuscene") # TODO: check if it works
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - x_0
            / self._extract(self.sqrt_recip_m1_alphas_cumprod.to(x_t.device), t, x_t.shape)
        )
    
    
    def q_sample(self, x_start, t, noise=None):
        """ Diffuse the data (t == 0 means diffused for 1 step)     q(x_t | x_0) """
        if noise is None:
            if isfunction(lambda: torch.rand_like(x_start)):
                noise = lambda: torch.randn_like(x_start)()
            else:
                noise = lambda: torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """ Mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)"""
        assert x_start.shape == x_t.shape, "shape of x_start: {}, shape of x_t: {}".format(x_start.shape, x_t.shape)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] ==
                posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    

    def q_mean_variance(self, x_start, t):
        """ Diffusion step: q(x_t | x_{t-1}) """
        print("q_mean_variance_diffuscene") # TODO: check if it works
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance


    def p_mean_variance(self, denoise_fn: Callable, x_t: Tensor, t: Tensor, condition=None, condition_cross=None):
        model_out: Tensor = denoise_fn(x_t, t, condition, condition_cross)
        out_dict = {"model_out": model_out}

        # get variance
        if self.model_var_type in ("learned", "learned_range"):
            assert model_out.size(1) == 2 * x_t.size(1)
            model_out, tmp = torch.chunk(model_out, 2, dim=1)
            if self.model_var_type == "learned":
                model_log_var = tmp
                model_var = model_log_var.exp()
            else:
                min_log = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
                max_log = self._extract(self.betas_log, t, x_t.shape)
                frac = (tmp + 1) * 0.5
                model_log_var = frac * max_log + (1 - frac) * min_log
                model_var = model_log_var.exp()
        elif self.model_var_type == "fixed_large":
            model_log_var = self._extract(self.posterior_log_variance_clipped_large, t, x_t.shape)
            model_var = self._extract(self.posterior_variance_large, t, x_t.shape)
        elif self.model_var_type == "fixed_small":
            model_log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
            model_var = self._extract(self.posterior_variance, t, x_t.shape)
        else:
            raise NotImplementedError
        
        out_dict["model_log_var"] = model_log_var
        out_dict["model_var"] = model_var

        clip = (lambda x: x.clamp(-1.0, 1.0)) if self.clip_denoised else (lambda x: x)

        # get mean
        if self.model_mean_type == "x_prev":
            model_mean = model_out
            pred_x_start = clip(self._predict_xstart_from_xprev(x_t, t, model_mean))
        elif self.model_mean_type == "x_start":
            pred_x_start = clip(model_out)
            model_mean, _, _ = self.q_posterior_mean_variance(pred_x_start, x_t, t)
        elif self.model_mean_type == "eps":
            out_dict["pred_eps"] = model_out
            pred_x_start = clip(self._predict_xstart_from_eps(x_t, t, model_out))
            model_mean, _, _ = self.q_posterior_mean_variance(pred_x_start, x_t, t)
        else:
            raise NotImplementedError

        out_dict["model_mean"] = model_mean # \tilde{x}_{t-1}
        out_dict["pred_x_start"] = pred_x_start # \tilde{x}_0

        return out_dict # model_out, model_mean, model_var, model_log_var, pred_x_start
