# trainer_2D.py
#-*- coding:utf-8 -*-
#
# *Main part of the code is adopted from the following repository: https://github.com/lucidrains/denoising-diffusion-pytorch


import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision.utils import save_image
from inspect import isfunction
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from apex import amp
    APEX_AVAILABLE = True
    print("APEX: ON")
except:
    APEX_AVAILABLE = False
    print("APEX: OFF")


# ---------- iDWT (Haar, 1-level) for sampling/saving ----------
# must match the forward DWT filters used in dataset.py
_H = 0.5
_HAAR_K = torch.tensor([
    [[[_H, _H], [_H, _H]]],    # LL
    [[[_H, _H], [-_H, -_H]]],  # LH
    [[[ _H,-_H], [_H,-_H]]],   # HL
    [[[ _H,-_H], [-_H, _H]]]   # HH
], dtype=torch.float32)

def idwt_haar_1level(w: torch.Tensor) -> torch.Tensor:
    # w: [B,4,H/2,W/2] -> [B,1,H,W]
    K = _HAAR_K.to(w.device)              # [4,1,2,2]
    x = F.conv_transpose2d(w, K, stride=2)
    return x

# plug these in from your train config if you standardize bands
#WAVELET_MU = None  # torch.tensor([mu_LL, mu_LH, mu_HL, mu_HH], device=...)
#WAVELET_STD = None # torch.tensor([..], device=...)


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# gaussian diffusion trainer class
def extract(a: torch.Tensor, t: torch.LongTensor, x_shape):
    # a: [T], t: [B], returns [B,1,1,1] broadcastable to x
    b = t.shape[0]
    out = a.gather(0, t).float()
    return out.view(b, *((1,) * (len(x_shape) - 1)))

#def extract(a, t, x_shape):
#    b, *_ = t.shape
#    out = a.gather(-1, t)
#    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)


class GaussianDiffusion(nn.Module):
    """
    Diffusion in *wavelet-coefficient space* (standardized).
    - Optional LL-only soft window in std-space via set_ll_bounds_std(...)
    - Optional HF tanh bound (bands 1..3) each sampling step via soft_a
    - Always-on dynamic thresholding in coeff space (Imagen-style), independent of clip_denoised
    - Safe posterior (c2 clamped) to avoid x_t amplification
    - Optional LL auxiliary loss at small t to stabilize illumination (x0_hat vs x_start on LL)
    """
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels=1,
        timesteps=1000,
        loss_type='l1',
        betas=None,
        with_condition=False,
        with_pairwised=False,
        apply_bce=False,
        lambda_bce=0.0,
        band_loss_weights=None,
    ):
        super().__init__()

        # ---------- basic ----------
        self.channels       = channels
        self.image_size     = image_size
        self.denoise_fn     = denoise_fn
        self.with_condition = with_condition
        self.with_pairwised = with_pairwised
        self.apply_bce      = apply_bce
        self.lambda_bce     = lambda_bce
        self.loss_type      = loss_type

        # ---------- coefficient-space helpers ----------
        # (a) LL soft-window bounds (set later via set_ll_bounds_std)
        self.ll_bounds_std = None
        # (b) always-on dynamic threshold (Imagen-style)
        self.enable_coeff_dynamic_threshold = True
        self.coeff_dynamic_thresh_p = 0.995  # try 0.995–0.999

        # optional debug print during sampling
        self.debug_p_sample = False

        # ---------- noise schedule ----------
        if betas is not None:
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod      = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.num_timesteps = int(betas.shape[0])

        to_torch = partial(torch.tensor, dtype=torch.float32)

        # schedule buffers
        self.register_buffer('betas',                          to_torch(betas))
        self.register_buffer('alphas_cumprod',                 to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',            to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod',            to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',  to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',   to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',      to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',    to_torch(np.sqrt(1. / alphas_cumprod - 1.0)))

        # posterior buffers
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance',              to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped',  to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',            to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

        # sanity checks (safe betas, monotone acp, c2<=1)
        assert (self.betas > 0).all() and (self.betas < 1).all(), "betas must be in (0,1)"
        acp = self.alphas_cumprod
        assert torch.all(acp[1:] <= acp[:-1]), "alphas_cumprod must be non-increasing"
        with torch.no_grad():
            T = self.num_timesteps
            t_test = torch.tensor([0, T//4, T//2, 3*T//4, T-1], device=acp.device, dtype=torch.long)
            c2_test = extract(self.posterior_mean_coef2, t_test, (len(t_test), 1, 1, 1))
            assert (c2_test <= 1.0 + 1e-5).all(), f"posterior_mean_coef2>1 somewhere: {c2_test.flatten().tolist()}"

        # ---------- optional band weights ----------
        if band_loss_weights is not None:
            bw = torch.tensor(band_loss_weights, dtype=torch.float32).view(4)
            assert bw.numel() == 4, "band_loss_weights must be [LL, LH, HL, HH]"
            self.register_buffer('band_loss_weights_buf', bw)
        else:
            self.band_loss_weights_buf = None

        # ---------- LL auxiliary loss knobs ----------
        self.ll_x0_aux_weight = 0.12  # 0.05–0.10 works well
        self.ll_aux_tmax      = 300   # apply only when t <= 100

    # -------------------- helpers --------------------
    #@staticmethod
    #def _soft_clip_interval(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, softness: torch.Tensor):
    #    """Asymmetric soft clip: lo + (hi - lo) * sigmoid(x / s), in std-space."""
    #    s = torch.clamp(softness, min=1e-3)
    #    return lo + (hi - lo) * torch.sigmoid(x / s)
    
    @staticmethod
    def _soft_clip_interval(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, softness: torch.Tensor):
        """
        Identity for x in [lo,hi]. For x<lo or x>hi, smoothly push back using softplus.
        'softness' is a width parameter (bigger = gentler).
        """
        s = torch.clamp(softness, min=1e-6).to(x.dtype)
        below = F.softplus((lo - x) / s) * s      # >0 only when x < lo
        above = F.softplus((x - hi) / s) * s      # >0 only when x > hi
        return x + below - above


    def _dynamic_threshold_coeff(self, x: torch.Tensor, p: float) -> torch.Tensor:
        """
        Per-sample, per-band dynamic threshold (Imagen-style) in coeff std-space.
        s = quantile(|x|, p) over H×W; clip to ±s, then divide by s.
        Uses a floor on s to avoid blowing up tiny signals.
        """
        B, C, H, W = x.shape
        s = torch.quantile(x.abs().flatten(2), q=p, dim=2).view(B, C, 1, 1)
        s = torch.clamp(s, min=1.0)
        return x.clamp(min=-s, max=s) / s

    def _soft_bound_x0(self, x0: torch.Tensor, soft_a):
        """HF symmetric tanh bound (a * tanh(x0/a)) per band; pass `soft_a` with 4 values or tensor."""
        if soft_a is None:
            return x0
        if not torch.is_tensor(soft_a):
            a = torch.as_tensor([soft_a] * x0.shape[1], dtype=x0.dtype, device=x0.device)
        else:
            a = soft_a.to(x0.device, dtype=x0.dtype)
        a = a.view(1, -1, 1, 1)
        return torch.tanh(x0 / (a + 1e-8)) * a

    def set_ll_bounds_std(self, lo_std: float, hi_std: float, softness: float = 1.0):
        """Register LL soft-window bounds (in standardized coeff space)."""
        device = self.betas.device
        self.ll_bounds_std = (
            torch.tensor(lo_std, device=device, dtype=torch.float32),
            torch.tensor(hi_std, device=device, dtype=torch.float32),
            torch.tensor(softness, device=device, dtype=torch.float32),
        )

    # -------------------- forward/ posterior math --------------------
    def q_mean_variance(self, x_start: torch.Tensor, t: torch.LongTensor):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        var  = extract(1. - self.alphas_cumprod, t, x_start.shape)
        logv = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, var, logv

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor):
        return (
            extract(self.sqrt_recip_alphas_cumprod,   t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.LongTensor):
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape).clamp(max=0.9995)  # safe
        mean  = c1 * x_start + c2 * x_t
        var   = extract(self.posterior_variance,             t, x_t.shape)
        log_v = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_v

    def p_mean_variance(self, x: torch.Tensor, t: torch.LongTensor,
                        clip_denoised: bool, c=None, soft_a=None):
        """
        One reverse step in *standardized coeff space*.
        - Predict eps -> x0_hat
        - (optional) LL soft window in std-space
        - (optional) HF-only dynamic threshold (once)
        - (optional) HF-only soft bound via tanh with per-band soft_a
        - Compute posterior from processed x0_hat
        """

        # 1) predict eps and x0_hat
        if self.with_condition and c is not None:
            model_in = torch.cat([x, c], dim=1)
            eps = self.denoise_fn(model_in, t)
        else:
            eps = self.denoise_fn(x, t)
        x0_hat = self.predict_start_from_noise(x, t=t, noise=eps)  # standardized coeffs

        # 2) LL-only soft window in std-space (if configured)
        if self.ll_bounds_std is not None:
            lo_std, hi_std, s_std = self.ll_bounds_std
            ll = x0_hat[:, :1]
            ll = self._soft_clip_interval(ll, lo_std, hi_std, s_std)  # your "soft box" impl
            x0_hat = torch.cat([ll, x0_hat[:, 1:]], dim=1)

        # 3) HF-only dynamic thresholding (do this ONCE, before any soft/tanh)
        if getattr(self, "enable_coeff_dynamic_threshold", False):
            p = float(getattr(self, "coeff_dynamic_thresh_p", 0.995))
            hf = x0_hat[:, 1:]
            # per-sample scale from p-quantile; don't backprop through the quantile
            s = torch.quantile(hf.detach().abs().flatten(1), p, dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
            # avoid boosting if s < 1
            s = torch.maximum(s, torch.ones_like(s))
            hf = (hf / s).clamp_(-1.0, 1.0)
            x0_hat = torch.cat([x0_hat[:, :1], hf], dim=1)

        # 4) Optional HF soft bound via tanh (coeff-space); pass soft_a as [LL,LH,HL,HH]
        if soft_a is not None:
            if not torch.is_tensor(soft_a):
                soft_a = torch.as_tensor(soft_a, device=x0_hat.device, dtype=x0_hat.dtype)
            if soft_a.ndim == 1:
                soft_a = soft_a.view(1, -1, 1, 1)  # [1,4,1,1]
            a_hf = soft_a[:, 1:].clamp_min(1e-6)
            hf = x0_hat[:, 1:]
            hf = torch.tanh(hf / a_hf) * a_hf
            x0_hat = torch.cat([x0_hat[:, :1], hf], dim=1)

        # 5) (usually False in coeff space, but supported)
        if clip_denoised:
            x0_hat = x0_hat.clamp_(-1.0, 1.0)

        # 6) posterior using processed x0_hat
        model_mean, posterior_var, posterior_log_var = self.q_posterior(
            x_start=x0_hat, x_t=x, t=t
        )
        return model_mean, posterior_var, posterior_log_var


    # -------------------- sampling --------------------
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.LongTensor, condition_tensors=None,
                 clip_denoised: bool = False, repeat_noise: bool = False, soft_a=None):
        """
        One reverse step in coeff space. We bound x̂0 softly inside p_mean_variance;
        no pixel clamp here.
        """
        b, _, h, w = x.shape
        device = x.device

        # align condition
        c = None
        if self.with_condition:
            if condition_tensors is None:
                raise ValueError("with_condition=True but condition_tensors is None")
            c = condition_tensors
            if c.shape[-2:] != (h, w):
                c = F.interpolate(c, size=(h, w), mode='nearest')
            c = c.to(device, non_blocking=True)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, c=c, clip_denoised=clip_denoised, soft_a=soft_a
        )

        # helpful debug (uses clamped c2 implicitly through model_mean)
        if self.debug_p_sample:
            c1_dbg = extract(self.posterior_mean_coef1, t, x.shape)
            c2_dbg = extract(self.posterior_mean_coef2, t, x.shape).clamp(max=0.9995)
            mm_from_xt = (c2_dbg * x).abs().mean().item()
            print(f"[p_sample] t={int(t[0])} coef1={c1_dbg.mean().item():.6f} "
                  f"coef2={c2_dbg.mean().item():.6f} |x_t|={x.abs().mean().item():.3e} "
                  f"|mean|={model_mean.abs().mean().item():.3e} |c2*x_t|={mm_from_xt:.3e}")

        noise = noise_like(x.shape, device, repeat_noise)
        nonzero = (1 - (t == 0).float()).reshape(b, *((1,) * (x.ndim - 1)))
        return model_mean + nonzero * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors=None, clip_denoised: bool = False, soft_a=None):
        """Return standardized coeffs of shape (B,4,H/2,W/2)."""
        device = self.betas.device
        b, _, h, w = shape
        img = torch.randn(shape, device=device)

        c = None
        if self.with_condition:
            if condition_tensors is None:
                raise ValueError("with_condition=True but condition_tensors is None")
            c = condition_tensors
            if c.shape[-2:] != (h, w):
                c = F.interpolate(c, size=(h, w), mode='nearest')
            c = c.to(device, non_blocking=True)

        for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps, desc="Sampling"):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, condition_tensors=c, clip_denoised=clip_denoised, soft_a=soft_a)

        return img

    @torch.no_grad()
    def sample(self, batch_size=2, condition_tensors=None):
        return self.p_sample_loop(
            (batch_size, self.channels, self.image_size, self.image_size),
            condition_tensors=condition_tensors,
            clip_denoised=False
        )

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    # -------------------- training --------------------
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod,            t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,  t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, condition_tensors=None, noise=None):
        """
        x_start: standardized coeffs [B,4,H/2,W/2]
        condition_tensors: [B,1,H/2,W/2] (already downsampled by dataset)
        """
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # align cond
        if self.with_condition:
            if condition_tensors is None:
                raise ValueError("with_condition=True but condition_tensors is None")
            if condition_tensors.shape[-2:] != (h, w):
                condition_tensors = F.interpolate(condition_tensors, size=(h, w), mode='nearest')

        # forward diffuse
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict eps
        model_in = torch.cat([x_noisy, condition_tensors], dim=1) if self.with_condition else x_noisy
        eps_pred = self.denoise_fn(model_in, t)

        # base eps loss
        if self.loss_type == 'l1':
            # Huber (smooth-L1) is L1 in the tails, L2 near zero -> kills rare huge outliers
            #err = (eps_pred - noise).abs()
            err = F.smooth_l1_loss(eps_pred, noise, beta=1.0, reduction='none')
        elif self.loss_type == 'l2':
            err = (eps_pred - noise) ** 2
        else:
            raise NotImplementedError(f"Unknown loss_type: {self.loss_type}")

        if self.band_loss_weights_buf is not None:
            err = err * self.band_loss_weights_buf.view(1, 4, 1, 1)

        loss = err.mean()

        # LL x0 auxiliary (for small t)
        if self.ll_x0_aux_weight > 0:
            x0_hat = self.predict_start_from_noise(x_noisy, t=t, noise=eps_pred)
            #ll_err = (x0_hat[:, :1] - x_start[:, :1]).abs()  # L1 on LL only
            #mask   = (t <= self.ll_aux_tmax).float().view(b, 1, 1, 1)
            #ll_err_masked = (ll_err * mask).sum() / (mask.sum() + 1e-8)
            aux_w    = getattr(self, "ll_x0_aux_weight", 0.0)
            aux_tmax = getattr(self, "ll_aux_tmax", 0)
            if aux_w > 0:
                # reconstruct x0_hat from (x_t, t, ε_pred) in standardized space
                x0_hat = self.predict_start_from_noise(x_noisy, t=t, noise=eps_pred)     # [B,4,H/2,W/2]
                # per-sample LL error (L1) so we can mask by timestep
                ll_err_per_sample = (x0_hat[:, :1] - x_start[:, :1]).abs().mean(dim=(1,2,3))  # [B]
                mask_b = (t <= aux_tmax).float()                                         # [B]
                denom  = mask_b.sum()
                if denom > 0:
                    ll_err_masked = (ll_err_per_sample * mask_b).sum() / denom
                    loss = loss + aux_w * ll_err_masked

        return loss

    def forward(self, x, condition_tensors=None, *args, **kwargs):
        b, c, h, w = x.shape
        img_size = self.image_size
        assert h == img_size and w == img_size, f'Expected (H,W)=({img_size},{img_size}), got ({h},{w}).'
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, condition_tensors=condition_tensors, *args, **kwargs)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 2,
        train_lr = 2e-6,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        with_condition = False,
        with_pairwised = False):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, num_workers=4, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.train_lr = train_lr
        self.train_batch_size = train_batch_size
        self.with_condition = with_condition

        self.step = 0

        # assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'
        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.log_dir = self.create_log_dir()
        self.writer = SummaryWriter(log_dir=self.log_dir)#"./logs")
        self.reset_parameters()

    def create_log_dir(self):
        now = datetime.datetime.now().strftime("%y-%m-%dT%H%M%S")
        log_dir = os.path.join("/storage/data/li46460_wdm_ddpm/original/logs", now)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'optimizer': self.opt.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
    
    @torch.no_grad()
    def sample_and_save(self, filename=None, batch_size=1, cond=None, de_standardize=False, mu=None, std=None):
        self.ema_model.eval()
        device = next(self.ema_model.parameters()).device

        # fetch condition if needed
        if self.with_condition and cond is None:
            try:
                cond = self.ds.sample_conditions(batch_size=batch_size)
            except Exception:
                batch = next(self.dl)
                cond = batch[0] if isinstance(batch, (tuple, list)) else batch['input']
        if cond is not None:
            cond = cond.to(device).float()[:batch_size]

        Hh = self.image_size  # already set to input_size//2 when you built diffusion
        coeffs = self.ema_model.p_sample_loop(
            shape=(batch_size, 4, Hh, Hh),
            condition_tensors=cond,
            clip_denoised=False
        )

        if de_standardize and (mu is not None) and (std is not None):
            mu  = mu.view(1,4,1,1).to(device)
            std = std.view(1,4,1,1).to(device)
            coeffs = coeffs * (std + 1e-6) + mu

        imgs = idwt_haar_1level(coeffs)  # [B,1,H,W]
        imgs = (imgs.clamp(-1, 1) + 1) * 0.5

        # save grid
        out = self.results_folder / (filename or f'samples-step{self.step:06d}.png')
        save_image(imgs, str(out), nrow=min(batch_size, 4))


    def train(self):

        self.model.train()

        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()

        while self.step < self.train_num_steps:
            accumulated_loss = 0.0

            for i in range(self.gradient_accumulate_every):
                batch = next(self.dl)

                if self.with_condition:
                    # Accept either (cond, target) or {'input':..., 'target':...}
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        input_tensors, target_tensors = batch
                    elif isinstance(batch, dict):
                        input_tensors = batch['input']
                        target_tensors = batch['target']
                    else:
                        raise ValueError("Batch must be (cond, target) or dict with keys 'input' and 'target'")

                    # Move to device
                    input_tensors  = input_tensors.cuda(non_blocking=True).float()   # [B,1,H/2,W/2]
                    target_tensors = target_tensors.cuda(non_blocking=True).float()  # [B,4,H/2,W/2]

                    # Safety: ensure H/W match (should already, given the new dataset)
                    if input_tensors.shape[-2:] != target_tensors.shape[-2:]:
                        input_tensors = F.interpolate(input_tensors, size=target_tensors.shape[-2:], mode='nearest')

                    # Diffusion wrapper returns the batch-mean loss; no extra sum/divide needed
                    loss = self.model(target_tensors, condition_tensors=input_tensors)

                else:
                    data = batch.cuda(non_blocking=True).float()
                    loss = self.model(data)

                # Ensure scalar (mean over batch); remove the old `.sum() / self.batch_size`
                if loss.ndim > 0:
                    loss = loss.mean()

                print(f'{self.step}.{i}: {loss.item():.6f}')
                # Correct per-step scaling for gradient accumulation
                backwards(loss / self.gradient_accumulate_every, self.opt)
                accumulated_loss += loss.item()

            # Record here
            #average_loss = np.mean(accumulated_loss)
            average_loss = accumulated_loss / float(self.gradient_accumulate_every)

            end_time = time.time()
            #self.writer.add_scalar("training_loss", average_loss, self.step)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                # sample in coeff space (no clamp), iDWT, save
                #self.sample_and_save(filename=f'sample-{milestone}.png', batch_size=1)
                de_std = getattr(self.ds, 'standardize', False)
                mu = getattr(self.ds, 'mu', None)
                sigma = getattr(self.ds, 'sigma', None)
                self.sample_and_save(
                    filename=f'sample-{milestone}.png',
                    batch_size=1,
                    de_standardize=de_std,
                    mu=mu,
                    std=sigma
                )
                self.save(milestone)
            # logging last
            self.writer.add_scalar("training_loss", average_loss, self.step)

            self.step += 1

        print('training completed')
        end_time = time.time()
        execution_time = (end_time - start_time)/3600
        self.writer.add_hparams(
            {
                "lr": self.train_lr,
                "batchsize": self.train_batch_size,
                "image_size":self.image_size,
                "execution_time (hour)":execution_time
            },
            {"last_loss":average_loss}
        )
        self.writer.close()
