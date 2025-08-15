
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

# ---------------- Haar kernels + DWT/IDWT (batched) ----------------

_H = 0.5
_HAAR_K = torch.tensor([
    [[[_H, _H], [_H, _H]]],    # LL
    [[[_H, _H], [-_H, -_H]]],  # LH
    [[[ _H,-_H], [_H,-_H]]],   # HL
    [[[ _H,-_H], [-_H, _H]]]   # HH
], dtype=torch.float32)

def dwt_haar_1level_batched(x: torch.Tensor) -> torch.Tensor:
    """x: [B,1,H,W] -> [B,4,H/2,W/2] (H,W even)."""
    K = _HAAR_K.to(x.device, x.dtype)
    return F.conv2d(x, K, stride=2)

def idwt_haar_1level(w: torch.Tensor) -> torch.Tensor:
    """w: [B,4,H/2,W/2] -> [B,1,H,W]."""
    K = _HAAR_K.to(w.device, w.dtype)
    return F.conv_transpose2d(w, K, stride=2)

# --- utilities (place near top of diffusion_model/trainer.py) ---
class EMA:
    def __init__(self, beta: float):
        self.beta = float(beta)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1.0 - self.beta) * new

    @torch.no_grad()
    def update_model_average(self, ma_model, current_model):
        for ma_param, cur_param in zip(ma_model.parameters(), current_model.parameters()):
            ma_param.data = self.update_average(ma_param.data, cur_param.data)

def cycle(dl):
    """endless dataloader iterator"""
    while True:
        for batch in dl:
            yield batch


# ---------------- schedules / utils ----------------

def extract(a: torch.Tensor, t: torch.LongTensor, x_shape):
    b = t.shape[0]
    out = a.gather(0, t).float()
    return out.view(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    make = (lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,)*(len(shape)-1))))
    return make() if repeat else torch.randn(shape, device=device)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    acp = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1 - (acp[1:] / acp[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

# ---------------- Diffusion (WDM-style) ----------------

class GaussianDiffusion(nn.Module):
    """
    Diffusion in wavelet-coefficient space (2D, C=4 bands).
    - Dataset supplies wavelet coeffs with LL already ÷3
    - Optional ε-prediction (default) or x₀-prediction (toggle `predict_x0`)
    - Per-step pixel-space clamp (clip_denoised=True) for stability
    """

    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels=4,
        timesteps=1000,
        loss_type='l1',
        betas=None,
        with_condition=False,
        predict_x0=False,        # <--- NEW: toggle
        # (unused but kept for API compat)
        with_pairwised=False,
        apply_bce=False,
        lambda_bce=0.0,
        band_weights=None
    ):
        super().__init__()
        self.channels       = channels
        self.image_size     = image_size
        self.denoise_fn     = denoise_fn
        self.with_condition = with_condition
        self.loss_type      = loss_type
        self.predict_x0     = bool(predict_x0)
        self.debug_p_sample = False
        if band_weights is not None:
            bw = torch.as_tensor(band_weights, dtype=torch.float32)
            if bw.numel() != 4:
                raise ValueError("band_weights must have 4 values for [LL,LH,HL,HH].")
            self.register_buffer('band_weights', bw / bw.sum())
        else:
            self.band_weights = None

        # schedule
        betas = (betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas)
        betas = betas if betas is not None else cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod      = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.num_timesteps = int(betas.shape[0])
        to_t = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('_dummy', torch.zeros(1))  # for device access
        self.register_buffer('betas',                         to_t(betas))
        self.register_buffer('alphas_cumprod',                to_t(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',           to_t(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod',           to_t(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_t(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',  to_t(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',     to_t(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',   to_t(np.sqrt(1. / alphas_cumprod - 1.0)))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance',             to_t(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_t(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',           to_t(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',           to_t((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

        # sanity checks
        assert (self.betas > 0).all() and (self.betas < 1).all()
        acp = self.alphas_cumprod
        assert torch.all(acp[1:] <= acp[:-1])
        with torch.no_grad():
            T = self.num_timesteps
            t_test = torch.tensor([0, T//4, T//2, 3*T//4, T-1], device=acp.device, dtype=torch.long)
            c2 = extract(self.posterior_mean_coef2, t_test, (len(t_test),1,1,1))
            assert (c2 <= 1.0 + 1e-5).all(), f"posterior_mean_coef2>1 found: {c2.flatten().tolist()}"

    # ---------- forward / posterior math ----------

        # ---- correlated noise: N(0,1) in image space -> DWT -> LL÷3 format ----
    def _make_wavelet_noise(self, x_like: torch.Tensor) -> torch.Tensor:
        # x_like: [B,4,Hh,Wh] in LL÷3 format
        B, _, Hh, Wh = x_like.shape
        device = x_like.device
        noise_img = torch.randn(B, 1, Hh*2, Wh*2, device=device)
        w = dwt_haar_1level_batched(noise_img)   # [B,4,Hh,Wh]
        w[:, :1] = w[:, :1] / 3.0                # keep LL in ÷3 convention
        return w

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
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape).clamp(max=0.9995)
        mean  = c1 * x_start + c2 * x_t
        var   = extract(self.posterior_variance,             t, x_t.shape)
        log_v = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_v

    # ---------- WDM clamp helper (public so debuggers can call it) ----------

    @torch.no_grad()
    def process_xstart_wdm(self, x0_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Coeffs -> (LL×3) -> IDWT -> clamp [-1,1] -> DWT -> (LL÷3), all batched.
        """
        x0_unscaled = x0_coeffs.clone()
        x0_unscaled[:, :1] = x0_unscaled[:, :1] * 3.0     # LL ×3
        img = idwt_haar_1level(x0_unscaled).clamp_(-1.0, 1.0)
        w  = dwt_haar_1level_batched(img)
        w[:, :1] = w[:, :1] / 3.0                         # LL ÷3
        return w

    # ---------- reverse step ----------

    def p_mean_variance(self, x: torch.Tensor, t: torch.LongTensor,
                        clip_denoised: bool, c=None, soft_a=None):
        """
        One reverse step in wavelet space.
        - If predict_x0: model outputs x0 directly
        - Else: model outputs ε, we reconstruct x0
        - Optional per-step pixel-space clamp (clip_denoised=True)
        """
        # prepare model input
        if self.with_condition and c is not None:
            model_in = torch.cat([x, c], dim=1)
        else:
            model_in = x

        # predict x0 or eps
        if self.predict_x0:
            x0_hat = self.denoise_fn(model_in, t)                 # wavelet coeffs, LL already ÷3
        else:
            eps = self.denoise_fn(model_in, t)
            x0_hat = self.predict_start_from_noise(x, t=t, noise=eps)

        # WDM pixel-space clamp
        if clip_denoised:
            x0_hat = self.process_xstart_wdm(x0_hat)

        # posterior
        model_mean, posterior_var, posterior_log_var = self.q_posterior(
            x_start=x0_hat, x_t=x, t=t
        )
        return model_mean, posterior_var, posterior_log_var

    # ---------- sampling ----------

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.LongTensor, condition_tensors=None,
                 clip_denoised: bool = True, repeat_noise: bool = False, soft_a=None):
        """
        Single reverse step with optional WDM clamp (clip_denoised=True).
        """
        b, _, h, w = x.shape
        device = x.device

        # align cond
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

        if self.debug_p_sample:
            c1_dbg = extract(self.posterior_mean_coef1, t, x.shape)
            c2_dbg = extract(self.posterior_mean_coef2, t, x.shape).clamp(max=0.9995)
            mm_from_xt = (c2_dbg * x).abs().mean().item()
            print(f"[p_sample] t={int(t[0])} c1={c1_dbg.mean().item():.6f} "
                  f"c2={c2_dbg.mean().item():.6f} |x_t|={x.abs().mean().item():.3e} "
                  f"|mean|={model_mean.abs().mean().item():.3e} |c2*x_t|={mm_from_xt:.3e}")

        noise = noise_like(x.shape, device, repeat_noise)
        nonzero = (1 - (t == 0).float()).reshape(b, *((1,) * (x.ndim - 1)))
        return model_mean + nonzero * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors=None, clip_denoised: bool = True, soft_a=None):
        """
        Returns coeffs of shape (B,4,H/2,W/2).
        """
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
            img = self.p_sample(img, t, condition_tensors=c, clip_denoised=clip_denoised)

        return img

    @torch.no_grad()
    def sample(self, batch_size=2, condition_tensors=None):
        return self.p_sample_loop(
            (batch_size, self.channels, self.image_size, self.image_size),
            condition_tensors=condition_tensors,
            clip_denoised=True
        )

    # ---------- training ----------

    def q_sample(self, x_start, t, noise=None):
        noise = noise if noise is not None else self._make_wavelet_noise(x_start)
        return (
            extract(self.sqrt_alphas_cumprod,           t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, condition_tensors=None, noise=None):
        """
        x_start: wavelet coeffs [B,4,H/2,W/2] from dataset (LL already ÷3)
        condition_tensors: [B,1,H/2,W/2]
        """
        b, c, h, w = x_start.shape
        if noise is None:
            noise = self._make_wavelet_noise(x_start)

        # align cond
        if self.with_condition:
            if condition_tensors is None:
                raise ValueError("with_condition=True but condition_tensors is None")
            if condition_tensors.shape[-2:] != (h, w):
                condition_tensors = F.interpolate(condition_tensors, size=(h, w), mode='nearest')

        # diffuse forward
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_in = torch.cat([x_noisy, condition_tensors], dim=1) if self.with_condition else x_noisy

        if self.predict_x0:
            # x0-prediction (WDM-style). Target is x_start (LL already ÷3)
            x0_pred = self.denoise_fn(model_in, t)
            if self.loss_type == 'l2':
                #loss = (x0_pred - x_start).pow(2).mean()
                if self.band_weights is None:
                    loss = (x0_pred - x_start).pow(2).mean()
                else:
                    per = []
                    for band in range(4):
                        per.append(((x0_pred[:,band:band+1]-x_start[:,band:band+1])**2).mean() * self.band_weights[band])
                    loss = sum(per)
            elif self.loss_type == 'l1':
                #loss = F.smooth_l1_loss(x0_pred, x_start, beta=1.0, reduction='mean')
                if self.band_weights is None:
                    loss = F.smooth_l1_loss(x0_pred, x_start, beta=1.0, reduction='mean')
                else:
                    per = []
                    for band in range(4):
                        band_loss = F.smooth_l1_loss(x0_pred[:,band:band+1], x_start[:,band:band+1], beta=1.0, reduction='mean')
                        per.append(band_loss * self.band_weights[band])
                    loss = sum(per)
            else:
                raise NotImplementedError(self.loss_type)
        else:
            # ε-prediction (classic DDPM)
            eps_pred = self.denoise_fn(model_in, t)
            #noise = noise if noise is not None else self._make_wavelet_noise(x_start)
            if self.loss_type == 'l2':
                loss = (eps_pred - noise).pow(2).mean()
            elif self.loss_type == 'l1':
                loss = F.smooth_l1_loss(eps_pred, noise, beta=1.0, reduction='mean')
            else:
                raise NotImplementedError(self.loss_type)

        return loss

    def forward(self, x, condition_tensors=None, *args, **kwargs):
        b, c, h, w = x.shape
        img_size = self.image_size
        assert (h, w) == (img_size, img_size), f'Expected {(img_size,img_size)}, got {(h,w)}'
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
            clip_denoised=True
        )

        if de_standardize and (mu is not None) and (std is not None):
            mu  = mu.view(1,4,1,1).to(device)
            std = std.view(1,4,1,1).to(device)
            coeffs = coeffs * (std + 1e-6) + mu

        vis = coeffs.clone()
        vis[:, :1] *= 3.0                    # unscale LL for IDWT
        imgs = idwt_haar_1level(vis)
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
