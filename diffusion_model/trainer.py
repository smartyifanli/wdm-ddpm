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

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 1,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None,
        with_condition = False,
        with_pairwised = False,
        apply_bce = False,
        lambda_bce = 0.0,
        band_loss_weights=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.with_condition = with_condition
        self.with_pairwised = with_pairwised
        self.apply_bce = apply_bce
        self.lambda_bce = lambda_bce
        self.band_loss_weights = None

        if band_loss_weights is not None:
            w = torch.tensor(band_loss_weights, dtype=torch.float32)
            assert w.numel() == 4, "band_loss_weights must have 4 entries [LL,LH,HL,HH]"
            self.register_buffer('band_loss_weights_buf', w)
            self.band_loss_weights = 'buf'  # marker to say it's present

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef3', to_torch(
            1. - (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t, c=None):
        x_hat = 0
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + x_hat
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, c=None):
        x_hat = 0.
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise -
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
        )

    def q_posterior(self, x_start, x_t, t, c=None):
        x_hat = 0.
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, c=None):
        if self.with_condition:
            eps = self.denoise_fn(torch.cat([x, c], 1), t)
        else:
            eps = self.denoise_fn(x, t)

        x_recon = self.predict_start_from_noise(x, t=t, noise=eps)

        # --- NEW: dynamic thresholding in standardized coeff space ---
        if clip_denoised:  # treat this flag as "apply safe bound"
            # per-sample, get 99.5% quantile of |x0|
            B = x_recon.shape[0]
            x_flat = x_recon.view(B, -1).abs()
            s = torch.quantile(x_flat, 0.995, dim=1, keepdim=True)  # [B,1]
            s = s.view(B, 1, 1, 1).clamp(min=1.0)                   # avoid tiny s
            x_recon = (x_recon / s).clamp(-1, 1) * s                # scale then clip
        # ------------------------------------------------------------

        model_mean, var, logvar = self.q_posterior(x_start=x_recon, x_t=x, t=t, c=c)
        return model_mean, var, logvar


    @torch.no_grad()
    def p_sample(self, x, t, condition_tensors=None, clip_denoised=None, repeat_noise=False):
        """
        One reverse step. In wavelet space we default to no clamp.
        x:   [B, 4, H/2, W/2]
        cond:[B, 1, H/2, W/2]
        """
        if clip_denoised is None:
            # In coefficient space, pixel-range clamping [-1,1] is not appropriate.
            clip_denoised = False

        b, _, h, w = x.shape
        device = x.device

        # keep cond aligned
        if self.with_condition:
            if condition_tensors is None:
                raise ValueError("with_condition=True but condition_tensors is None")
            if condition_tensors.shape[-2:] != (h, w):
                condition_tensors = F.interpolate(condition_tensors, size=(h, w), mode='nearest')
            condition_tensors = condition_tensors.to(device, non_blocking=True)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, c=condition_tensors, clip_denoised=clip_denoised
        )

        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (x.ndim - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors=None, clip_denoised=None):
        """
        shape: (B, 4, H/2, W/2)  —  sampling directly in wavelet space
        Returns: [B, 4, H/2, W/2] coefficients (do iDWT outside this function).
        """
        device = self.betas.device
        b, _, h, w = shape
        img = torch.randn(shape, device=device)

        if clip_denoised is None:
            clip_denoised = False  # default for coeff space

        # validate / align condition once up-front
        cond = None
        if self.with_condition:
            if condition_tensors is None:
                raise ValueError("with_condition=True but condition_tensors is None")
            cond = condition_tensors
            if cond.shape[-2:] != (h, w):
                cond = F.interpolate(cond, size=(h, w), mode='nearest')
            cond = cond.to(device, non_blocking=True)

        for i in tqdm(reversed(range(0, self.num_timesteps)),
                    desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, condition_tensors=cond, clip_denoised=clip_denoised)

        return img


    @torch.no_grad()
    def sample(self, batch_size = 2, condition_tensors = None):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), condition_tensors = condition_tensors)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def q_sample(self, x_start, t, noise=None, c=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_hat = 0.
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise + x_hat
        )

    def p_losses(self, x_start, t, condition_tensors=None, noise=None):
        """
        x_start: [B, 4, H/2, W/2]  (wavelet coeffs)
        condition_tensors: [B, 1, H/2, W/2] (already downsampled by dataset)
        """
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # ensure cond matches spatial size (robust even if a mismatch slips in)
        if self.with_condition:
            if condition_tensors is None:
                raise ValueError("with_condition=True but condition_tensors is None")
            if condition_tensors.shape[-2:] != (h, w):
                condition_tensors = F.interpolate(condition_tensors, size=(h, w), mode='nearest')

        # diffuse target
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # concat noisy target + condition (channel-wise)
        if self.with_condition:
            model_in = torch.cat([x_noisy, condition_tensors], dim=1)  # [B, 4+1, H/2, W/2]
        else:
            model_in = x_noisy

        #x_recon = self.denoise_fn(model_in, t)  # expect out_channels == 4

        # predict noise in wavelet space
        x_recon = self.denoise_fn(model_in, t)  # [B,4,H/2,W/2]

        # raw error per pixel
        if self.loss_type == 'l1':
            err = (x_recon - noise).abs()
        elif self.loss_type == 'l2':
            err = (x_recon - noise) ** 2
        else:
            raise NotImplementedError()

        # optional band weighting: downweight LL, upweight HF
        if hasattr(self, 'band_loss_weights_buf'):
            w = self.band_loss_weights_buf.view(1, 4, 1, 1)
            err = err * w

        loss = err.mean()
        return loss



    def forward(self, x, condition_tensors=None, *args, **kwargs):
        b, c, h, w, device, img_size = *x.shape, x.device, self.image_size  # Removed depth_size
        assert h == img_size and w == img_size, f'Expected dimensions: height={img_size}, width={img_size}. Actual: height={h}, width={w}.'
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
