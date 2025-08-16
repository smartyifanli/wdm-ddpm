# train_wdm_2D.py (updated)
# -*- coding:utf-8 -*-
"""
WDM-style 2D training script (LLÃ·3, per-step pixel clamp in image space).
Assumes your diffusion model implements the WDM clamp inside
GaussianDiffusion.p_mean_variance(..., clip_denoised=True).
Optionally supports x0-prediction if your GaussianDiffusion has `predict_x0`.
"""

from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
#from dataset_wavelet import CTImageGenerator, CTPairImageGenerator, create_train_val_test_datasets
from dataset import Wavelet2DDataset, create_train_val_datasets_9_1_split_wavelet
import argparse
import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import time
from pathlib import Path
from diffusion_model.trainer import idwt_haar_1level
from torchvision.utils import save_image
# -----------------------------------------------------------------------------
# CUDA selection (optional; adjust as needed)
# -----------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_root', type=str, default="/storage/data/TRAIL_Yifan/MH/")
parser.add_argument('--input_size', type=int, default=512)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=1)
parser.add_argument('--train_lr', type=float, default=1e-6)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--timesteps', type=int, default=500)
parser.add_argument('--save_and_sample_every', type=int, default=500)
parser.add_argument('--with_condition', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="")
parser.add_argument('--val_results_dir', type=str, default="/storage/data/li46460_wdm_ddpm/wdm-ddpm/val_results")
parser.add_argument('--test_results_dir', type=str, default="/storage/data/li46460_wdm_ddpm/wdm-ddpm/results")
parser.add_argument('--images_dir', type=str, default="/storage/data/li46460_wdm_ddpm/wdm-ddpm/images")
parser.add_argument('--run_test_after_training', action='store_true')

# Trainer / diffusion niceties
parser.add_argument('--loss_type', type=str, default='l1', choices=['l1','l2'])
parser.add_argument('--ema_decay', type=float, default=0.9999)
parser.add_argument('--gradient_clip_val', type=float, default=1.0)
parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear','cosine'])
parser.add_argument('--gradient_accumulate_every', type=int, default=2)
parser.add_argument('--band_weights', type=str, default='1.0,1.0,1.0,1.0')
parser.add_argument('--predict_x0', action='store_true', default=False, help='Use x0-prediction (wavelet MSE) if supported by diffusion class')

args = parser.parse_args()

# Unpack
DATA_ROOT = args.data_root
INPUT_SIZE = int(args.input_size)
WITH_COND = bool(args.with_condition)
SAVE_AND_SAMPLE_EVERY = int(args.save_and_sample_every)
TIMESTEPS = int(args.timesteps)
LOSS_TYPE = args.loss_type
EMA_DECAY = float(args.ema_decay)
GRAD_CLIP_VAL = float(args.gradient_clip_val)
GRAD_ACC = int(args.gradient_accumulate_every)
RUN_TEST = bool(args.run_test_after_training)
RESUME = args.resume_weight

VAL_DIR = args.val_results_dir
TEST_DIR = args.test_results_dir
IMAGES_DIR = args.images_dir

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _probe_ts(num_timesteps: int, fracs):
    Tm1 = int(num_timesteps) - 1
    return sorted({max(0, min(Tm1, int(round(f * Tm1)))) for f in fracs})


def ask_and_clear_dir(path, description):
    """
    Ask user if they want to clear the given directory.
    """
    if os.path.exists(path) and os.listdir(path):
        while True:
            response = input(f"The {description} '{path}' is not empty. Do you want to clear it? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                import shutil
                shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)
                print(f"âœ… Cleared {description}: {path}")
                break
            elif response in ['n', 'no']:
                print(f"â„¹ï¸ Keeping existing contents in {description}: {path}")
                break
            else:
                print("Please enter 'y' or 'n'.")
    else:
        os.makedirs(path, exist_ok=True)
        print(f"âœ… Created empty {description}: {path}")


def _clip_flags(ds):
    # For WDM we will always enable pixel clamp at sampling time via clip_denoised=True.
    # This function keeps compatibility but is not used to decide the clamp.
    mode = getattr(ds, 'clip_mode', 'none')
    use_clip = (mode == 'hard')
    K = float(getattr(ds, 'clip_k', 4.0))
    return use_clip, K

# -----------------------------------------------------------------------------
# Prepare dirs
# -----------------------------------------------------------------------------
ask_and_clear_dir(VAL_DIR,  "validation results folder")
ask_and_clear_dir(TEST_DIR, "test results folder")
ask_and_clear_dir(IMAGES_DIR, "images folder")

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
if WITH_COND:
    full_dataset = Wavelet2DDataset(
        data_root=DATA_ROOT,
        input_size=INPUT_SIZE,
        with_condition=True,
    )

    # Debug first sample
    print("\nðŸ” DEBUGGING FIRST SAMPLE:")
    if len(full_dataset) > 0:
        cond, coeffs = full_dataset[0]
        print(f"  Input (mask) shape: {cond.shape}  range=[{cond.min():.4f}, {cond.max():.4f}]  mean={cond.mean():.4f}  std={cond.std():.4f}")
        print(f"  Target (coeffs) shape: {coeffs.shape}  range=[{coeffs.min():.4f}, {coeffs.max():.4f}]  mean={coeffs.mean():.4f}  std={coeffs.std():.4f}")
        pos = torch.sum(cond > -0.5).item(); tot = cond.numel()
        print(f"  Input positive pixels: {pos}/{tot} ({pos/tot*100:.1f}%)")

    # subject-aware 9:1 split
    train_dataset, val_dataset, train_subjects, val_subjects = create_train_val_datasets_9_1_split_wavelet(
        full_dataset, random_state=42
    )
    test_dataset = full_dataset

    # For WDM we ignore coeff-space clip flags and force pixel clamp at sampling.
    full_dataset.clip_mode = 'none'

    # Save test indices (all indices)
    test_indices = list(range(len(full_dataset)))
    indices_file = os.path.join(IMAGES_DIR, 'test_indices.json')
    with open(indices_file, 'w') as f:
        json.dump({
            'test_indices': test_indices,
            'total_dataset_size': len(full_dataset),
            'test_size': len(test_dataset)
        }, f, indent=2)
    print(f"âœ… Test indices saved to: {indices_file}")

else:
    # Unconditional path (kept for compatibility)
    full_dataset = CTImageGenerator(DATA_ROOT, input_size=INPUT_SIZE)
    all_idx = list(range(len(full_dataset)))
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(all_idx, train_size=0.9, random_state=42)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)
    test_dataset  = full_dataset
    train_subjects = ["unconditional_train"]
    val_subjects   = ["unconditional_val"]

    test_indices = list(range(len(full_dataset)))
    indices_file = os.path.join(IMAGES_DIR, 'test_indices.json')
    with open(indices_file, 'w') as f:
        json.dump({
            'test_indices': test_indices,
            'total_dataset_size': len(full_dataset),
            'test_size': len(test_dataset)
        }, f, indent=2)
    print(f"âœ… Test indices saved to: {indices_file}")

print(f"Total dataset size: {len(full_dataset)}")
print(f"Train size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")
print(f"Test size:  {len(test_dataset)}")

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
image_size_half = INPUT_SIZE // 2
in_channels  = 4 + (1 if WITH_COND else 0)
out_channels = 4

model = create_model(
    image_size_half,
    args.num_channels,
    args.num_res_blocks,
    in_channels=in_channels,
    out_channels=out_channels,
).cuda()

# Manual init (optional)
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.GroupNorm):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

print("Applying manual weight initialization...")
model.apply(init_weights)
print("âœ… Model weights initialized")

# Diffusion wrapper (expects your GaussianDiffusion implements WDM clamp step)
band_weights = [float(x) for x in args.band_weights.split(',')]
diffusion = GaussianDiffusion(
    model,
    image_size=image_size_half,
    timesteps=TIMESTEPS,
    loss_type=LOSS_TYPE,
    with_condition=WITH_COND,
    channels=out_channels,
    predict_x0=args.predict_x0
).cuda()

# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------

def calculate_ssim(real_images, generated_images):
    try:
        from skimage.metrics import structural_similarity as ssim
        scores = []
        for i in range(len(real_images)):
            a = real_images[i].squeeze().cpu().numpy()
            b = generated_images[i].squeeze().cpu().numpy()
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            b = (b - b.min()) / (b.max() - b.min() + 1e-8)
            scores.append(ssim(a, b, data_range=1.0))
        return float(np.mean(scores))
    except Exception:
        print("SSIM requires scikit-image. Returning 0.0")
        return 0.0

# Organized save under subject/positive/wdm_generation

def save_generation_organized(generated_img, original_dataset, test_index, data_root):
    try:
        ct_path, mk_path = original_dataset.pair_files[test_index]
        ct_filename = os.path.basename(ct_path)
        import re
        m = re.search(r'(\d+)_(\d+)\.png$', ct_filename)
        if not m:
            print(f"âš ï¸ Could not parse filename: {ct_filename}")
            return False
        subject_id, slice_id = m.group(1), m.group(2)
        out_dir = os.path.join(data_root, subject_id, 'positive', 'wdm_generation')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{subject_id}_{slice_id}.png")
        save_image_png(generated_img, out_path)
        print(f"âœ… Saved: {subject_id}/positive/wdm_generation/{subject_id}_{slice_id}.png")
        return True
    except Exception as e:
        print(f"âŒ Error saving organized generation for index {test_index}: {e}")
        return False


def save_image_png(img_tensor, filepath):
    x = img_tensor.detach().cpu()
    if x.dim() == 4 and x.size(1) == 1:
        x = x[0]  # [1,H,W]
    if x.dim() == 3 and x.size(0) == 1:
        x = x[0]
    # normalize to [0,255]
    x = x.float()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = (x * 255.0).clamp(0,255).byte().cpu().numpy()
    from PIL import Image
    Image.fromarray(x).save(filepath)

# -----------------------------------------------------------------------------
# Validation + custom Trainer
# -----------------------------------------------------------------------------
class CTTrainer(Trainer):
    def __init__(self, val_dataset, val_dir, images_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_dataset = val_dataset
        self.val_dir = val_dir
        self.images_dir = images_dir
        self.loss_history = []

    def plot_loss_curve(self):
        if not self.loss_history:
            return
        steps = [d['step'] for d in self.loss_history]
        losses = [d['loss'] for d in self.loss_history]
        plt.figure(figsize=(10,4))
        plt.plot(steps, losses, linewidth=1.25)
        plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.grid(True, alpha=0.3)
        out = os.path.join(self.images_dir, 'loss_curve.png')
        plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
        print(f"ðŸ“Š Loss curve saved to: {out}")

    @torch.no_grad()
    def validate_and_save(self, milestone):
        print("Generating validation samples...")
        step_dir = os.path.join(self.val_dir, f"epoch_{milestone * self.save_and_sample_every}")
        os.makedirs(step_dir, exist_ok=True)
        n = min(4, len(self.val_dataset))
        for i in range(n):
            try:
                base_ds = self.val_dataset.dataset if isinstance(self.val_dataset, Subset) else self.val_dataset
                ct_path, mk_path = base_ds.file_paths_at(self.val_dataset.indices[i] if isinstance(self.val_dataset, Subset) else i)
                subject_id, slice_id = base_ds.get_subject_slice_from_ct(ct_path)

                cond, target_coeffs = self.val_dataset[i]
                cond = cond.unsqueeze(0).cuda().float()        # [1,1,H/2,W/2]
                target_coeffs = target_coeffs.unsqueeze(0).cuda().float()  # [1,4,H/2,W/2]

                Hh = self.image_size
                # WDM: force pixel clamp each step
                gen_coeffs = self.ema_model.p_sample_loop(
                    shape=(1,4,Hh,Hh),
                    condition_tensors=cond,
                    clip_denoised=True,
                )

                # For visualization: unscale LL (Ã—3) before IDWT
                gen_vis = gen_coeffs.clone(); gen_vis[:,0] *= 3.0
                tgt_vis = target_coeffs.clone(); tgt_vis[:,0] *= 3.0

                # Mix A/B (be sure to unscale LL before IDWT)
                mixA = gen_coeffs.clone();  mixA[:,1:] = target_coeffs[:,1:]
                mixB = target_coeffs.clone(); mixB[:,1:] = gen_coeffs[:,1:]
                mixA_vis = mixA.clone(); mixA_vis[:,0] *= 3.0
                mixB_vis = mixB.clone(); mixB_vis[:,0] *= 3.0

                gen_img01 = (idwt_haar_1level(gen_vis).clamp(-1,1) + 1) * 0.5
                tgt_img01 = (idwt_haar_1level(tgt_vis).clamp(-1,1) + 1) * 0.5
                mixA_img01 = (idwt_haar_1level(mixA_vis).clamp(-1,1) + 1) * 0.5
                mixB_img01 = (idwt_haar_1level(mixB_vis).clamp(-1,1) + 1) * 0.5

                subj_dir = os.path.join(step_dir, subject_id, 'positive', 'generation')
                os.makedirs(subj_dir, exist_ok=True)
                save_image((cond+1)*0.5, os.path.join(subj_dir, f"{subject_id}_{slice_id}_mask.png"))
                save_image(tgt_img01, os.path.join(subj_dir, f"{subject_id}_{slice_id}_target.png"))
                save_image(gen_img01, os.path.join(subj_dir, f"{subject_id}_{slice_id}_generated.png"))
                save_image(mixA_img01, os.path.join(subj_dir, f"{subject_id}_{slice_id}_mixA_genLL_targetHF.png"))
                save_image(mixB_img01, os.path.join(subj_dir, f"{subject_id}_{slice_id}_mixB_targetLL_genHF.png"))
                print(f"Saved {subject_id}_{slice_id} to {subj_dir}")

            except Exception as e:
                print(f"Error generating validation sample {i}: {e}")
                continue

    def train(self):
        from functools import partial
        backwards = partial(loss_backwards, self.fp16)
        start = time.time()
        while self.step < self.train_num_steps:
            acc = 0.0
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dl)
                if self.with_condition:
                    if isinstance(batch, (list,tuple)) and len(batch)==2:
                        cond, target = batch
                    elif isinstance(batch, dict):
                        cond, target = batch['input'], batch['target']
                    else:
                        raise ValueError("Batch must be (cond,target) or dict with keys 'input'/'target'")
                    cond = cond.cuda(non_blocking=True).float()
                    target = target.cuda(non_blocking=True).float()
                    loss = self.model(target, condition_tensors=cond)
                else:
                    data = batch.cuda(non_blocking=True).float()
                    loss = self.model(data)
                if loss.ndim:
                    loss = loss.mean()
                print(f"{self.step}.{i}: {loss.item():.6f}")
                self.loss_history.append({'step': self.step, 'loss': float(loss.item())})
                backwards(loss / self.gradient_accumulate_every, self.opt)
                acc += float(loss.item())

            avg_loss = acc / float(self.gradient_accumulate_every)
            self.writer.add_scalar("training_loss", avg_loss, self.step)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_VAL)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
            self.opt.step(); self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                ms = self.step // self.save_and_sample_every
                self.save(ms)
                self.validate_and_save(ms)
                self.plot_loss_curve()

            self.step += 1

        print("training completed")
        self.plot_loss_curve()
        elapsed_h = (time.time() - start) / 3600.0
        self.writer.add_hparams(
            {"lr": self.train_lr, "batchsize": self.train_batch_size, "image_size": self.image_size, "hours": elapsed_h},
            {"last_loss": avg_loss}
        )
        self.writer.close()

# Separate backward helper (kept for AMP compatibility)

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        try:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(**kwargs)
        except Exception:
            print("APEX not available; falling back to regular backward")
            loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# -----------------------------------------------------------------------------
# Optional Wavelet Debugger (lightweight, WDM-compatible)
# -----------------------------------------------------------------------------
class WaveletDebugger:
    def __init__(self, dataset, model, diffusion, save_dir="debug_outputs"):
        self.dataset  = dataset
        self.base_ds  = dataset.dataset if isinstance(dataset, Subset) else dataset
        self.model    = model
        self.diffusion = diffusion
        from pathlib import Path
        self.save_dir = Path(save_dir); self.save_dir.mkdir(exist_ok=True)
        self.debug_wdm_processing()
        
    def debug_wdm_processing(self):
        """Debug WDM-specific coefficient processing"""
        print("\nðŸ”§ WDM PROCESSING ANALYSIS\n" + "-"*40)
        
        cond, coeffs = self.dataset[0]
        
        # Show dataset coeffs (LL already scaled)
        self._coef_stats(coeffs, "dataset (LLÃ·3)")
        
        # Show what happens when we unscale for visualization
        coeffs_unscaled = coeffs.clone()
        coeffs_unscaled[0] = coeffs_unscaled[0] * 3.0
        self._coef_stats(coeffs_unscaled, "unscaled for iDWT (LLÃ—3)")
        
        # Test the WDM clamp function
        test_coeffs = coeffs.unsqueeze(0).cuda()
        processed = self.diffusion.process_xstart_wdm(test_coeffs)
        self._coef_stats(processed.squeeze(0).cpu(), "after WDM clamp")
        
        # Energy ratio check
        ll_energy = coeffs_unscaled[0].pow(2).mean()
        hf_energy = coeffs_unscaled[1:].pow(2).mean()
        print(f"Energy ratio LL:HF = {ll_energy/hf_energy:.1f}:1 (should be >10)")

    
    def _backward_probe(self, coeffs, cond):
        """Debug the backward sampling process - where explosions typically occur"""
        print(f"\nâª DIFFUSION BACKWARD PROCESS\n" + "-"*40)
        
        B = 1
        x0 = coeffs.unsqueeze(0).cuda()      # [1,4,H/2,W/2] 
        c  = cond.unsqueeze(0).cuda()        # [1,1,H/2,W/2]
        T = self.diffusion.num_timesteps
        
        # Test model predictions at different noise levels
        self.model.eval()
        with torch.no_grad():
            for tval in _probe_ts(T, [0.75, 0.5, 0.25, 0.1, 0.05]):
                t = torch.tensor([tval], device='cuda', dtype=torch.long)
                
                # Add noise (forward)
                noise = torch.randn_like(x0)
                x_noisy = self.diffusion.q_sample(x0, t, noise=noise)
                
                # Model prediction (backward)
                if self.diffusion.with_condition:
                    model_input = torch.cat([x_noisy, c], dim=1)
                else:
                    model_input = x_noisy
                
                if self.diffusion.predict_x0:
                    x0_pred = self.model(model_input, t)
                    print(f"t={tval:3d}: x0_pred range=[{x0_pred.min():8.3f},{x0_pred.max():8.3f}] std={x0_pred.std():6.3f}")
                else:
                    eps_pred = self.model(model_input, t)
                    x0_pred = self.diffusion.predict_start_from_noise(x_noisy, t, eps_pred)
                    print(f"t={tval:3d}: eps_pred range=[{eps_pred.min():8.3f},{eps_pred.max():8.3f}] std={eps_pred.std():6.3f}")
                    print(f"      x0_pred range=[{x0_pred.min():8.3f},{x0_pred.max():8.3f}] std={x0_pred.std():6.3f}")
                
                # Test WDM clamp (critical!)
                x0_clamped = self.diffusion.process_xstart_wdm(x0_pred)
                print(f"      WDM_clamp range=[{x0_clamped.min():8.3f},{x0_clamped.max():8.3f}] std={x0_clamped.std():6.3f}")
                
                # Check for explosions
                if x0_pred.abs().max() > 100:
                    print(f"ðŸš¨ EXPLOSION DETECTED at t={tval}: max={x0_pred.abs().max():.1f}")
                    self._analyze_explosion(x0_pred)
                elif x0_pred.abs().max() > 10:
                    print(f"âš ï¸  Large values at t={tval}: max={x0_pred.abs().max():.1f}")

    def _analyze_explosion(self, x0_pred):
        """Analyze which bands are exploding"""
        print("   Band analysis:")
        bands = ['LL', 'LH', 'HL', 'HH']
        for i, band in enumerate(bands):
            b_max = x0_pred[0, i].abs().max().item()
            b_mean = x0_pred[0, i].abs().mean().item()
            print(f"   {band}: max={b_max:8.3f} mean={b_mean:6.3f}")

    def debug_pipeline(self):
        # Your existing debug
        print("\nðŸ” COMPREHENSIVE WAVELET DEBUGGING\n" + "="*60)
        cond, coeffs = (self.dataset[0] if not isinstance(self.dataset[0], dict)
                        else (self.dataset[0]['input'], self.dataset[0]['target']))
        print(f"Sample shapes: cond={cond.shape}, target_coeffs={coeffs.shape}")
        self._coef_stats(coeffs, "raw (dataset)")
        self._forward_probe(coeffs, cond)
        self._backward_probe(coeffs, cond)
        self._recon_check(coeffs)
        

    def _coef_stats(self, coeffs, name):
        print(f"\nðŸ“Š COEFFICIENT STATISTICS: {name}\n" + "-"*40)
        names = ['LL','LH','HL','HH']
        for i,nm in enumerate(names):
            b = coeffs[i].flatten()
            print(f"{nm:2s}: range=[{b.min().item():7.3f},{b.max().item():7.3f}] Î¼={b.mean().item():7.3f} Ïƒ={b.std().item():7.3f} energy={(b.pow(2).mean().item()):7.3f}")

    def _forward_probe(self, coeffs, cond):
        print("\nâ© DIFFUSION FORWARD PROCESS\n" + "-"*40)
        B = 1
        x0 = coeffs.unsqueeze(0).cuda()
        c  = cond.unsqueeze(0).cuda()
        T = self.diffusion.num_timesteps
        for tval in _probe_ts(T, [0.0,0.05,0.1,0.25,0.5,0.75,1.0]):
            t = torch.tensor([tval], device='cuda', dtype=torch.long)
            xt = self.diffusion.q_sample(x0, t)
            print(f"t={tval:3d}: range=[{xt.min():8.3f},{xt.max():8.3f}] std={xt.std():6.3f} mean={xt.mean():7.3f}")

    def _recon_check(self, coeffs):
        print("\nðŸ”„ RECONSTRUCTION QUALITY TEST\n" + "-"*40)
        vis = coeffs.unsqueeze(0).clone(); vis[:,0] *= 3.0
        recon = idwt_haar_1level(vis)
        print(f"Reconstruction range: [{recon.min():.3f},{recon.max():.3f}]")
        out = (recon.clamp(-1,1) + 1)*0.5
        save_image(out, self.save_dir / "test_reconstruction.png")
        print(f"âœ… Test reconstruction saved to {self.save_dir / 'test_reconstruction.png'}")


def run_comprehensive_debug(dataset, model, diffusion):
    dbg = WaveletDebugger(dataset, model, diffusion)
    dbg.debug_pipeline()
    print("\nðŸ“‹ DEBUGGING COMPLETE! See:", dbg.save_dir)

# -----------------------------------------------------------------------------
# Kick off training
# -----------------------------------------------------------------------------


def main():
    trainer = CTTrainer(
        val_dataset,        # positional arg 1 for CTTrainer
        VAL_DIR,           # positional arg 2 for CTTrainer  
        IMAGES_DIR,        # positional arg 3 for CTTrainer
        diffusion,         # positional arg 1 for parent Trainer (diffusion_model)
        train_dataset,     # positional arg 2 for parent Trainer (dataset)
        # Now the rest as keyword arguments for parent Trainer
        image_size=image_size_half,
        train_batch_size=args.batchsize,
        train_lr=args.train_lr,
        train_num_steps=args.epochs,
        gradient_accumulate_every=GRAD_ACC,
        ema_decay=EMA_DECAY,
        fp16=False,
        with_condition=WITH_COND,
        save_and_sample_every=SAVE_AND_SAMPLE_EVERY,
    )


    # Optional pre-train debug
    run_comprehensive_debug(full_dataset, model, diffusion)
    diffusion.debug_p_sample = True

    # Resume (optional)
    if len(RESUME) > 0 and os.path.exists(RESUME):
        if RESUME.endswith("ema_model_final.pth"):
            diffusion.load_state_dict(torch.load(RESUME))
            print("âœ… EMA model loaded for inference")
        else:
            ckpt = torch.load(RESUME, map_location='cuda')
            trainer.step = ckpt.get('step', 0)
            trainer.model.load_state_dict(ckpt['model'])
            trainer.ema_model.load_state_dict(ckpt['ema'])
            if 'optimizer' in ckpt:
                trainer.opt.load_state_dict(ckpt['optimizer'])
                print("âœ… Optimizer state loaded")
            print(f"âœ… Training resumed from step {trainer.step}")

    # Train
    trainer.train()

    # Save EMA
    ema_ckpt = os.path.join(TEST_DIR, 'ema_model_final.pth')
    torch.save(trainer.ema_model.state_dict(), ema_ckpt)
    print(f"âœ… EMA weights saved to {ema_ckpt}")

    # Post-train debug with EMA-UNet
    unet_ema = trainer.ema_model.denoise_fn
    run_comprehensive_debug(val_dataset, unet_ema, trainer.ema_model)

    # Optional test sweep
    if RUN_TEST:
        print("\nðŸ§ª RUNNING TEST EVALUATION...")
        test_model_wavelet(trainer.ema_model, test_dataset, TEST_DIR, full_dataset, DATA_ROOT, WITH_COND)
    else:
        print("\nðŸ’¡ To run test evaluation later, pass --run_test_after_training")
        print("   Test indices:", os.path.join(IMAGES_DIR, 'test_indices.json'))


# -----------------------------------------------------------------------------
# Test-time generator (subject-aware save)
# -----------------------------------------------------------------------------
@torch.no_grad()
def test_model_wavelet(ema_model, test_dataset, out_dir, full_dataset, data_root, with_condition=True):
    ema_model.eval(); os.makedirs(out_dir, exist_ok=True)

    def base_and_idx(ds, i):
        return (ds.dataset, ds.indices[i]) if isinstance(ds, Subset) else (ds, i)

    saved = 0
    for i in range(len(test_dataset)):
        try:
            base_ds, orig_idx = base_and_idx(test_dataset, i)
            ct_path, mk_path = base_ds.file_paths_at(orig_idx)
            subject_id, slice_id = base_ds.get_subject_slice_from_ct(ct_path)

            cond, target_coeffs = test_dataset[i]
            cond = cond.unsqueeze(0).cuda()
            target_coeffs = target_coeffs.unsqueeze(0).cuda()

            Hh = ema_model.image_size
            # WDM: force pixel clamp ON
            gen_coeffs = ema_model.p_sample_loop(
                shape=(1,4,Hh,Hh),
                condition_tensors=cond,
                clip_denoised=True,
            )

            # Unscale LL for IDWT
            gen_vis = gen_coeffs.clone(); gen_vis[:,0] *= 3.0
            img01 = (idwt_haar_1level(gen_vis).clamp(-1,1) + 1) * 0.5

            subj_gen_dir = os.path.join(data_root, subject_id, 'positive', 'wdm_generation')
            os.makedirs(subj_gen_dir, exist_ok=True)
            save_image(img01, os.path.join(subj_gen_dir, f"{subject_id}_{slice_id}.png"))
            saved += 1
        except Exception as e:
            print(f"âŒ Test sample {i} failed: {e}")
    print(f"âœ… Saved {saved} generations to subject folders.")


if __name__ == "__main__":
    main()
