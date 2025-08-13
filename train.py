# train_2D.py
#-*- coding:utf-8 -*-
# +
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
#from dataset_wavelet import CTImageGenerator, CTPairImageGenerator, create_train_val_test_datasets
from dataset import CTImageGenerator, CTPairImageGenerator
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


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# -




parser = argparse.ArgumentParser()
#parser.add_argument('-l', '--labelfolder', type=str, default="../crop/labels/")
#parser.add_argument('-s', '--scanfolder', type=str, default="../crop/scans/")
parser.add_argument('-d', '--data_root', type=str, default="/storage/data/TRAIL_Yifan/MH/")
parser.add_argument('--input_size', type=int, default=512)  # Changed to 512 to preserve original size
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=1)  # Changed from 3 to 1 for CT
parser.add_argument('--train_lr', type=float, default=1e-6)
parser.add_argument('--batchsize', type=int, default=1)  # Reduced to 1 for 512x512 images
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--timesteps', type=int, default=100)  # Updated from 250 to 1000
parser.add_argument('--save_and_sample_every', type=int, default=1000)  # Changed to 100 epochs
parser.add_argument('--with_condition', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="")
parser.add_argument('--val_results_dir', type=str, default="/storage/data/li46460_wdm_ddpm/wdm-ddpm/val_results")
parser.add_argument('--test_results_dir', type=str, default="/storage/data/li46460_wdm_ddpm/wdm-ddpm/results")
parser.add_argument('--images_dir', type=str, default="/storage/data/li46460_wdm_ddpm/wdm-ddpm/images")
parser.add_argument('--run_test_after_training', action='store_true', help='Run test evaluation after training')

# ✅ ADD these new advanced parameters
parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2'], help='Loss function type')
parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate')
parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='Noise schedule')
parser.add_argument('--gradient_accumulate_every', type=int, default=2, help='Gradient accumulation steps')
parser.add_argument('--band_weights', type=str, default='2.5, 1.0, 1.0, 1.05',
                    help='Comma-separated weights for [LL,LH,HL,HH] in loss')


args = parser.parse_args()

#labelfolder = args.labelfolder
#scanfolder = args.scanfolder
data_root = args.data_root
input_size = args.input_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight
train_lr = args.train_lr
val_results_dir = args.val_results_dir
test_results_dir = args.test_results_dir
images_dir = args.images_dir
run_test_after_training = args.run_test_after_training
loss_type = args.loss_type
ema_decay = args.ema_decay
gradient_clip_val = args.gradient_clip_val
beta_schedule = args.beta_schedule
gradient_accumulate_every = args.gradient_accumulate_every


def _probe_ts(num_timesteps: int, fracs):
    """
    Map fractional positions in [0,1] to valid integer timesteps [0, T-1].
    Example: fracs=[0.05, 0.25, 0.5, 0.75, 1.0]
    """
    Tm1 = int(num_timesteps) - 1
    return sorted({max(0, min(Tm1, int(round(f * Tm1)))) for f in fracs})

# Create validation results directory
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
                print(f"✅ Cleared {description}: {path}")
                break
            elif response in ['n', 'no']:
                print(f"ℹ️ Keeping existing contents in {description}: {path}")
                break
            else:
                print("Please enter 'y' or 'n'.")
    else:
        os.makedirs(path, exist_ok=True)
        print(f"✅ Created empty {description}: {path}")

def _clip_flags(ds):
    """Return (use_clip, K) based only on ds.clip_mode."""
    mode = getattr(ds, 'clip_mode', 'none')
    use_clip = (mode == 'hard')       # only 'hard' uses clip_denoised in coeff space
    K = float(getattr(ds, 'clip_k', 4.0))
    return use_clip, K

def _destandardize(coeffs, ds):
    """Map standardized (and maybe hard-clipped) coeffs back to raw wavelet space."""
    if not getattr(ds, 'standardize', False):
        return coeffs
    use_clip, K = _clip_flags(ds)
    x = coeffs
    if use_clip:                      # only undo hard [-K,K]/K
        x = x * K
    mu  = ds.mu.view(1, 4, 1, 1).to(x.device)
    std = ds.sigma.view(1, 4, 1, 1).to(x.device)
    return x * (std + 1e-6) + mu

# Ask user to clear paths at the start of training
ask_and_clear_dir(args.val_results_dir, "validation results folder")
ask_and_clear_dir(args.test_results_dir, "test results folder")
ask_and_clear_dir(args.images_dir, "images folder")


if with_condition:
    full_dataset = Wavelet2DDataset(
        data_root=data_root,
        input_size=input_size,
        with_condition=with_condition,
        standardize=False,
        clip_mode='none',   # or 'soft'
        clip_k=6.0,
        soft_a=4.0
    )



    # Debug: Check first sample
    print("🔍 DEBUGGING FIRST SAMPLE:")
    # 🔍 DEBUGGING FIRST SAMPLE:
    if len(full_dataset) > 0:
        cond, coeffs = full_dataset[0]          # cond:[1,H/2,W/2], coeffs:[4,H/2,W/2]
        print(f"  Input (mask) shape: {cond.shape}  range=[{cond.min():.4f}, {cond.max():.4f}]  mean={cond.mean():.4f}  std={cond.std():.4f}")
        print(f"  Target (coeffs) shape: {coeffs.shape}  range=[{coeffs.min():.4f}, {coeffs.max():.4f}]  mean={coeffs.mean():.4f}  std={coeffs.std():.4f}")

        # Simple sparsity-ish checks (mask is −1/1; treat > −0.5 as ‘on’)
        input_positive  = torch.sum(cond > -0.5).item()
        total_pixels    = cond.numel()
        target_positive = torch.sum(coeffs > -0.5).item()
        print(f"  Input positive pixels: {input_positive}/{total_pixels} ({input_positive/total_pixels*100:.1f}%)")
        print(f"  Target positive (>−0.5) coeff-px: {target_positive}/{coeffs.numel()} ({target_positive/coeffs.numel()*100:.1f}%)")

    
    # Split into train/val/test with 7:1:2 ratio and save indices
    train_dataset, val_dataset, train_subjects, val_subjects = create_train_val_datasets_9_1_split_wavelet(
        full_dataset, random_state=42
    )
    test_dataset = full_dataset
    
    # --- add in train.py, after you build (train_dataset, val_dataset) ---
    def compute_wavelet_band_stats_on_subset(ds_subset):
        """Return per-band (mu, sigma) over the subset, without changing dataset contents."""
        # ds_subset is a torch.utils.data.Subset pointing to Wavelet2DDataset
        base_ds = ds_subset.dataset if isinstance(ds_subset, torch.utils.data.Subset) else ds_subset
        indices = ds_subset.indices if isinstance(ds_subset, torch.utils.data.Subset) else range(len(ds_subset))

        # running sums to avoid big tensors in memory
        sum_b = torch.zeros(4, dtype=torch.float64)
        sumsq_b = torch.zeros(4, dtype=torch.float64)
        n_pix = 0

        for i in indices:
            _, w = base_ds[i]                 # w: [4, H/2, W/2]  (non-standardized in your current setup)
            w = w.double()
            sum_b += w.view(4, -1).sum(dim=1)
            sumsq_b += (w.view(4, -1) ** 2).sum(dim=1)
            n_pix += w[0].numel()

        mu = (sum_b / n_pix).float()
        var = (sumsq_b / n_pix - (sum_b / n_pix) ** 2).float().clamp_min(1e-8)
        sigma = var.sqrt()
        return mu, sigma
    

    # after create_train_val_datasets_9_1_split_wavelet(...)
    mu, sigma = compute_wavelet_band_stats_on_subset(train_dataset)

    # Attach to the SAME dataset instance that backs your Subsets,
    # then enable standardization so __getitem__ starts normalizing the bands.
    full_dataset.mu = mu
    full_dataset.sigma = sigma
    full_dataset.standardize = True
    full_dataset.clip_mode = 'none'         # sampling will use clip_denoised=False
    # full_dataset.clip_k = 6.0             # only if you switch to 'hard'
    # full_dataset.soft_a = 4.0             # only if you switch to 'soft'

    print(f"Per-band μ: {mu.tolist()}")
    print(f"Per-band σ: {sigma.tolist()}")


    # Save test indices for reproducible testing
    test_indices = list(range(len(full_dataset)))  # All indices since test = whole dataset
    indices_file = os.path.join(images_dir, 'test_indices.json')
    with open(indices_file, 'w') as f:
        json.dump({
            'test_indices': test_indices,
            'total_dataset_size': len(full_dataset),
            'test_size': len(test_dataset)
        }, f, indent=2)
    print(f"✅ Test indices saved to: {indices_file}")
    
else:
    full_dataset = CTImageGenerator(
        data_root,
        input_size=input_size,
    )
    
    # Use simple splitting for unconditional (9:1 train:val, all as test)
    all_indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        all_indices, train_size=0.9, random_state=42
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices) 
    test_dataset = full_dataset  # Whole dataset as test
    train_subjects = ["unconditional_train"]
    val_subjects = ["unconditional_val"]

    # Save test indices
    test_indices = list(range(len(full_dataset)))  # All indices since test = whole dataset
    indices_file = os.path.join(images_dir, 'test_indices.json')
    with open(indices_file, 'w') as f:
        json.dump({
            'test_indices': test_indices,
            'total_dataset_size': len(full_dataset),
            'test_size': len(test_dataset)
        }, f, indent=2)
    print(f"✅ Test indices saved to: {indices_file}")

print(f"Total dataset size: {len(full_dataset)}")
print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")


image_size_half = input_size // 2
in_channels  = 4 + (1 if with_condition else 0)   # =5 when conditional
out_channels = 4

model = create_model(
    image_size_half,          # UNet’s spatial size now H/2, W/2
    num_channels,
    num_res_blocks,
    in_channels=in_channels,  # 5
    out_channels=out_channels # 4
).cuda()

# ✅ ADD: Manual weight initialization to fix zero outputs
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
print("✅ Model weights initialized")

band_weights = [float(x) for x in args.band_weights.split(',')]
diffusion = GaussianDiffusion(
    model,
    image_size=image_size_half,
    timesteps=args.timesteps,
    loss_type=loss_type,
    with_condition=with_condition,
    channels=out_channels,
    band_loss_weights=band_weights,
).cuda()

diffusion.debug_p_sample     = False
diffusion.ll_x0_aux_weight   = 0.12                 # your new choice
diffusion.ll_aux_tmax        = int(0.30 * diffusion.num_timesteps)  # ~30% of T

if with_condition:
    # 4) Compute standardized LL bounds (from raw [-2, +2])
    mu_LL  = mu[0].item()         # or full_dataset.mu[0].item() (same now)
    sig_LL = sigma[0].item()
    lo_std = (-2.0 - mu_LL) / (sig_LL + 1e-6)
    hi_std = ( +2.0 - mu_LL) / (sig_LL + 1e-6)

    # 5) Register the bounds (slightly soft)
    diffusion.set_ll_bounds_std(lo_std, hi_std, softness=1.0)
    print(f"LL bounds (std space): lo={lo_std:.3f}, hi={hi_std:.3f}")
    #if len(resume_weight) > 0 and os.path.exists(resume_weight):
    #    weight = torch.load(resume_weight, map_location='cuda')
    #    diffusion.load_state_dict(weight['ema'])
    #    print("Model Loaded!")
else:
    # optional: skip LL bounds entirely for unconditional CTImageGenerator
    pass


def loss_backwards(fp16, loss, optimizer, **kwargs):
    """Handle backward pass with optional mixed precision"""
    if fp16:
        try:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(**kwargs)
        except ImportError:
            print("APEX not available, using regular backward")
            loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

def calculate_fid(real_images, generated_images):
    """Calculate FID score between real and generated images"""
    try:
        from pytorch_fid import fid_score
        from pytorch_fid.inception import InceptionV3
        import torch.nn.functional as F
        
        # Convert images to RGB if needed and resize to 299x299 for Inception
        def prepare_images(images):
            if len(images.shape) == 3:  # Single image
                images = images.unsqueeze(0)
            
            # Convert grayscale to RGB
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # Resize to 299x299 for Inception
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            return images
        
        real_prep = prepare_images(real_images)
        gen_prep = prepare_images(generated_images)
        
        # Calculate FID (simplified implementation)
        # This is a placeholder - you might want to use a proper FID implementation
        print("Note: FID calculation requires pytorch-fid package. Install with: pip install pytorch-fid")
        return 0.0  # Placeholder
        
    except ImportError:
        print("pytorch-fid not available. Install with: pip install pytorch-fid")
        return 0.0

def calculate_ssim(real_images, generated_images):
    """Calculate SSIM between real and generated images"""
    try:
        from skimage.metrics import structural_similarity as ssim
        
        ssim_scores = []
        for i in range(len(real_images)):
            real_img = real_images[i].squeeze().cpu().numpy()
            gen_img = generated_images[i].squeeze().cpu().numpy()
            
            # Normalize to [0, 1]
            real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min() + 1e-8)
            gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min() + 1e-8)
            
            score = ssim(real_img, gen_img, data_range=1.0)
            ssim_scores.append(score)
        
        return np.mean(ssim_scores)
        
    except ImportError:
        print("scikit-image not available. Install with: pip install scikit-image")
        return 0.0

def save_generation_organized(generated_img, original_dataset, test_index, data_root):
    """
    Save generated image in organized folder structure
    Args:
        generated_img: The generated image tensor/array
        original_dataset: The full dataset to get file paths
        test_index: Index in the original dataset (not subset)
        data_root: Root data directory
    """
    try:
        # Get original file paths
        mask_file, ct_file = original_dataset.pair_files[test_index]
        
        # Extract subject_id and slice_id from ct_file path
        # ct_file example: "/storage/data/TRAIL_Yifan/MH/2171/positive/ct/2171_170.png"
        ct_filename = os.path.basename(ct_file)  # "2171_170.png"
        
        # Extract subject_id and slice_id using regex
        import re
        match = re.search(r'(\d+)_(\d+)\.png$', ct_filename)
        if not match:
            print(f"⚠️ Could not parse filename: {ct_filename}")
            return False
            
        subject_id = match.group(1)  # "2171"
        slice_id = match.group(2)    # "170"
        
        # Create generation folder path
        generation_folder = os.path.join(data_root, subject_id, "positive", "generation")
        os.makedirs(generation_folder, exist_ok=True)
        
        # Create output filename with same naming convention
        output_filename = f"{subject_id}_{slice_id}.png"
        output_path = os.path.join(generation_folder, output_filename)
        
        # Save the generated image
        save_image_png(generated_img, output_path)
        
        print(f"✅ Saved: {subject_id}/positive/generation/{output_filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving organized generation for index {test_index}: {e}")
        return False
    
@torch.no_grad()
def test_model_wavelet(ema_model, test_dataset, test_results_dir, full_dataset, data_root, with_condition=True):
    from diffusion_model.trainer import idwt_haar_1level
    ema_model.eval()
    os.makedirs(test_results_dir, exist_ok=True)

    # convenience to unwrap Subset
    def base_and_idx(ds, i):
        return (ds.dataset, ds.indices[i]) if isinstance(ds, torch.utils.data.Subset) else (ds, i)

    saved = 0
    #for i in range(min(len(test_dataset), 50)):
    for i in range(len(test_dataset)):
        try:
            base_ds_paths, orig_idx = base_and_idx(test_dataset, i)
            ct_path, mk_path = base_ds_paths.file_paths_at(orig_idx)
            subject_id, slice_id = base_ds_paths.get_subject_slice_from_ct(ct_path)

            cond, target_coeffs = test_dataset[i]         # (mask_down2, coeffs)
            cond = cond.unsqueeze(0).cuda()               # [1,1,H/2,W/2]
            # where you build `cond`/`cond_batch` for sampling or debug
            #cond = (cond + 1.0) * 0.5 - 0.25
            target_coeffs = target_coeffs.unsqueeze(0).cuda()

            Hh = ema_model.image_size
            base_ds_std = test_dataset.dataset if isinstance(test_dataset, Subset) else test_dataset
            use_clip, _ = _clip_flags(base_ds_std)
            soft_a = torch.tensor([1.5, 12, 12, 16], device='cuda')  # [LL, LH, HL, HH]
            

            gen_coeffs = ema_model.p_sample_loop(
                shape=(1, 4, Hh, Hh),
                condition_tensors=cond,
                clip_denoised=use_clip,
                soft_a=soft_a
            )

            gen_coeffs    = _destandardize(gen_coeffs, base_ds_std)
            target_coeffs = _destandardize(target_coeffs, base_ds_std)

            # ---- LL per-image mean/std calibration (raw coeffs) ----
            with torch.no_grad():
                ll = gen_coeffs[:, 0:1, :, :]
                mu_LL  = base_ds_std.mu[0].to(ll.device)
                std_LL = base_ds_std.sigma[0].to(ll.device)

                m = ll.mean(dim=(2,3), keepdim=True)
                s = ll.std(dim=(2,3), keepdim=True) + 1e-6

                # mild, safe bounds
                gamma = (std_LL / s).clamp(0.8, 1.2)                      # contrast
                beta  = (mu_LL - m).clamp(-0.2 * std_LL, 0.2 * std_LL)    # bias

                gen_coeffs[:, 0:1, :, :] = ll * gamma + beta


            gen_img01 = (idwt_haar_1level(gen_coeffs).clamp(-1,1) + 1)*0.5
            gt_img01  = (idwt_haar_1level(target_coeffs).clamp(-1,1) + 1)*0.5

            # subject-aware save under data_root / subject / positive / generation
            subj_gen_dir = os.path.join(data_root, subject_id, "positive", "wdm_generation")
            os.makedirs(subj_gen_dir, exist_ok=True)

            save_image(gen_img01, os.path.join(subj_gen_dir, f"{subject_id}_{slice_id}.png"))
            saved += 1

            # (optional) also write to test_results_dir for metrics
            #save_image(gen_img01, os.path.join(test_results_dir, f"test_{i:03d}_generated.png"))
            #save_image(gt_img01,  os.path.join(test_results_dir, f"test_{i:03d}_target.png"))
            #save_image((cond+1)*0.5, os.path.join(test_results_dir, f"test_{i:03d}_mask.png"))

        except Exception as e:
            print(f"❌ Test sample {i} failed: {e}")

    print(f"✅ Saved {saved} generations to subject folders.")


def save_image_png(img_array, filepath, add_stats=False, is_input_label=False):
    """Save numpy array as PNG image with optional statistics"""
    # Handle tensor input
    if hasattr(img_array, 'detach'):  # Check if it's a tensor
        img_array = img_array.detach().cpu().numpy()
    
    # Handle different channel configurations
    if len(img_array.shape) > 2:
        if img_array.shape[0] == 1:  # Single channel
            img_array = img_array[0]  # Remove channel dimension
        elif img_array.shape[0] == 3:  # RGB
            img_array = np.transpose(img_array, (1, 2, 0))  # CHW to HWC
        else:
            # For other channel numbers, save as grayscale (first channel)
            img_array = img_array[0]
    
    # Print statistics if requested
    if add_stats:
        print(f"Image stats for {os.path.basename(filepath)}: min={img_array.min():.4f}, max={img_array.max():.4f}, mean={img_array.mean():.4f}")
    
    # Check if image is all zeros or very close to zero
    if np.abs(img_array).max() < 1e-6:
        print(f"WARNING: {os.path.basename(filepath)} appears to be nearly all zeros!")
    
    # Special handling for input labels (masks/segmentations)
    if is_input_label:
        # For binary masks, use simple thresholding
        if img_array.min() >= -1.1 and img_array.max() <= 1.1:  # Likely normalized to [-1,1]
            # Convert from [-1,1] to [0,1] then threshold
            img_array = (img_array + 1.0) / 2.0
            img_array = (img_array > 0.5).astype(np.float32)
        img_array = (img_array * 255).astype(np.uint8)
    else:
        # For CT scans, use standard normalization
        # Normalize to 0-255 range
        if img_array.max() > img_array.min():
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        else:
            # Handle case where all values are the same
            img_array = np.zeros_like(img_array)
        
        img_array = (img_array * 255).astype(np.uint8)
    
    # Create PIL image and save
    if len(img_array.shape) == 3:
        pil_image = Image.fromarray(img_array, mode='RGB')
    else:
        pil_image = Image.fromarray(img_array, mode='L')
    
    pil_image.save(filepath)

# Custom trainer class that includes validation and loss tracking
class CTTrainer(Trainer):
    def __init__(self, val_dataset, val_results_dir, images_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_dataset = val_dataset
        self.val_results_dir = val_results_dir
        self.images_dir = images_dir
        self.loss_history = []  # Track loss over time
        
    def plot_loss_curve(self):
        """Plot and save loss curve"""
        if not self.loss_history:
            return
            
        plt.figure(figsize=(12, 6))
        
        steps = [item['step'] for item in self.loss_history]
        losses = [item['loss'] for item in self.loss_history]
        
        plt.subplot(1, 2, 1)
        plt.plot(steps, losses, 'b-', alpha=0.7, linewidth=1)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Step')
        plt.grid(True, alpha=0.3)
        
        # Plot moving average
        if len(losses) > 50:
            window_size = min(50, len(losses) // 10)
            moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            moving_steps = steps[window_size-1:]
            plt.plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
            plt.legend()
        
        # Epoch-based plot (approximate)
        plt.subplot(1, 2, 2)
        if len(steps) > 0:
            # Estimate epochs (assuming save_and_sample_every steps per "epoch")
            epoch_steps = []
            epoch_losses = []
            step_interval = self.save_and_sample_every
            
            for i in range(0, len(steps), step_interval):
                if i < len(steps):
                    epoch_steps.append(steps[i] // step_interval)
                    # Average loss over this interval
                    start_idx = i
                    end_idx = min(i + step_interval, len(losses))
                    avg_loss = np.mean(losses[start_idx:end_idx])
                    epoch_losses.append(avg_loss)
            
            plt.plot(epoch_steps, epoch_losses, 'g-o', linewidth=2, markersize=4)
            plt.xlabel('Epoch (approx)')
            plt.ylabel('Average Loss')
            plt.title('Training Loss vs Epoch')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.images_dir, 'loss_curve.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Loss curve saved to: {plot_path}")
        
    def validate_and_save(self, milestone):
        """Generate samples from validation set and save results (wavelet -> iDWT, subject-aware)."""
        print("Generating validation samples...")

        # epoch folder
        val_milestone_dir = os.path.join(self.val_results_dir, f"epoch_{milestone * self.save_and_sample_every}")
        os.makedirs(val_milestone_dir, exist_ok=True)

        num_val_samples = min(4, len(self.val_dataset))
        for i in range(num_val_samples):
            try:
                # --- unwrap Subset to recover subject + slice filenames ---
                if isinstance(self.val_dataset, torch.utils.data.Subset):
                    base_ds = self.val_dataset.dataset
                    orig_idx = self.val_dataset.indices[i]
                else:
                    base_ds = self.val_dataset
                    orig_idx = i

                ct_path, mk_path = base_ds.file_paths_at(orig_idx)  # you added this helper
                subject_id, slice_id = base_ds.get_subject_slice_from_ct(ct_path)

                # --- fetch sample (supports tuple or dict) ---
                sample = self.val_dataset[i]
                if isinstance(sample, (tuple, list)) and len(sample) == 2:
                    cond, target_coeffs = sample               # cond:[1,H/2,W/2], target:[4,H/2,W/2]
                elif isinstance(sample, dict):
                    cond, target_coeffs = sample['input'], sample['target']
                else:
                    raise ValueError("Unexpected val sample format")

                cond = cond.unsqueeze(0).cuda().float()         # [1,1,H/2,W/2]
                # where you build `cond`/`cond_batch` for sampling or debug
                #cond = (cond + 1.0) * 0.5 - 0.25

                target_coeffs = target_coeffs.unsqueeze(0).cuda().float()  # [1,4,H/2,W/2]


                # --- sample coeffs in wavelet space (no clamp), then iDWT ---
                Hh = self.image_size  # this should be input_size//2 from your diffusion
                base_ds = self.val_dataset.dataset if isinstance(self.val_dataset, torch.utils.data.Subset) else self.val_dataset
                use_clip, _ = _clip_flags(base_ds)

                soft_a = torch.tensor([1.5, 12, 12, 16], device='cuda')  # [LL, LH, HL, HH]
                gen_coeffs = self.ema_model.p_sample_loop(
                    shape=(1, 4, Hh, Hh),
                    condition_tensors=cond,
                    clip_denoised=use_clip,
                    soft_a=soft_a
                )

                # de-standardize both, if applicable (K only if hard)
                gen_coeffs    = _destandardize(gen_coeffs, base_ds)
                target_coeffs = _destandardize(target_coeffs, base_ds)

                # ---- LL per-image mean/std calibration (raw coeffs) ----
                with torch.no_grad():
                    ll = gen_coeffs[:, 0:1, :, :]
                    mu_LL  = base_ds.mu[0].to(ll.device)
                    std_LL = base_ds.sigma[0].to(ll.device)

                    m = ll.mean(dim=(2,3), keepdim=True)
                    s = ll.std(dim=(2,3), keepdim=True) + 1e-6

                    # mild, safe bounds
                    gamma = (std_LL / s).clamp(0.8, 1.2)                      # contrast
                    beta  = (mu_LL - m).clamp(-0.2 * std_LL, 0.2 * std_LL)    # bias

                    gen_coeffs[:, 0:1, :, :] = ll * gamma + beta


                # 🚨 UNCOMMENT AND ADD THIS DEBUG CODE HERE:
                print(f"\n🚨 EMERGENCY DEBUG - STEP {self.step}:")
                print(f"TARGET coeffs (raw): range=[{target_coeffs.min():.4f}, {target_coeffs.max():.4f}], mean={target_coeffs.mean():.4f}, std={target_coeffs.std():.4f}")
                print(f"GEN coeffs (raw):    range=[{gen_coeffs.min():.4f}, {gen_coeffs.max():.4f}], mean={gen_coeffs.mean():.4f}, std={gen_coeffs.std():.4f}")

                # Check individual bands 
                for band in range(4):
                    band_names = ['LL', 'LH', 'HL', 'HH']
                    t_band = target_coeffs[0, band]
                    g_band = gen_coeffs[0, band]
                    print(f"  {band_names[band]} (raw): Target=[{t_band.min():.3f},{t_band.max():.3f}] vs Gen=[{g_band.min():.3f},{g_band.max():.3f}]")

                # Check energy in standardized space
                gen_energy = (gen_coeffs ** 2).mean()
                target_energy = (target_coeffs ** 2).mean()
                print(f"Energy (raw): Target={target_energy:.4f}, Generated={gen_energy:.4f}, Ratio={gen_energy/target_energy:.4f}")


                # Add this RIGHT in your validate_and_save method, 
                # just after gen_coeffs is generated:
                      
                subj_dir = os.path.join(val_milestone_dir, subject_id, "positive", "generation")
                os.makedirs(subj_dir, exist_ok=True)
                #base_ds = self.val_dataset.dataset if isinstance(self.val_dataset, torch.utils.data.Subset) else self.val_dataset

                # --- LL/HF ablation in standardized space ---
                mixA = gen_coeffs.clone()         # (A) gen LL + target HF
                mixA[:, 1:] = target_coeffs[:, 1:]

                mixB = target_coeffs.clone()      # (B) target LL + gen HF
                mixB[:, 1:] = gen_coeffs[:, 1:]

                # de-standardize both (respecting clip_mode)
                mode = getattr(base_ds, 'clip_mode', 'none')
                def destd(t):
                    t = t.clone()
                    if mode == 'hard':
                        K = getattr(base_ds, 'clip_k', 4.0)
                        t = t * K
                    mu  = base_ds.mu.view(1,4,1,1).to(t.device)
                    std = base_ds.sigma.view(1,4,1,1).to(t.device)
                    return t * (std + 1e-6) + mu

                #mixA_raw = destd(mixA)
                #mixB_raw = destd(mixB)

                # iDWT -> [0,1]
                mixA_img01 = (idwt_haar_1level(mixA).clamp(-1,1) + 1) * 0.5
                mixB_img01 = (idwt_haar_1level(mixB).clamp(-1,1) + 1) * 0.5

                save_image(mixA_img01, os.path.join(subj_dir, f"{subject_id}_{slice_id}_mixA_genLL_targetHF.png"))
                save_image(mixB_img01, os.path.join(subj_dir, f"{subject_id}_{slice_id}_mixB_targetLL_genHF.png"))


                gen_img  = idwt_haar_1level(gen_coeffs)         # [1,1,H,W] ~ [-1,1]
                gt_img   = idwt_haar_1level(target_coeffs)      # [1,1,H,W] ~ [-1,1]
                gen01    = (gen_img.clamp(-1, 1) + 1) * 0.5     # [0,1]
                gt01     = (gt_img.clamp(-1, 1) + 1) * 0.5
                mask01   = (cond + 1) * 0.5                     # [0,1]

                save_image(mask01, os.path.join(subj_dir, f"{subject_id}_{slice_id}_mask.png"))
                save_image(gt01,   os.path.join(subj_dir, f"{subject_id}_{slice_id}_target.png"))
                save_image(gen01,  os.path.join(subj_dir, f"{subject_id}_{slice_id}_generated.png"))

                print(f"Saved {subject_id}_{slice_id} to {subj_dir}")

            except Exception as e:
                print(f"Error generating validation sample {i}: {e}")
                continue

    def train(self):
        """Override train to include validation (tuple/dict safe, wavelet-ready)."""
        from functools import partial
        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()

        while self.step < self.train_num_steps:
            accumulated_loss = 0.0

            for i in range(self.gradient_accumulate_every):
                batch = next(self.dl)

                if self.with_condition:
                    # Accept (cond, target) or dict {'input':..., 'target':...}
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        input_tensors, target_tensors = batch
                    elif isinstance(batch, dict):
                        input_tensors, target_tensors = batch['input'], batch['target']
                    else:
                        raise ValueError("Batch must be (cond, target) or dict with 'input' and 'target'")

                    input_tensors  = input_tensors.cuda(non_blocking=True).float()  # [B,1,H/2,W/2]
                    #input_tensors = (input_tensors + 1.0) * 0.5 - 0.25
                    target_tensors = target_tensors.cuda(non_blocking=True).float() # [B,4,H/2,W/2]

                    loss = self.model(target_tensors, condition_tensors=input_tensors)  # diffusion returns mean loss
                else:
                    data = batch.cuda(non_blocking=True).float()
                    loss = self.model(data)

                if loss.ndim > 0:
                    loss = loss.mean()

                print(f'{self.step}.{i}: {loss.item():.6f}')
                self.loss_history.append({'step': self.step, 'loss': loss.item()})
                backwards(loss / self.gradient_accumulate_every, self.opt)
                accumulated_loss += loss.item()

            average_loss = accumulated_loss / float(self.gradient_accumulate_every)
            self.writer.add_scalar("training_loss", average_loss, self.step)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
            # also clip *values* to kill rare spikes that norm-clip may miss
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)
                self.validate_and_save(milestone)
                self.plot_loss_curve()

            self.step += 1

        print('training completed')
        self.plot_loss_curve()
        end_time = time.time()
        execution_time = (end_time - start_time) / 3600
        self.writer.add_hparams(
            {"lr": self.train_lr, "batchsize": self.train_batch_size, "image_size": self.image_size, "execution_time (hour)": execution_time},
            {"last_loss": average_loss}
        )
        self.writer.close()



trainer = CTTrainer(
    val_dataset,
    val_results_dir,
    images_dir,
    diffusion,
    train_dataset,                # wavelet dataset
    image_size=image_size_half,   # optional; not used by your Trainer logic
    train_batch_size=args.batchsize,
    train_lr=train_lr,
    train_num_steps=args.epochs,
    gradient_accumulate_every=gradient_accumulate_every,
    ema_decay=ema_decay,
    fp16=False,
    with_condition=with_condition,
    save_and_sample_every=save_and_sample_every,
)


# systematic_debugging.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class WaveletDebugger:
    def __init__(self, dataset, model, diffusion, save_dir="debug_outputs"):
        self.dataset = dataset                               # can be Subset
        self.base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
        self.model = model
        self.diffusion = diffusion
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
 
    def debug_pipeline_step_by_step(self):
        print("🔍 COMPREHENSIVE WAVELET DEBUGGING")
        print("=" * 60)

        # get a sample (Subset returns (cond, coeffs) just fine)
        sample = self.dataset[0]
        cond, target_coeffs = sample if isinstance(sample, (list, tuple)) else (sample['input'], sample['target'])
        print(f"Sample shapes: cond={cond.shape}, target_coeffs={target_coeffs.shape}")

        self.debug_coefficient_statistics(target_coeffs, "1_target_coeffs")
        self.debug_standardization_effects(target_coeffs)    # <-- now uses self.base_ds
        self.debug_diffusion_forward(target_coeffs, cond)
        self.debug_model_predictions(target_coeffs, cond)
        self.debug_sampling_process(cond)
        self.debug_reconstruction_quality()
     
    def debug_coefficient_statistics(self, coeffs, name):
        """Analyze coefficient band statistics in detail"""
        print(f"\n📊 COEFFICIENT STATISTICS: {name}")
        print("-" * 40)
        
        band_names = ['LL', 'LH', 'HL', 'HH']
        stats = {}
        
        for i, band_name in enumerate(band_names):
            band_data = coeffs[i].flatten()
            stats[band_name] = {
                'min': band_data.min().item(),
                'max': band_data.max().item(),
                'mean': band_data.mean().item(),
                'std': band_data.std().item(),
                'median': band_data.median().item(),
                'p01': torch.quantile(band_data, 0.01).item(),
                'p99': torch.quantile(band_data, 0.99).item(),
                'zeros': (band_data.abs() < 1e-6).sum().item() / len(band_data),
                'energy': (band_data ** 2).mean().item()
            }
            
            print(f"{band_name:3s}: range=[{stats[band_name]['min']:7.3f}, {stats[band_name]['max']:7.3f}] "
                  f"μ={stats[band_name]['mean']:7.3f} σ={stats[band_name]['std']:7.3f} "
                  f"energy={stats[band_name]['energy']:7.3f}")
        
        # Energy ratios (should be LL >> others)
        ll_energy = stats['LL']['energy']
        hf_energy = np.mean([stats[b]['energy'] for b in ['LH', 'HL', 'HH']])
        #print(f"Energy ratio LL/HF: {ll_energy/hf_energy:.2f} (should be >10)")
        
        # Save histogram
        self.plot_coefficient_histograms(coeffs, name)
        
        return stats
    
    def debug_standardization_effects(self, target_coeffs_ds):
        print(f"\n🔬 STANDARDIZATION ANALYSIS")
        print("-" * 40)

        # target_coeffs_ds is whatever the dataset returned (standardized if enabled)
        is_std = getattr(self.base_ds, 'standardize', False)

        if is_std:
            mu = self.base_ds.mu.view(4,1,1)
            sigma = self.base_ds.sigma.view(4,1,1)
            # If dataset used hard clip before standardizing (you don't), unscale by K here.
            if getattr(self.base_ds, 'clip_mode', 'none') == 'hard':
                K = getattr(self.base_ds, 'clip_k', 4.0)
                target_coeffs_ds = target_coeffs_ds * K

            # Recover RAW coeffs to inspect physics
            raw = target_coeffs_ds * (sigma + 1e-6) + mu
            self.debug_coefficient_statistics(raw, "raw (de-standardized)")

            # And this shows what the dataset actually feeds the model (standardized)
            self.debug_coefficient_statistics(
                (raw - mu) / (sigma + 1e-6),
                "standardized (as fed to model)"
            )
        else:
            # No standardization: just show raw once
            self.debug_coefficient_statistics(target_coeffs_ds, "raw (dataset)")


    def apply_clipping(self, coeffs):
        mode = getattr(self.base_ds, 'clip_mode', 'none')
        if mode == 'hard':
            K = getattr(self.base_ds, 'clip_k', 4.0)
            return torch.clamp(coeffs, -K, K) / K
        if mode == 'soft':
            a = getattr(self.base_ds, 'soft_a', 4.0)
            return torch.tanh(coeffs / a) * a
        return coeffs
    

    def debug_diffusion_forward(self, target_coeffs, cond):
        """Debug the diffusion forward process (adding noise)"""
        print(f"\n⏩ DIFFUSION FORWARD PROCESS")
        print("-" * 40)
        
        # Test different timestepsdebug_diffusion_forward
        T = self.diffusion.num_timesteps
        timesteps = _probe_ts(T, [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0])
        
        target_batch = target_coeffs.unsqueeze(0).cuda()
        cond_batch = cond.unsqueeze(0).cuda()
        
        for t_val in timesteps:
            #t = torch.tensor([t_val], device='cuda')
            t = torch.tensor([t_val], device='cuda', dtype=torch.long)
            
            # Forward diffusion
            noisy = self.diffusion.q_sample(target_batch, t)
            
            # Check statistics
            print(f"t={t_val:3d}: range=[{noisy.min():8.3f}, {noisy.max():8.3f}] "
                  f"std={noisy.std():6.3f} mean={noisy.mean():7.3f}")
            
            # Check for explosive values
            if noisy.abs().max() > 1000:
                print(f"🚨 EXPLOSIVE VALUES at t={t_val}: max={noisy.abs().max()}")
                
            # Save visualization at key timesteps
            if t_val in [0, 250, 999]:
                self.save_coefficient_visualization(noisy[0], f"forward_t{t_val}")
    
    def debug_model_predictions(self, target_coeffs, cond):
        """Debug what the model actually predicts"""
        print(f"\n🧠 MODEL PREDICTION ANALYSIS")
        print("-" * 40)
        
        self.model.eval()
        target_batch = target_coeffs.unsqueeze(0).cuda()
        cond_batch = cond.unsqueeze(0).cuda()
        
        with torch.no_grad():
            # Test at different noise levels
            T = self.diffusion.num_timesteps
            for t_val in _probe_ts(T, [0.05, 0.25, 0.50, 0.75]):
                #t = torch.tensor([t_val], device='cuda')
                t = torch.tensor([t_val], device='cuda', dtype=torch.long)
                
                # Create noisy input
                noise = torch.randn_like(target_batch)
                noisy_input = self.diffusion.q_sample(target_batch, t, noise=noise)
                
                # Model input (noisy + condition)
                if self.diffusion.with_condition:
                    # Ensure condition matches spatial size
                    if cond_batch.shape[-2:] != noisy_input.shape[-2:]:
                        cond_resized = torch.nn.functional.interpolate(
                            cond_batch, size=noisy_input.shape[-2:], mode='nearest'
                        )
                    else:
                        cond_resized = cond_batch
                    
                    model_input = torch.cat([noisy_input, cond_resized], dim=1)
                else:
                    model_input = noisy_input
                
                # Get model prediction
                predicted = self.model(model_input, t)
                
                print(f"t={t_val:3d}: pred_range=[{predicted.min():8.3f}, {predicted.max():8.3f}] "
                      f"pred_std={predicted.std():6.3f}")
                
                # Compare with ground truth noise
                if self.diffusion.loss_type in ['l1', 'l2']:
                    error = (predicted - noise).abs().mean()
                    print(f"      noise_prediction_error={error.item():.6f}")
                
                # Check for unusual predictions
                if predicted.abs().max() > 100:
                    print(f"🚨 EXTREME PREDICTIONS at t={t_val}: max={predicted.abs().max()}")
                    
                    # Analyze which bands are problematic
                    for i, band in enumerate(['LL', 'LH', 'HL', 'HH']):
                        band_max = predicted[0, i].abs().max()
                        if band_max > 100:
                            print(f"   {band} band max: {band_max}")
    
    #def debug_sampling_process(self, cond, soft_a=4.0, log_x0=True):
    def debug_sampling_process(self, cond, soft_a=None, log_x0=True):
        if soft_a is None:
            soft_a = torch.tensor([1.5, 12, 12, 16], device='cuda')
        """Debug the reverse sampling process with *soft-bound* on x0 in coeff space."""
        print(f"\n⏪ SAMPLING PROCESS ANALYSIS (soft_a={soft_a})")
        print("-" * 40)

        # Prepare condition
        cond_batch = cond.unsqueeze(0).cuda()  # [1,1,H/2,W/2]

        # Initialize random noise in standardized coeff space
        shape = (1, 4, self.diffusion.image_size, self.diffusion.image_size)
        img = torch.randn(shape, device='cuda')

        # Use soft bound, not pixel clamp
        use_clip = False  # coeff space: keep False
        # support float or per-band tensor for soft_a
        if not torch.is_tensor(soft_a):
            soft_a_t = torch.as_tensor([soft_a]*4, dtype=img.dtype, device=img.device)  # [4]
        else:
            soft_a_t = soft_a.to(img.device, dtype=img.dtype)
        soft_a_t = soft_a_t.view(1, 4, 1, 1)  # [1,4,1,1]

        # Optional: helper to peek at x0_hat (soft-bounded) for logging
        @torch.no_grad()
        def _x0_soft(x_t, t_tensor):
            if self.diffusion.with_condition:
                eps = self.model(torch.cat([x_t, cond_batch], 1), t_tensor)
            else:
                eps = self.model(x_t, t_tensor)
            x0 = self.diffusion.predict_start_from_noise(x_t, t=t_tensor, noise=eps)
            # same soft bound the sampler uses
            return self.diffusion._soft_bound_x0(x0, soft_a_t)

        #check_steps = [999, 750, 500, 250, 100, 50, 10, 0]
        T = self.diffusion.num_timesteps
        check_steps = sorted(
            _probe_ts(T, [1.0, 0.75, 0.50, 0.25, 0.10, 0.05, 0.01, 0.0]),
            reverse=True
        )

        for i in reversed(range(self.diffusion.num_timesteps)):
            t = torch.full((1,), i, device='cuda', dtype=torch.long)

            # One reverse step with *soft* constraint on x0 in coeff space
            img = self.diffusion.p_sample(
                img, t,
                condition_tensors=cond_batch,
                clip_denoised=use_clip,   # keep False in coeff space
                soft_a=soft_a_t           # <— the important part
            )

            if i in check_steps:
                print(f"Sampling t={i:3d}: x_t range=[{img.min():8.3f}, {img.max():8.3f}] std={img.std():6.3f}")

                # Peek at bounded x0_hat (helps confirm soft clamp is effective)
                if log_x0:
                    x0s = _x0_soft(img, t)
                    x0min, x0max, x0std = x0s.min().item(), x0s.max().item(), x0s.std().item()
                    print(f"             x0̂(soft) range=[{x0min:8.3f}, {x0max:8.3f}] std={x0std:6.3f}")

                # Save visualizations
                self.save_coefficient_visualization(img[0],  f"sampling_xt_t{i}")
                if log_x0:
                    self.save_coefficient_visualization(x0s[0], f"sampling_x0soft_t{i}")

        return img

    
    def debug_reconstruction_quality(self):
        print(f"\n🔄 RECONSTRUCTION QUALITY TEST")
        print("-" * 40)
        _, target_coeffs = self.dataset[0]

        # de-standardize using base dataset flags
        if getattr(self.base_ds, 'standardize', False):
            mu = self.base_ds.mu.view(4,1,1)
            sigma = self.base_ds.sigma.view(4,1,1)
            # only multiply by K if hard clip was used at dataset time
            if getattr(self.base_ds, 'clip_mode', 'none') == 'hard':
                K = getattr(self.base_ds, 'clip_k', 4.0)
                target_coeffs = target_coeffs * K
            raw_coeffs = target_coeffs * (sigma + 1e-6) + mu
        else:
            raw_coeffs = target_coeffs

        reconstructed = idwt_haar_1level(raw_coeffs.unsqueeze(0))
        print(f"Reconstruction range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
        recon_01 = (reconstructed.clamp(-1,1) + 1) * 0.5
        save_image(recon_01, self.save_dir / "test_reconstruction.png")
        print(f"✅ Test reconstruction saved to {self.save_dir / 'test_reconstruction.png'}")
    
    def plot_coefficient_histograms(self, coeffs, name):
        """Plot histograms of coefficient values"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        band_names = ['LL', 'LH', 'HL', 'HH']
        
        for i, (ax, band_name) in enumerate(zip(axes, band_names)):
            data = coeffs[i].flatten().cpu().numpy()
            ax.hist(data, bins=100, alpha=0.7, density=True)
            ax.set_title(f'{band_name} Band')
            ax.set_xlabel('Coefficient Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            ax.axvline(data.mean(), color='red', linestyle='--', label=f'μ={data.mean():.3f}')
            ax.axvline(data.mean() + data.std(), color='orange', linestyle='--', alpha=0.7)
            ax.axvline(data.mean() - data.std(), color='orange', linestyle='--', alpha=0.7)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{name}_histograms.png", dpi=150)
        plt.close()
    
    def save_coefficient_visualization(self, coeffs, name):
        """Visualize coefficient bands"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        band_names = ['LL', 'LH', 'HL', 'HH']
        
        for i, (ax, band_name) in enumerate(zip(axes, band_names)):
            band_data = coeffs[i].cpu().numpy()
            
            # Use appropriate color scale
            vmin, vmax = np.percentile(band_data, [1, 99])
            im = ax.imshow(band_data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title(f'{band_name}: [{band_data.min():.3f}, {band_data.max():.3f}]')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{name}_bands.png", dpi=150)
        plt.close()

# MAIN DEBUGGING FUNCTION TO ADD TO YOUR TRAIN.PY
def run_comprehensive_debug(dataset, model, diffusion):
    """Run comprehensive debugging before training"""
    debugger = WaveletDebugger(dataset, model, diffusion)
    debugger.debug_pipeline_step_by_step()
    
    print(f"\n📋 DEBUGGING COMPLETE!")
    print(f"Check outputs in: {debugger.save_dir}")
    print(f"Review the analysis above to identify issues.")

# Add this to your train.py BEFORE starting training:
# Uncomment the next line to run debugging
run_comprehensive_debug(full_dataset, model, diffusion)
diffusion.debug_p_sample = True

trainer.train()
# use the EMA model for all post-train analysis / sampling
# after trainer.train()
unet_ema = trainer.ema_model.denoise_fn   # <- this is the EMA UNet
run_comprehensive_debug(val_dataset, unet_ema, trainer.ema_model)


# pick which data to inspect (val is usually better than train)
#run_comprehensive_debug(val_dataset, trainer.ema_model, diffusion)
# or: run_comprehensive_debug(train_dataset, trainer.ema_model, diffusion)


if len(resume_weight) > 0 and os.path.exists(resume_weight):
    if resume_weight.endswith("ema_model_final.pth"):
        diffusion.load_state_dict(torch.load(resume_weight))
        print("✅ EMA model loaded for inference")
    else:
        ckpt = torch.load(resume_weight, map_location='cuda')
        trainer.step = ckpt['step']
        trainer.model.load_state_dict(ckpt['model'])
        trainer.ema_model.load_state_dict(ckpt['ema'])
        if 'optimizer' in ckpt:
            trainer.opt.load_state_dict(ckpt['optimizer'])  # ✅ Load optimizer state
            print("✅ Optimizer state loaded")
        print(f"✅ Training resumed from step {trainer.step}")


ema_ckpt_path = os.path.join('/storage/data/li46460_wdm_ddpm/wdm-ddpm/results', 'ema_model_final.pth')
torch.save(trainer.ema_model.state_dict(), ema_ckpt_path)
print(f"✅ EMA weights saved to {ema_ckpt_path}")

# Run test evaluation if requested
if run_test_after_training:
    print(f"\n🧪 RUNNING TEST EVALUATION...")
    test_model_wavelet(trainer.ema_model, test_dataset, test_results_dir, full_dataset, data_root, with_condition)
else:
    print(f"\n💡 To run test evaluation, use: --run_test_after_training")
    print(f"   Test indices are saved in: {os.path.join(images_dir, 'test_indices.json')}")
    print(f"   You can run testing later using the saved indices.")
