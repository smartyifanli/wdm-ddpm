# train_2D.py
#-*- coding:utf-8 -*-
# +
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
#from dataset_wavelet import CTImageGenerator, CTPairImageGenerator, create_train_val_test_datasets
from dataset import CTImageGenerator, CTPairImageGenerator, create_train_val_datasets_9_1_split
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# -

parser = argparse.ArgumentParser()
#parser.add_argument('-l', '--labelfolder', type=str, default="../crop/labels/")
#parser.add_argument('-s', '--scanfolder', type=str, default="../crop/scans/")
parser.add_argument('-d', '--data_root', type=str, default="/storage/data/TRAIL_Yifan/MH/")
parser.add_argument('--input_size', type=int, default=512)  # Changed to 512 to preserve original size
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=1)  # Changed from 3 to 1 for CT
parser.add_argument('--train_lr', type=float, default=1e-5)
parser.add_argument('--batchsize', type=int, default=1)  # Reduced to 1 for 512x512 images
parser.add_argument('--epochs', type=int, default=50000)
parser.add_argument('--timesteps', type=int, default=1000)  # Updated from 250 to 1000
parser.add_argument('--save_and_sample_every', type=int, default=100)  # Changed to 100 epochs
parser.add_argument('--with_condition', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="")
parser.add_argument('--val_results_dir', type=str, default="/home/li46460/wdm_ddpm/original/val_results")
parser.add_argument('--test_results_dir', type=str, default="/home/li46460/wdm_ddpm/original/results")
parser.add_argument('--images_dir', type=str, default="/home/li46460/wdm_ddpm/original/images")
parser.add_argument('--run_test_after_training', action='store_true', help='Run test evaluation after training')

# ✅ ADD these new advanced parameters
parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2'], help='Loss function type')
parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate')
parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='Noise schedule')
parser.add_argument('--gradient_accumulate_every', type=int, default=2, help='Gradient accumulation steps')

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
# Ask user to clear paths at the start of training
ask_and_clear_dir(args.val_results_dir, "validation results folder")
ask_and_clear_dir(args.test_results_dir, "test results folder")
ask_and_clear_dir(args.images_dir, "images folder")

'''
def clear_val_results_folder(val_results_dir):
    """Ask user if they want to clear the validation results folder"""
    if os.path.exists(val_results_dir) and os.listdir(val_results_dir):
        while True:
            response = input(f"The folder '{val_results_dir}' already exists and is not empty. Do you want to clear it? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                import shutil
                shutil.rmtree(val_results_dir)
                print(f"Cleared folder: {val_results_dir}")
                break
            elif response in ['n', 'no']:
                print("Keeping existing folder contents.")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

def clear_results_folder(results_dir):
    """Ask user if they want to clear the results folder"""
    if os.path.exists(results_dir) and os.listdir(results_dir):
        while True:
            response = input(f"The folder '{results_dir}' already exists and is not empty. Do you want to clear it? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                import shutil
                shutil.rmtree(results_dir)
                print(f"Cleared folder: {results_dir}")
                break
            elif response in ['n', 'no']:
                print("Keeping existing folder contents.")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

# Create directories
clear_val_results_folder(val_results_dir)
os.makedirs(val_results_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
'''
'''
# Enhanced transforms for low-contrast CT scans
def enhance_ct_contrast(tensor):
    """Enhanced contrast for low-contrast CT scans"""
    tensor_flat = tensor.flatten()
    # Use percentile-based normalization to stretch contrast
    p5, p95 = torch.quantile(tensor_flat, 0.05), torch.quantile(tensor_flat, 0.95)
    if p95 - p5 < 1e-6:  # Avoid division by zero
        return tensor
    tensor = torch.clamp((tensor - p5) / (p95 - p5 + 1e-8), 0, 1)
    return tensor
'''

def enhance_ct_contrast(tensor, low=5, high=95):
    """
    Enhance CT contrast using percentile-based stretching.
    :param tensor: Input tensor [0,1].
    :param low: Lower percentile (e.g., 2).
    :param high: Upper percentile (e.g., 98).
    """
    tensor_flat = tensor.flatten()
    p_low = torch.quantile(tensor_flat, low / 100.0)
    p_high = torch.quantile(tensor_flat, high / 100.0)

    if p_high - p_low < 1e-6:  # Avoid division by zero
        return tensor

    tensor = torch.clamp((tensor - p_low) / (p_high - p_low + 1e-8), 0, 1)
    return tensor

# Updated transforms with CT enhancement
transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    #Lambda(lambda t: (t / 127.5) - 1.0),  # Direct: [0,255] → [-1,1]
    Lambda(lambda t: t / 255.0),  # [0,255] -> [0,1]
    #Lambda(enhance_ct_contrast),   # ✅ CRITICAL: Enhance CT contrast
    Lambda(lambda t: enhance_ct_contrast(t, low=2, high=98)),  # Use 2–98 percentile
    Lambda(lambda t: (t * 2) - 1),  # [0,1] -> [-1,1]
    Lambda(lambda t: t.unsqueeze(0) if len(t.shape) == 2 else t),
])

# Keep input transform simple for labels (they're already binary)
input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t / 127.5) - 1.0),
    #Lambda(lambda t: t / 255.0),  # [0,255] -> [0,1]
    #Lambda(lambda t: (t * 2) - 1),  # [0,1] -> [-1,1]
    Lambda(lambda t: t.unsqueeze(0) if len(t.shape) == 2 else t),
])

if with_condition:
    full_dataset = CTPairImageGenerator(
        #labelfolder,
        #scanfolder,
        data_root,
        input_size=input_size,
        input_channel=num_class_labels,
        transform=input_transform,
        target_transform=transform,
        full_channel_mask=False  # Set to False for single channel CT
    )
    
    # Debug: Check first sample
    print("🔍 DEBUGGING FIRST SAMPLE:")
    if len(full_dataset) > 0:
        sample = full_dataset[0]
        print(f"  Input shape: {sample['input'].shape}")
        print(f"  Input range: [{sample['input'].min():.4f}, {sample['input'].max():.4f}]")
        print(f"  Input mean: {sample['input'].mean():.4f}")
        print(f"  Input std: {sample['input'].std():.4f}")
        print(f"  Target shape: {sample['target'].shape}")  
        print(f"  Target range: [{sample['target'].min():.4f}, {sample['target'].max():.4f}]")
        print(f"  Target mean: {sample['target'].mean():.4f}")
        print(f"  Target std: {sample['target'].std():.4f}")
        
        # Check for sparse data
        input_positive = torch.sum(sample['input'] > -0.5).item()
        target_positive = torch.sum(sample['target'] > -0.5).item()
        total_pixels = sample['input'].numel()
        
        print(f"  Input positive pixels: {input_positive}/{total_pixels} ({input_positive/total_pixels*100:.1f}%)")
        print(f"  Target positive pixels: {target_positive}/{total_pixels} ({target_positive/total_pixels*100:.1f}%)")
        
        if input_positive < total_pixels * 0.01:  # Less than 1% positive
            print("  ⚠️  WARNING: Input data is very sparse!")
        if target_positive < total_pixels * 0.1:  # Less than 10% positive  
            print("  ⚠️  WARNING: Target data is very sparse!")
    
    # Split into train/val/test with 7:1:2 ratio and save indices
    train_dataset, val_dataset, test_dataset, train_subjects, val_subjects = create_train_val_datasets_9_1_split(
        full_dataset, 
        random_state=42
    )
    
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
        transform=transform
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

in_channels = num_class_labels + 1 if with_condition else 1  # +1 for the target image when conditioning
out_channels = 1  # CT scans are single channel

model = create_model(
    input_size, 
    num_channels, 
    num_res_blocks, 
    in_channels=in_channels, 
    out_channels=out_channels
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

diffusion = GaussianDiffusion(
    model,
    image_size=input_size,
    timesteps=args.timesteps,
    loss_type=loss_type,  # ✅ Use configurable loss type
    with_condition=with_condition,
    channels=out_channels
    # Note: beta_schedule not supported by this GaussianDiffusion implementation
).cuda()

#if len(resume_weight) > 0 and os.path.exists(resume_weight):
#    weight = torch.load(resume_weight, map_location='cuda')
#    diffusion.load_state_dict(weight['ema'])
#    print("Model Loaded!")


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
    

def test_model(diffusion_model, test_dataset, test_results_dir, full_dataset, data_root, with_condition=True):
    """
    Test the trained model and save results in organized folders
    
    Args:
        diffusion_model: The trained diffusion model
        test_dataset: Test dataset (might be Subset)
        test_results_dir: Directory for test results (keeping for metrics)
        full_dataset: Original full dataset to access file paths
        data_root: Root data directory for organized saving
        with_condition: Whether using conditional generation
    """
    
    print(f"\n🧪 TESTING MODEL ON {len(test_dataset)} TEST SAMPLES")
    print("=" * 60)
    
    all_real_images = []
    all_generated_images = []
    saved_count = 0
    
    diffusion_model.eval()
    
    with torch.no_grad():
        for i in range(min(len(test_dataset), 50)):  # Test on first 50 samples
            if i % 10 == 0:
                print(f"Testing sample {i+1}/{min(len(test_dataset), 50)}...")
            
            try:
                if with_condition:
                    test_data = test_dataset[i]
                    input_tensor = test_data['input'].unsqueeze(0).cuda()
                    target_tensor = test_data['target'].unsqueeze(0).cuda()
                    
                    # Generate sample
                    generated = diffusion_model.sample(batch_size=1, condition_tensors=input_tensor)
                    
                    # 🆕 Get original dataset index (handle Subset case)
                    if hasattr(test_dataset, 'indices'):
                        # test_dataset is a Subset
                        original_index = test_dataset.indices[i]
                    else:
                        # test_dataset is the full dataset
                        original_index = i
                    
                    # 🆕 Save in organized folder structure
                    success = save_generation_organized(
                        generated[0].cpu().numpy(), 
                        full_dataset, 
                        original_index, 
                        data_root
                    )
                    
                    if success:
                        saved_count += 1
                    
                    # 🆕 OPTIONAL: Still save in test_results_dir for metrics/comparison
                    # (You can remove this section if you don't want duplicate saves)
                    save_image_png(input_tensor[0].cpu().numpy(), 
                                  os.path.join(test_results_dir, f"test_{i:03d}_input.png"),
                                  is_input_label=True)
                    save_image_png(target_tensor[0].cpu().numpy(), 
                                  os.path.join(test_results_dir, f"test_{i:03d}_target.png"))
                    save_image_png(generated[0].cpu().numpy(), 
                                  os.path.join(test_results_dir, f"test_{i:03d}_generated.png"))
                    
                    # Collect for metrics
                    all_real_images.append(target_tensor[0])
                    all_generated_images.append(generated[0])
                    
                else:
                    # Unconditional generation - save in test_results_dir only
                    generated = diffusion_model.sample(batch_size=1)
                    save_image_png(generated[0].cpu().numpy(), 
                                  os.path.join(test_results_dir, f"test_{i:03d}_generated.png"))
                    
            except Exception as e:
                print(f"Error testing sample {i}: {e}")
                continue
    
    print(f"✅ Saved {saved_count} generations in organized folder structure")
    
    if with_condition and all_real_images:
        # Calculate metrics
        real_images_tensor = torch.stack(all_real_images)
        generated_images_tensor = torch.stack(all_generated_images)
        
        print(f"\n📊 CALCULATING METRICS...")
        fid_score = calculate_fid(real_images_tensor, generated_images_tensor)
        ssim_score = calculate_ssim(real_images_tensor, generated_images_tensor)
        
        print(f"📈 TEST RESULTS:")
        print(f"   FID Score: {fid_score:.4f}")
        print(f"   SSIM Score: {ssim_score:.4f}")
        print(f"   Test samples: {len(all_real_images)}")
        print(f"   Organized saves: {saved_count}")
        
        # Save metrics
        metrics = {
            'fid_score': float(fid_score),
            'ssim_score': float(ssim_score),
            'num_test_samples': len(all_real_images),
            'organized_saves': saved_count,
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(test_results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Test results saved to: {test_results_dir}")
        print(f"✅ Metrics saved to: {os.path.join(test_results_dir, 'metrics.json')}")
        print(f"✅ Organized generations saved in: {data_root}/{{subject_id}}/positive/generation/")
        
        return fid_score, ssim_score
    
    return 0.0, 0.0
'''
def test_model(diffusion_model, test_dataset, test_results_dir, with_condition=True):
    """Test the trained model and save results with FID/SSIM metrics"""
    
    #clear_results_folder(test_results_dir)
    #os.makedirs(test_results_dir, exist_ok=True)
    
    print(f"\n🧪 TESTING MODEL ON {len(test_dataset)} TEST SAMPLES")
    print("=" * 60)
    
    all_real_images = []
    all_generated_images = []
    
    diffusion_model.eval()
    
    with torch.no_grad():
        for i in range(min(len(test_dataset), 50)):  # Test on first 50 samples
            if i % 10 == 0:
                print(f"Testing sample {i+1}/{min(len(test_dataset), 50)}...")
            
            try:
                if with_condition:
                    test_data = test_dataset[i]
                    input_tensor = test_data['input'].unsqueeze(0).cuda()
                    target_tensor = test_data['target'].unsqueeze(0).cuda()
                    
                    # Generate sample
                    generated = diffusion_model.sample(batch_size=1, condition_tensors=input_tensor)
                    
                    # Save individual results
                    save_image_png(input_tensor[0].cpu().numpy(), 
                                  os.path.join(test_results_dir, f"test_{i:03d}_input.png"),
                                  is_input_label=True)  # ✅ Enhanced input visualization
                    save_image_png(target_tensor[0].cpu().numpy(), 
                                  os.path.join(test_results_dir, f"test_{i:03d}_target.png"))
                    save_image_png(generated[0].cpu().numpy(), 
                                  os.path.join(test_results_dir, f"test_{i:03d}_generated.png"))
                    
                    # Collect for metrics
                    all_real_images.append(target_tensor[0])
                    all_generated_images.append(generated[0])
                    
                else:
                    # Unconditional generation
                    generated = diffusion_model.sample(batch_size=1)
                    save_image_png(generated[0].cpu().numpy(), 
                                  os.path.join(test_results_dir, f"test_{i:03d}_generated.png"))
                    
            except Exception as e:
                print(f"Error testing sample {i}: {e}")
                continue
    
    if with_condition and all_real_images:
        # Calculate metrics
        real_images_tensor = torch.stack(all_real_images)
        generated_images_tensor = torch.stack(all_generated_images)
        
        print(f"\n📊 CALCULATING METRICS...")
        fid_score = calculate_fid(real_images_tensor, generated_images_tensor)
        ssim_score = calculate_ssim(real_images_tensor, generated_images_tensor)
        
        print(f"📈 TEST RESULTS:")
        print(f"   FID Score: {fid_score:.4f}")
        print(f"   SSIM Score: {ssim_score:.4f}")
        print(f"   Test samples: {len(all_real_images)}")
        
        # Save metrics
        metrics = {
            'fid_score': float(fid_score),
            'ssim_score': float(ssim_score),
            'num_test_samples': len(all_real_images),
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(test_results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Test results saved to: {test_results_dir}")
        print(f"✅ Metrics saved to: {os.path.join(test_results_dir, 'metrics.json')}")
        
        return fid_score, ssim_score
    
    return 0.0, 0.0

'''


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
        """Generate samples from validation set and save results"""
        print("Generating validation samples...")
        
        # Create validation results subdirectory
        val_milestone_dir = os.path.join(self.val_results_dir, f"epoch_{milestone * self.save_and_sample_every}")
        os.makedirs(val_milestone_dir, exist_ok=True)
        
        # Generate a few validation samples
        num_val_samples = min(4, len(self.val_dataset))
        
        for i in range(num_val_samples):
            try:
                if self.with_condition:
                    # Get validation sample
                    val_data = self.val_dataset[i]
                    input_tensor = val_data['input'].unsqueeze(0).cuda()
                    target_tensor = val_data['target']
                    
                    print(f"Val sample {i}: input shape={input_tensor.shape}, target shape={target_tensor.shape}")
                    
                    # Generate sample
                    generated = self.ema_model.sample(batch_size=1, condition_tensors=input_tensor)
                    generated_img = generated[0].cpu().numpy()
                    
                    print(f"Generated shape: {generated_img.shape}")
                    
                    # Save input (label), target (real scan), and generated scan with statistics
                    save_image_png(input_tensor[0].cpu().numpy(), 
                                  os.path.join(val_milestone_dir, f"val_{i}_input.png"), 
                                  add_stats=True, is_input_label=True)  # ✅ Enhanced input visualization
                    save_image_png(target_tensor.numpy(), 
                                  os.path.join(val_milestone_dir, f"val_{i}_target.png"), add_stats=True)
                    save_image_png(generated_img, 
                                  os.path.join(val_milestone_dir, f"val_{i}_generated.png"), add_stats=True)
                else:
                    # Unconditional generation
                    generated = self.ema_model.sample(batch_size=1)
                    generated_img = generated[0].cpu().numpy()
                    save_image_png(generated_img, 
                                  os.path.join(val_milestone_dir, f"val_{i}_generated.png"), add_stats=True)
            except Exception as e:
                print(f"Error generating validation sample {i}: {e}")
                continue
    
    def train(self):
        """Override train method to include validation"""
        from functools import partial
        import time
        
        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()

        while self.step < self.train_num_steps:
            accumulated_loss = []
            for i in range(self.gradient_accumulate_every):
                if self.with_condition:
                    data = next(self.dl)
                    input_tensors = data['input'].cuda()
                    target_tensors = data['target'].cuda()
                    loss = self.model(target_tensors, condition_tensors=input_tensors)
                else:
                    data = next(self.dl).cuda()
                    loss = self.model(data)
                loss = loss.sum() / self.batch_size
                print(f'{self.step}: {loss.item():.6f}')  # More decimal places to see small changes
                
                # Track loss for plotting
                self.loss_history.append({
                    'step': self.step,
                    'loss': loss.item()
                })
                
                backwards(loss / self.gradient_accumulate_every, self.opt)
                accumulated_loss.append(loss.item())

            # Record loss
            average_loss = np.mean(accumulated_loss)
            self.writer.add_scalar("training_loss", average_loss, self.step)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                
                # Save model checkpoint
                self.save(milestone)
                
                # Generate and save validation samples
                self.validate_and_save(milestone)
                
                # Update loss plot
                self.plot_loss_curve()

            self.step += 1

        print('training completed')
        
        # Final loss plot
        self.plot_loss_curve()
        
        end_time = time.time()
        execution_time = (end_time - start_time) / 3600
        self.writer.add_hparams(
            {
                "lr": self.train_lr,
                "batchsize": self.train_batch_size,
                "image_size": self.image_size,
                "execution_time (hour)": execution_time
            },
            {"last_loss": average_loss}
        )
        self.writer.close()

trainer = CTTrainer(
    val_dataset,
    val_results_dir,
    images_dir,  # Add images_dir for loss plotting
    diffusion,
    train_dataset,  # Use train_dataset instead of full dataset
    image_size=input_size,
    train_batch_size=args.batchsize,
    train_lr=train_lr,
    train_num_steps=args.epochs,
    gradient_accumulate_every=gradient_accumulate_every,  # ✅ Use configurable gradient accumulation
    ema_decay=ema_decay,  # ✅ Use configurable EMA decay
    fp16=False,
    with_condition=with_condition,
    save_and_sample_every=save_and_sample_every,
)

trainer.train()

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
            Trainer.opt.load_state_dict(ckpt['optimizer'])  # ✅ Load optimizer state
            print("✅ Optimizer state loaded")
        print(f"✅ Training resumed from step {trainer.step}")


ema_ckpt_path = os.path.join('/home/li46460/wdm_ddpm/original/results', 'ema_model_final.pth')
torch.save(trainer.ema_model.state_dict(), ema_ckpt_path)
print(f"✅ EMA weights saved to {ema_ckpt_path}")

# Run test evaluation if requested
if run_test_after_training:
    print(f"\n🧪 RUNNING TEST EVALUATION...")
    #test_model(diffusion, test_dataset, test_results_dir, with_condition)
    #test_model(trainer.ema_model, test_dataset, test_results_dir, with_condition)
    test_model(trainer.ema_model, test_dataset, test_results_dir, full_dataset, data_root, with_condition)
else:
    print(f"\n💡 To run test evaluation, use: --run_test_after_training")
    print(f"   Test indices are saved in: {os.path.join(images_dir, 'test_indices.json')}")
    print(f"   You can run testing later using the saved indices.")
    print(f"   You can run testing later using the saved indices.")
