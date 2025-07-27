# sample_2D.py
#-*- coding:utf-8 -*-
from diffusion_model.trainer import GaussianDiffusion, num_to_groups
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from torchvision.transforms import Compose, Lambda
from PIL import Image
import numpy as np
import argparse
import torch
import os
import glob
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', type=str, default="../crop/labels/")
parser.add_argument('-e', '--exportfolder', type=str, default="exports/")
parser.add_argument('--input_size', type=int, default=512)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=1)  # Changed from 3 to 1 for CT
parser.add_argument('--timesteps', type=int, default=1000)
parser.add_argument('-w', '--weightfile', type=str, default="results/model-final.pt")
args = parser.parse_args()

exportfolder = args.exportfolder
inputfolder = args.inputfolder
input_size = args.input_size
batchsize = args.batchsize
weightfile = args.weightfile
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_samples = args.num_samples
in_channels = args.num_class_labels + 1  # +1 for conditioning (mask + target)
out_channels = 1  # CT scans are single channel
device = "cuda"

# Find PNG mask files (change from .nii.gz to .png)
mask_list = sorted(glob.glob(f"{inputfolder}/*_label.png"))
print(f"Found {len(mask_list)} mask files")

def read_image(file_path, is_label=False):
    """Read 2D image from PNG file"""
    img = Image.open(file_path).convert('L')  # Convert to grayscale
    img = np.array(img, dtype=np.float32)
    # Don't normalize here - will be done in transform
    return img

def resize_img_2d(img, target_size):
    """Resize 2D image to target size"""
    h, w = img.shape[:2]
    if h != target_size or w != target_size:
        img = cv2.resize(img, (target_size, target_size))
    return img

def label2masks(masked_img):
    """Convert label image to multi-channel format (simplified for CT)"""
    # For CT, typically binary masks (background=0, foreground=255)
    # Convert to single channel format
    result_img = np.zeros(masked_img.shape + (in_channels - 1,))
    
    # Assuming binary masks: background (0) and foreground (255)
    if in_channels > 1:
        result_img[masked_img > 127, 0] = 1  # Threshold at 127 for binary
    
    return result_img if in_channels > 1 else masked_img[..., np.newaxis]

# 2D transforms (simplified from 3D)
input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t / 127.5) - 1.0),  # Direct: [0,255] → [-1,1]
    Lambda(lambda t: t.permute(2, 0, 1) if len(t.shape) == 3 else t.unsqueeze(0)),  # Add channel dim
    Lambda(lambda t: t.unsqueeze(0))  # Add batch dimension
])

model = create_model(
    input_size, 
    num_channels, 
    num_res_blocks, 
    in_channels=in_channels, 
    out_channels=out_channels
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size=input_size,
    timesteps=args.timesteps,
    loss_type='l2',  # Changed from 'L1' to 'l2' 
    with_condition=True,
    channels=out_channels
).cuda()

# Load model weights
if os.path.exists(weightfile):
    weight_data = torch.load(weightfile, map_location='cuda')
    diffusion.load_state_dict(weight_data['ema'])
    print("Model Loaded!")
else:
    print(f"Weight file not found: {weightfile}")
    exit(1)

def save_image_png(img_array, filepath, is_input_label=False):
    """Save numpy array as PNG image"""
    # Handle tensor input
    if hasattr(img_array, 'detach'):
        img_array = img_array.detach().cpu().numpy()
    
    # Handle different channel configurations
    if len(img_array.shape) > 2:
        if img_array.shape[0] == 1:  # Single channel
            img_array = img_array[0]  # Remove channel dimension
        else:
            # For other channel numbers, save as grayscale (first channel)
            img_array = img_array[0]
    
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
        if img_array.max() > img_array.min():
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        else:
            img_array = np.zeros_like(img_array)
        img_array = (img_array * 255).astype(np.uint8)
    
    # Create PIL image and save
    pil_image = Image.fromarray(img_array, mode='L')
    pil_image.save(filepath)

# Create output directories
img_dir = os.path.join(exportfolder, "image")   
msk_dir = os.path.join(exportfolder, "mask")   
os.makedirs(img_dir, exist_ok=True)
os.makedirs(msk_dir, exist_ok=True)

print(f"Starting generation for {len(mask_list)} masks...")
print(f"Output directories:")
print(f"  Generated images: {img_dir}")
print(f"  Original masks: {msk_dir}")

for k, inputfile in enumerate(mask_list):
    left = len(mask_list) - (k+1)
    print(f"Processing {k+1}/{len(mask_list)} | LEFT: {left}")
    
    # Extract filename for saving
    msk_name = os.path.basename(inputfile)
    base_name = msk_name.replace('_label.png', '')
    
    # Read and process mask
    refImg = read_image(inputfile, is_label=True)
    original_shape = refImg.shape
    
    # Resize mask to model input size
    refImg_resized = resize_img_2d(refImg, input_size)
    
    # Convert labels to mask format
    img = label2masks(refImg_resized)
    
    # Apply transforms
    input_tensor = input_transform(img)
    
    # Prepare batching
    batches = num_to_groups(num_samples, batchsize)
    steps = len(batches)
    sample_count = 0
    
    print(f"  Generating {num_samples} samples in {steps} batches")
    counter = 0
    
    for i, bsize in enumerate(batches):
        print(f"  Batch [{i+1}/{steps}]")
        condition_tensors = []
        counted_samples = []
        
        for b in range(bsize):
            condition_tensors.append(input_tensor)
            counted_samples.append(sample_count)
            sample_count += 1

        condition_tensors = torch.cat(condition_tensors, 0).cuda()
        
        # Generate samples
        with torch.no_grad():
            all_images_list = list(map(
                lambda n: diffusion.sample(batch_size=n, condition_tensors=condition_tensors), 
                [bsize]
            ))
            all_images = torch.cat(all_images_list, dim=0)
        
        # Save generated images
        for b, c in enumerate(counted_samples):
            counter += 1
            sample_image = all_images[b].cpu()
            
            # Save generated image
            generated_filename = f'{counter}_{base_name}_generated.png'
            save_image_png(sample_image, os.path.join(img_dir, generated_filename))
            
            # Save original mask for reference
            mask_filename = f'{counter}_{base_name}_mask.png'
            save_image_png(refImg_resized, os.path.join(msk_dir, mask_filename), is_input_label=True)
            
            print(f"    Saved: {generated_filename}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    print(f"  Completed: {msk_name}")

print("Generation completed!")
print(f"Generated images saved to: {img_dir}")
print(f"Reference masks saved to: {msk_dir}")