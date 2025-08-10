# dataset_2D.py
#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
from utils.dtypes import LabelEnum
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import cv2
import re
import os
from pathlib import Path

class CTPairImageGenerator(Dataset):
    def __init__(self,
            data_root: str,  # Now points to /storage/data/TRAIL_Yifan/MH
            input_size: int,
            input_channel: int = 1,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.data_root = Path(data_root)  # /storage/data/TRAIL_Yifan/MH
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        """🔄 NEW: Find all CT-mask pairs in the new structure"""
        pairs = []
        
        # Find all subject folders (numeric names like 2171, 2373, etc.)
        subject_folders = [d for d in self.data_root.iterdir() 
                          if d.is_dir() and d.name.isdigit()]
        
        print(f"📁 Found {len(subject_folders)} subject folders")
        
        for subject_folder in subject_folders:
            subject_id = subject_folder.name
            
            # Paths to ct and mask folders
            ct_folder = subject_folder / "positive" / "ct"
            mask_folder = subject_folder / "positive" / "mask"
            
            if not ct_folder.exists() or not mask_folder.exists():
                print(f"⚠️  Skipping {subject_id}: missing ct or mask folder")
                continue
            
            # Find all CT files for this subject
            ct_files = list(ct_folder.glob(f"{subject_id}_*.png"))
            
            for ct_file in ct_files:
                # Extract slice number from filename
                # Format: {subject_id}_{slice}.png (e.g., 1950_170.png)
                match = re.search(rf'{subject_id}_(\d+)\.png$', ct_file.name)
                if match:
                    slice_num = match.group(1)
                    
                    # Find corresponding mask file
                    mask_file = mask_folder / f"{subject_id}_{slice_num}.png"
                    
                    if mask_file.exists():
                        pairs.append((str(mask_file), str(ct_file)))
                    else:
                        print(f"⚠️  Missing mask for {ct_file.name}")
        
        print(f"✅ Found {len(pairs)} valid CT-mask pairs")
        
        # Print sample of found pairs for verification
        if pairs:
            print(f"📋 Sample pairs:")
            for i, (mask_path, ct_path) in enumerate(pairs[:3]):
                print(f"   {i+1}. Mask: {Path(mask_path).name}")
                print(f"      CT:   {Path(ct_path).name}")
        
        return pairs

    def get_subject_from_path(self, file_path):
        """Extract subject ID from file path"""
        path = Path(file_path)
        # Extract subject ID from filename (e.g., 1950_170.png -> 1950)
        match = re.search(r'(\d+)_\d+\.png$', path.name)
        return match.group(1) if match else None

    def label2masks(self, masked_img):
        """Convert label image to multi-channel format"""
        result_img = np.zeros(masked_img.shape + (self.input_channel - 1,)) if self.input_channel > 1 else masked_img[..., np.newaxis]
        
        if self.input_channel > 1:
            # You can modify these based on your mask label values
            result_img[masked_img == LabelEnum.BRAINAREA.value, 0] = 1
            if self.input_channel > 2:
                result_img[masked_img == LabelEnum.TUMORAREA.value, 1] = 1
        
        return result_img

    def read_image(self, file_path, is_label=False):
        """Read 2D image from PNG file"""
        if is_label:
            img = Image.open(file_path).convert('L')
            img = np.array(img, dtype=np.float32)
        else:
            img = Image.open(file_path).convert('L')
            img = np.array(img, dtype=np.float32)
        return img

    def resize_img(self, img):
        """Resize image to target size"""
        h, w = img.shape[:2]
        if h != self.input_size or w != self.input_size:
            if len(img.shape) == 3:
                img = cv2.resize(img, (self.input_size, self.input_size))
            else:
                img = cv2.resize(img, (self.input_size, self.input_size))
        return img

    def sample_conditions(self, batch_size: int):
        """Sample conditions for generation"""
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]  # mask files
        input_tensors = []
        
        for input_file in input_files:
            input_img = self.read_image(input_file, is_label=True)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img)
            
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
        
        return torch.cat(input_tensors, 0).cuda()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        mask_file, ct_file = self.pair_files[index]
        
        input_img = self.read_image(mask_file, is_label=True)
        input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
        input_img = self.resize_img(input_img)

        target_img = self.read_image(ct_file, is_label=False)
        target_img = self.resize_img(target_img)

        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input': input_img, 'target': target_img}


# 🔄 NEW: Updated dataset split function for subject-based splitting
def create_train_val_datasets_9_1_split(dataset, random_state=42):
    """
    Split dataset into train:validation = 9:1, test = whole dataset
    
    Args:
        dataset: CTPairImageGenerator dataset
        random_state: random seed for reproducibility
    """
    # Group indices by subject
    subject_indices = {}
    
    for i, (mask_file, ct_file) in enumerate(dataset.pair_files):
        # Extract subject ID from filename
        subject_id = dataset.get_subject_from_path(ct_file)
        
        if subject_id:
            if subject_id not in subject_indices:
                subject_indices[subject_id] = []
            subject_indices[subject_id].append(i)
    
    # Get all subjects for train/val split
    all_subjects = list(subject_indices.keys())
    
    # Split subjects into train:val = 9:1
    np.random.seed(random_state)
    train_subjects, val_subjects = train_test_split(
        all_subjects, 
        train_size=0.9,  # 90% subjects for training
        random_state=random_state
    )
    
    # Collect indices for train and val
    train_indices = []
    val_indices = []
    all_indices = list(range(len(dataset)))  # All indices for test
    
    for subject_id, indices in subject_indices.items():
        if subject_id in train_subjects:
            train_indices.extend(indices)
        elif subject_id in val_subjects:
            val_indices.extend(indices)
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = dataset  # Whole dataset as test set
    
    # Print detailed information
    print(f"\n📊 DATASET SPLIT (TRAIN:VAL = 9:1, TEST = ALL):")
    print(f"Total subjects: {len(subject_indices)}")
    print(f"Train subjects: {len(train_subjects)} ({train_subjects[:5]}{'...' if len(train_subjects) > 5 else ''})")
    print(f"Val subjects: {len(val_subjects)} ({val_subjects})")
    print(f"Test: All subjects (whole dataset)")
    
    # Count slices per split
    train_subject_counts = {subj: len(subject_indices[subj]) for subj in train_subjects}
    val_subject_counts = {subj: len(subject_indices[subj]) for subj in val_subjects}
    
    print(f"📊 Slice distribution:")
    print(f"   Train: {len(train_dataset)} slices from {len(train_subjects)} subjects")
    print(f"   Val: {len(val_dataset)} slices from {len(val_subjects)} subjects")
    print(f"   Test: {len(test_dataset)} slices from {len(subject_indices)} subjects (whole dataset)")
    
    return train_dataset, val_dataset, test_dataset, train_subjects, val_subjects


# For unconditional generation (if needed)
class CTImageGenerator(Dataset):
    def __init__(self, data_root, input_size, transform=None):
        self.data_root = Path(data_root)
        self.input_size = input_size
        self.inputfiles = self.find_ct_files()
        self.scaler = MinMaxScaler()
        self.transform = transform

    def find_ct_files(self):
        """Find all CT files in the new structure"""
        ct_files = []
        
        subject_folders = [d for d in self.data_root.iterdir() 
                          if d.is_dir() and d.name.isdigit()]
        
        for subject_folder in subject_folders:
            ct_folder = subject_folder / "positive" / "ct"
            if ct_folder.exists():
                ct_files.extend(list(ct_folder.glob("*.png")))
        
        print(f"Found {len(ct_files)} CT files for unconditional training")
        return [str(f) for f in ct_files]

    def read_image(self, file_path):
        img = Image.open(file_path).convert('L')
        img = np.array(img, dtype=np.float32)
        return img

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        img = self.read_image(inputfile)
        h, w = img.shape
        if h != self.input_size or w != self.input_size:
            img = cv2.resize(img, (self.input_size, self.input_size))

        if self.transform is not None:
            img = self.transform(img)
        return img
