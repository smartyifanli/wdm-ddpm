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


class CTImageGenerator(Dataset):
    def __init__(self, imagefolder, input_size, transform=None):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.inputfiles = glob(os.path.join(imagefolder, '*.png'))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path):
        img = Image.open(file_path).convert('L')  # Convert to grayscale
        img = np.array(img, dtype=np.float32)
        # Normalize to 0-1 range
        #img = img / 255.0
        return img

    def plot_samples(self, n_row=4):
        samples = [self[index] for index in np.random.randint(0, len(self), n_row*n_row)]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row*i+j]
                sample = sample[0] if len(sample.shape) > 2 else sample
                plt.subplot(n_row, n_row, n_row*i+j+1)
                plt.imshow(sample, cmap='gray')
        plt.show()

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

class CTPairImageGenerator(Dataset):
    def __init__(self,
            label_folder: str,
            scan_folder: str,
            input_size: int,
            input_channel: int = 1,  # Changed from 3 to 1 for CT
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.label_folder = label_folder
        self.scan_folder = scan_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        label_files = sorted(glob(os.path.join(self.label_folder, '*_label.png')))
        pairs = []
        for label_file in label_files:
            # Extract subject ID and slice number from filename
            # Format: subject_1767_slice_188_label.png
            basename = os.path.basename(label_file)
            # Replace '_label.png' with '_scan.png' to get corresponding scan file
            scan_basename = basename.replace('_label.png', '_scan.png')
            scan_file = os.path.join(self.scan_folder, scan_basename)
            
            if os.path.exists(scan_file):
                pairs.append((label_file, scan_file))
            else:
                print(f"Warning: No corresponding scan file found for {label_file}")
        
        print(f"Found {len(pairs)} valid pairs")
        return pairs

    def label2masks(self, masked_img):
        # For CT, we typically have fewer label classes than MRI
        # Adjust based on your specific CT segmentation labels
        result_img = np.zeros(masked_img.shape + (self.input_channel - 1,)) if self.input_channel > 1 else masked_img[..., np.newaxis]
        
        if self.input_channel > 1:
            # You can modify these based on your CT label values
            result_img[masked_img == LabelEnum.BRAINAREA.value, 0] = 1
            if self.input_channel > 2:
                result_img[masked_img == LabelEnum.TUMORAREA.value, 1] = 1
        
        return result_img

    def read_image(self, file_path, is_label=False):
        if is_label:
            # Read labels as grayscale and keep original values
            img = Image.open(file_path).convert('L')
            img = np.array(img, dtype=np.float32)
            # Don't normalize labels, keep original label values
        else:
            # Read scans as grayscale and normalize
            img = Image.open(file_path).convert('L')
            img = np.array(img, dtype=np.float32)
            # Normalize to 0-1 range
            #img = img / 255.0
        return img

    def plot(self, index):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        if input_img.shape[0] == 1:
            plt.imshow(input_img[0], cmap='gray')
        else:
            plt.imshow(input_img[0], cmap='gray')  # Show first channel
        plt.title('Input (Label)')
        
        plt.subplot(1, 2, 2)
        if target_img.shape[0] == 1:
            plt.imshow(target_img[0], cmap='gray')
        else:
            plt.imshow(target_img[0], cmap='gray')
        plt.title('Target (Scan)')
        plt.show()

    def resize_img(self, img):
        h, w = img.shape[:2]
        if h != self.input_size or w != self.input_size:
            if len(img.shape) == 3:  # Multi-channel
                img = cv2.resize(img, (self.input_size, self.input_size))
            else:  # Single channel
                img = cv2.resize(img, (self.input_size, self.input_size))
        return img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
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
        input_file, target_file = self.pair_files[index]
        
        input_img = self.read_image(input_file, is_label=True)
        input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
        input_img = self.resize_img(input_img)

        target_img = self.read_image(target_file, is_label=False)
        target_img = self.resize_img(target_img)

        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input': input_img, 'target': target_img}


def create_train_val_test_datasets(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
    """
    Split dataset into train/val/test with 7:1:2 ratio
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    indices = list(range(total_size))
    
    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=random_state
    )
    
    # Second split: separate train and val from remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_ratio_adjusted, random_state=random_state
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset