# dataset_2D.py
#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, ToTensor, Lambda
import torch.nn.functional as F
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

# ---------- CT enhancement (2–98 percentile contrast) ----------
def enhance_ct_contrast(img01: torch.Tensor, low=2, high=98):
    """
    img01: [H,W] or [1,H,W] in [0,1] torch tensor
    returns: same shape in [0,1]
    """
    x = img01.clone()
    if x.ndim == 3: x = x[0]
    arr = x.flatten().cpu().numpy()
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    if hi <= lo:  # degenerate
        return img01
    x = (x - lo) / (hi - lo)
    x = torch.clamp(x, 0.0, 1.0)
    return x.unsqueeze(0) if img01.ndim == 3 else x

# ---------- 1-level decimated HAAR DWT (torch only) ----------
# 2x2 filters with stride 2, normalized so inverse exists nicely.
_H = 0.5  # = 1/2, orthonormal scaling for 2x2 kernels

# shape: [out_ch, in_ch, kH, kW]
_HAAR_K = torch.tensor([
    [[[_H, _H], [_H, _H]]],   # LL
    [[[_H, _H], [-_H, -_H]]], # LH (vertical detail)
    [[[ _H, -_H], [_H, -_H]]],# HL (horizontal detail)
    [[[ _H, -_H], [-_H,  _H]]]# HH (diagonal detail)
], dtype=torch.float32)

def dwt_haar_1level(x: torch.Tensor) -> torch.Tensor:
    """
    x: [1,H,W] in [-1,1]  ->  w: [4,H/2,W/2]  (even H,W assumed)
    """
    if x.ndim != 3 or x.size(0) != 1:
        raise ValueError("dwt_haar_1level expects [1,H,W]")
    H, W = x.shape[-2:]
    if (H % 2) or (W % 2):
        # reflect-pad to even
        pad_h = (0, 1 if W % 2 else 0, 0, 1 if H % 2 else 0)  # (left,right,top,bottom)
        x = F.pad(x.unsqueeze(0), pad_h, mode='reflect').squeeze(0)
    k = _HAAR_K.to(x.device)
    y = F.conv2d(x.unsqueeze(0), k, stride=2)  # [1,4,H/2,W/2]
    return y.squeeze(0)

class Wavelet2DDataset(torch.utils.data.Dataset):
    """
    Expects directory layout:
      data_root/
        <subject_id>/
          positive/
            ct/*.png
            mask/*.png
    """
    def __init__(self, 
                data_root,
                input_size=512,
                with_condition=True,
                standardize=False,     # per-band μ/σ in wavelet space
                band_stats=None,       # dict with 'mu':Tensor[4], 'sigma':Tensor[4]
                enhance_low=2,
                enhance_high=98,
                clip_mode='none',      # 'none' | 'hard' | 'soft'
                clip_k=6.0,            # only if clip_mode=='hard'
                soft_a=4.0):           # only if clip_mode=='soft'
        self.data_root = Path(data_root)
        self.input_size = int(input_size)
        self.with_condition = with_condition
        self.standardize = standardize
        self.enhance_low = enhance_low
        self.enhance_high = enhance_high

        self.clip_mode = str(clip_mode)
        self.clip_k = float(clip_k)
        self.soft_a = float(soft_a)

        if self.input_size % 2 != 0:
            raise ValueError("input_size must be even for 1-level DWT.")

        if self.standardize:
            assert band_stats is not None and 'mu' in band_stats and 'sigma' in band_stats
            self.mu = torch.as_tensor(band_stats['mu'], dtype=torch.float32)      # [4]
            self.sigma = torch.as_tensor(band_stats['sigma'], dtype=torch.float32) # [4]

        # collect paired files (ct, mask)
        self.pair_files = []  # list of (ct_path, mask_path)
        subjects = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.isdigit()]
        for subj in sorted(subjects):
            ct_dir = subj / "positive" / "ct"
            mk_dir = subj / "positive" / "mask"
            if not (ct_dir.exists() and mk_dir.exists()):
                continue
            for ct_path in sorted(ct_dir.glob("*.png")):
                mk_path = mk_dir / ct_path.name
                if mk_path.exists():
                    self.pair_files.append((str(ct_path), str(mk_path)))


    def __len__(self):
        return len(self.pair_files)

    @staticmethod
    def _read_gray_uint8(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def get_subject_from_path(self, file_path: str):
        # e.g., ".../2171/positive/ct/2171_170.png" -> "2171"
        fname = Path(file_path).name
        m = re.match(r'(\d+)_\d+\.png$', fname)
        return m.group(1) if m else None

    def get_subject_slice_from_ct(self, ct_path: str):
        # -> ("2171", "170")
        fname = Path(ct_path).name
        m = re.match(r'(\d+)_(\d+)\.png$', fname)
        return (m.group(1), m.group(2)) if m else (None, None)

    def file_paths_at(self, idx: int):
        # Keep your tuple order consistent with how you built pair_files
        ct_path, mk_path = self.pair_files[idx]
        return ct_path, mk_path


    def _load_mask_as_condition(self, mk_path: str):
        # read -> resize(H,W) -> downsample (H/2,W/2) -> [-1,1]
        mk = self._read_gray_uint8(mk_path)
        if (mk.shape[1], mk.shape[0]) != (self.input_size, self.input_size):
            mk = cv2.resize(mk, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        mk = torch.from_numpy(mk).float()
        if mk.max() > 1:
            mk = (mk > 127.5).float()  # ensure binary
        mk = F.interpolate(mk[None, None], scale_factor=0.5, mode='nearest').squeeze(0)  # [1,H/2,W/2]
        mk = mk * 2.0 - 1.0
        return mk  # [1, H/2, W/2]

    @torch.no_grad()
    def sample_conditions(self, batch_size: int = 1, indices=None):
        """
        Return a batch of downsampled mask conditions in [-1,1]
        Shape: [B, 1, H/2, W/2]
        """
        if indices is None:
            #idxs = np.random.choice(len(self.pair_files), size=batch_size, replace=False)
            replace = batch_size > len(self.pair_files)
            idxs = np.random.choice(len(self.pair_files), size=batch_size, replace=replace)
        else:
            idxs = indices

        conds = []
        for i in idxs:
            ct_path, mk_path = self.pair_files[int(i)]  # NOTE: your pair_files = (ct_path, mk_path)
            conds.append(self._load_mask_as_condition(mk_path))
        return torch.stack(conds, dim=0)  # [B,1,H/2,W/2]


    def __getitem__(self, idx):
        ct_path, mk_path = self.pair_files[idx]

        # --- CT: read -> resize -> [0,1] -> contrast -> [-1,1] -> [1,H,W] ---
        ct = self._read_gray_uint8(ct_path)
        if (ct.shape[1], ct.shape[0]) != (self.input_size, self.input_size):
            ct = cv2.resize(ct, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        ct = torch.from_numpy(ct).float() / 255.0
        ct = enhance_ct_contrast(ct, low=self.enhance_low, high=self.enhance_high)
        ct = ct * 2.0 - 1.0
        if ct.ndim == 2:
            ct = ct.unsqueeze(0)  # [1,H,W]

        # --- DWT -> [4,H/2,W/2] ---
        w = dwt_haar_1level(ct)  # [4, H/2, W/2]

        # --- per-band standardization (if enabled) ---
        if self.standardize:
            w = (w - self.mu.view(4,1,1)) / (self.sigma.view(4,1,1) + 1e-6)

        # --- optional bounding in standardized space ---
        if self.clip_mode == 'none':
            pass
        elif self.clip_mode == 'hard':
            K = self.clip_k
            w = torch.clamp(w, -K, K) / K     # maps to [-1,1]
        elif self.clip_mode == 'soft':
            a = self.soft_a
            w = torch.tanh(w / a) * a         # no divide by K here
        else:
            raise ValueError(f"Unknown clip_mode: {self.clip_mode}")

        # --- condition mask (downsampled to H/2,W/2, in [-1,1]) ---
        # --- condition mask (downsampled to H/2,W/2, in [-1,1]) ---
        if self.with_condition:
            mk = self._read_gray_uint8(mk_path)
            if (mk.shape[1], mk.shape[0]) != (self.input_size, self.input_size):
                mk = cv2.resize(mk, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            mk = torch.from_numpy(mk).float()
            # binarize, same rule as _load_mask_as_condition
            if mk.max() > 1:
                mk = (mk > 127.5).float()
            # now to [-1,1]
            mk = mk * 2.0 - 1.0
            if mk.ndim == 2:
                mk = mk.unsqueeze(0)  # [1,H,W]
            h2, w2 = w.shape[-2], w.shape[-1]
            cond = F.interpolate(mk.unsqueeze(0), size=(h2, w2), mode='nearest').squeeze(0)
            return cond, w


        # unconditional
        return w


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
def create_train_val_datasets_9_1_split_wavelet(dataset, random_state=42):
    """
    Subject-aware 9:1 split for Wavelet2DDataset.
    Returns: train_subset, val_subset, all_subjects_train, all_subjects_val
    """
    # group indices by subject (using CT path)
    subject_indices = {}
    for i, (ct_path, mk_path) in enumerate(dataset.pair_files):
        sid = dataset.get_subject_from_path(ct_path)
        if sid is None: continue
        subject_indices.setdefault(sid, []).append(i)

    all_subjects = list(subject_indices.keys())
    np.random.seed(random_state)
    train_subjects, val_subjects = train_test_split(all_subjects, train_size=0.9, random_state=random_state)

    train_idx, val_idx = [], []
    for sid, idxs in subject_indices.items():
        (train_idx if sid in train_subjects else val_idx).extend(idxs)

    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)

    print(f"\n📊 DATASET SPLIT (WAVELET, subject-aware 9:1)")
    print(f"Subjects total: {len(all_subjects)} | Train: {len(train_subjects)} | Val: {len(val_subjects)}")
    print(f"Slices -> Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    return train_dataset, val_dataset, train_subjects, val_subjects


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
