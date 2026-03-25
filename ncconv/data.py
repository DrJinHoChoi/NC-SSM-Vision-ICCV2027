"""
Datasets for NC-Conv experiments.
=================================

1. CIFAR-10 + CIFAR-10-C (corruption benchmark)
2. TunnelVideoDataset (simulated tunnel passage)
3. CULaneFiles (real lane detection)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image


# =====================================================================
# CIFAR-10
# =====================================================================

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(batch_size=128, data_dir=None):
    """Get CIFAR-10 train/test loaders."""
    if data_dir is None:
        data_dir = '/content/data' if os.path.exists('/content') else './data'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(data_dir, train=False, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    return train_ds, test_ds, train_loader, test_loader


# =====================================================================
# Tunnel Video (Simulated)
# =====================================================================

class TunnelVideoDataset(Dataset):
    """Simulate tunnel passage as 8-frame video.

    Severity profile: [0, 0.3, 0.7, 1.0, 0.9, 0.8, 0.4, 0]
      f0: clean (before tunnel)
      f1: slight darkening (approaching)
      f2: severe dark (tunnel entry)
      f3: very dark (inside) -- hardest frame
      f4: very dark + noise (inside, vibration)
      f5: glare burst (exit)
      f6: recovering
      f7: clean (after tunnel)
    """
    def __init__(self, base_dataset, n_frames=8):
        self.base = base_dataset
        self.n_frames = n_frames
        self.severity_profile = [0, 0.3, 0.7, 1.0, 0.9, 0.8, 0.4, 0]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        frames = []
        for t in range(self.n_frames):
            s = self.severity_profile[t]
            if s < 0.01:
                frames.append(img)
            else:
                d = img.clone()
                d = d - s * 1.5           # darken
                d = d + torch.randn_like(d) * s * 0.3  # noise
                if t == 5:
                    d = d + s * 2.0        # exit glare
                frames.append(d)
        return torch.stack(frames), label


# =====================================================================
# CULane
# =====================================================================

class CULaneFiles(Dataset):
    """CULane lane detection dataset (file list based).

    Loads images + .lines.txt lane annotations.
    Row-anchor format: 4 lanes x 18 anchors x (x_pos, confidence).
    """
    def __init__(self, file_list, img_size=(288, 512), n_lanes=4, n_anchors=18):
        self.files = file_list
        self.n_lanes, self.n_anchors = n_lanes, n_anchors
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.anchor_ys = np.linspace(0.44, 1.0, n_anchors)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        jpg = self.files[idx]
        try:
            img = Image.open(jpg).convert('RGB')
        except Exception:
            img = Image.new('RGB', (1640, 590))
        img = self.transform(img)

        gt_x = np.zeros((self.n_lanes, self.n_anchors), dtype=np.float32)
        gt_conf = np.zeros((self.n_lanes, self.n_anchors), dtype=np.float32)
        try:
            with open(jpg.replace('.jpg', '.lines.txt')) as f:
                for li, line in enumerate(f.readlines()[:self.n_lanes]):
                    coords = list(map(float, line.strip().split()))
                    if len(coords) < 4:
                        continue
                    xs, ys = coords[0::2], coords[1::2]
                    for ai, ay in enumerate(self.anchor_ys):
                        ty = ay * 590
                        best_d, best_x = 999, -1
                        for xi, yi in zip(xs, ys):
                            d = abs(yi - ty)
                            if d < best_d and xi > 0:
                                best_d, best_x = d, xi
                        if best_x > 0 and best_d < 20:
                            gt_x[li, ai] = best_x / 1640
                            gt_conf[li, ai] = 1.0
        except Exception:
            pass
        return img, torch.tensor(gt_x), torch.tensor(gt_conf)


def get_culane_files(culane_root, driver='driver_37_30frame'):
    """Find valid CULane image files with annotations."""
    import subprocess
    driver_path = os.path.join(culane_root, driver)
    result = subprocess.run(['find', driver_path, '-name', '*.jpg'],
                            capture_output=True, text=True)
    all_jpgs = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
    valid = [j for j in all_jpgs if os.path.exists(j.replace('.jpg', '.lines.txt'))]
    np.random.seed(42)
    np.random.shuffle(valid)
    split = int(len(valid) * 0.8)
    return valid[:split], valid[split:]
