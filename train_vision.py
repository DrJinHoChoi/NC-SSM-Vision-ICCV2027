#!/usr/bin/env python3
# coding=utf-8
"""
NC-SSM Vision Training Pipeline
================================

Two modes:
  1. Synthetic: quick pipeline validation (no download needed)
  2. CULane:    real lane detection benchmark

Usage:
  python train_vision.py --mode synthetic --epochs 5        # quick test
  python train_vision.py --mode culane --data_dir ./data    # real training
  python train_vision.py --mode culane --eval_only          # eval only

Target: ICCV 2027 -- Tunnel / Rapid Illumination Environments
"""

import os
import sys
import json
import time
import math
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

from ncssm_vision import NanoMambaVision
from ncssm_vision_tasks import (
    NanoMambaLaneDetector, NanoMambaCriticalDetector,
    NanoMambaMultiTaskDetector,
    LaneDetectionLoss, CriticalObjectLoss,
    LaneHead, CriticalObjectHead,
    create_lane_detector_nano, create_lane_detector_tiny,
    create_lane_detector_small, create_lane_detector_medium,
    create_critical_detector_nano, create_critical_detector_tiny,
    create_critical_detector_small,
    create_multitask_detector_tiny,
)


# ============================================================================
# Synthetic Lane Dataset (for pipeline validation)
# ============================================================================

class SyntheticLaneDataset(Dataset):
    """Synthetic lane dataset for quick pipeline validation.

    Generates images with synthetic lanes and visibility conditions:
      - Normal: clear lanes on gray background
      - Dark:   reduced brightness (tunnel simulation)
      - Glare:  bright spots (tunnel exit simulation)
      - Fog:    reduced contrast (fog simulation)

    Audio analog: training with factory/babble/white noise augmentation.
    """

    def __init__(self, n_samples=1000, img_size=288, n_lanes=4, n_anchors=18,
                 condition='mixed', split='train'):
        self.n_samples = n_samples
        self.img_size = img_size
        self.n_lanes = n_lanes
        self.n_anchors = n_anchors
        self.condition = condition
        self.split = split
        self.anchor_ys = np.linspace(0.4, 1.0, n_anchors)

        # Pre-generate lanes for consistency
        np.random.seed(42 if split == 'train' else 123)
        self.lane_params = []
        for _ in range(n_samples):
            # Random lane parameters: x_center + curvature
            n_active = np.random.randint(2, n_lanes + 1)
            x_centers = np.sort(np.random.uniform(0.2, 0.8, n_active))
            curvatures = np.random.uniform(-0.3, 0.3, n_active)
            self.lane_params.append((n_active, x_centers, curvatures))

    def _render_image(self, idx):
        """Render synthetic lane image."""
        H = W = self.img_size
        n_active, x_centers, curvatures = self.lane_params[idx]

        # Background (gray road)
        img = np.ones((3, H, W), dtype=np.float32) * 0.3

        # Draw lanes as white lines
        gt_x = np.zeros((self.n_lanes, self.n_anchors), dtype=np.float32)
        gt_conf = np.zeros((self.n_lanes, self.n_anchors), dtype=np.float32)

        for i in range(n_active):
            if i >= self.n_lanes:
                break
            xc = x_centers[i]
            curv = curvatures[i]

            for j, y_norm in enumerate(self.anchor_ys):
                # Quadratic lane: x = xc + curv * (y - 0.7)^2
                x_norm = xc + curv * (y_norm - 0.7) ** 2
                x_norm = np.clip(x_norm, 0.01, 0.99)

                gt_x[i, j] = x_norm
                gt_conf[i, j] = 1.0

                # Draw on image
                px = int(x_norm * W)
                py = int(y_norm * H)
                # Lane line (white, 3px wide)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        xx = np.clip(px + dx, 0, W - 1)
                        yy = np.clip(py + dy, 0, H - 1)
                        img[:, yy, xx] = 0.9  # white lane

        # Add some texture
        noise = np.random.randn(3, H, W).astype(np.float32) * 0.02
        img = np.clip(img + noise, 0, 1)

        return img, gt_x, gt_conf

    def _apply_condition(self, img, idx):
        """Apply adverse visibility condition.

        Analogous to audio noise augmentation:
          Audio: mix clean audio with factory/babble/white noise at -15 to 15 dB
          Vision: apply dark/glare/fog degradation at random severity
        """
        if self.condition == 'normal':
            return img

        rng = np.random.RandomState(idx * 7 + 13)

        if self.condition == 'mixed':
            cond = rng.choice(['normal', 'dark', 'glare', 'fog', 'shadow'])
        else:
            cond = self.condition

        severity = rng.uniform(0.3, 0.8)

        if cond == 'dark':
            # Tunnel darkness: reduce brightness uniformly
            # Analogous to factory noise (stationary, broadband)
            img = img * (1.0 - severity * 0.8)

        elif cond == 'glare':
            # Tunnel exit glare: bright spot in upper area
            # Analogous to impulse noise (non-stationary)
            H, W = img.shape[1], img.shape[2]
            cy, cx = int(H * 0.3), int(W * 0.5)
            Y, X = np.mgrid[:H, :W]
            dist = ((Y - cy) ** 2 + (X - cx) ** 2) / (H * W * 0.1)
            glare = np.exp(-dist) * severity
            img = np.clip(img + glare[np.newaxis], 0, 1)

        elif cond == 'fog':
            # Fog: reduce contrast, add uniform brightness
            # Analogous to white noise (stationary, uniform)
            fog_level = severity * 0.6
            img = img * (1 - fog_level) + fog_level

        elif cond == 'shadow':
            # Partial shadow: darken left or right half
            # Analogous to narrowband noise (frequency-specific)
            H, W = img.shape[1], img.shape[2]
            if rng.random() > 0.5:
                img[:, :, :W // 2] *= (1.0 - severity * 0.7)
            else:
                img[:, :, W // 2:] *= (1.0 - severity * 0.7)

        return np.clip(img, 0, 1).astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img, gt_x, gt_conf = self._render_image(idx)

        # Always apply condition (including val split for per-condition eval)
        img = self._apply_condition(img, idx)

        return (
            torch.from_numpy(img),
            torch.from_numpy(gt_x),
            torch.from_numpy(gt_conf),
        )


# ============================================================================
# Synthetic Detection Dataset (for Critical Object Spotting)
# ============================================================================

class SyntheticDetectionDataset(Dataset):
    """Synthetic detection dataset for CenterNet-style Critical Object Spotting.

    Generates images with random objects (rectangles/circles) and produces
    CenterNet GT: heatmap, offset, size, mask.

    Audio analog: synthetic keyword injection at random SNR levels.
    """

    CLASSES = ['pedestrian', 'vehicle', 'traffic_sign',
               'traffic_light', 'construction_barrier']

    def __init__(self, n_samples=1000, img_size=288, n_classes=5,
                 patch_size=16, max_objects=5, condition='mixed', split='train'):
        self.n_samples = n_samples
        self.img_size = img_size
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.out_h = img_size // patch_size  # heatmap spatial size
        self.out_w = img_size // patch_size
        self.max_objects = max_objects
        self.condition = condition
        self.split = split

        np.random.seed(42 if split == 'train' else 123)
        self.object_params = []
        for _ in range(n_samples):
            n_obj = np.random.randint(1, max_objects + 1)
            objs = []
            for _ in range(n_obj):
                cls = np.random.randint(0, n_classes)
                cx = np.random.uniform(0.1, 0.9)
                cy = np.random.uniform(0.1, 0.9)
                w = np.random.uniform(0.05, 0.25)
                h = np.random.uniform(0.05, 0.30)
                objs.append((cls, cx, cy, w, h))
            self.object_params.append(objs)

    def _gaussian2d(self, shape, sigma=1.0):
        """Generate 2D Gaussian for heatmap."""
        m, n = [(s - 1.) / 2. for s in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < 1e-7] = 0
        return h

    def _draw_gaussian(self, heatmap, center, radius, k=1):
        """Draw Gaussian on heatmap at center location."""
        diameter = 2 * radius + 1
        gaussian = self._gaussian2d((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])
        H, W = heatmap.shape

        left = min(x, radius)
        right = min(W - x, radius + 1)
        top = min(y, radius)
        bottom = min(H - y, radius + 1)

        if left + right <= 0 or top + bottom <= 0:
            return

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom,
                                   radius - left:radius + right]
        if masked_heatmap.shape == masked_gaussian.shape:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    def _apply_condition(self, img, idx):
        """Same condition logic as lane dataset."""
        if self.condition == 'normal':
            return img

        rng = np.random.RandomState(idx * 7 + 13)
        if self.condition == 'mixed':
            cond = rng.choice(['normal', 'dark', 'glare', 'fog', 'shadow'])
        else:
            cond = self.condition

        severity = rng.uniform(0.3, 0.8)

        if cond == 'dark':
            img = img * (1.0 - severity * 0.8)
        elif cond == 'glare':
            H, W = img.shape[1], img.shape[2]
            cy, cx = int(H * 0.3), int(W * 0.5)
            Y, X = np.mgrid[:H, :W]
            dist = ((Y - cy) ** 2 + (X - cx) ** 2) / (H * W * 0.1)
            glare = np.exp(-dist) * severity
            img = np.clip(img + glare[np.newaxis], 0, 1)
        elif cond == 'fog':
            fog_level = severity * 0.6
            img = img * (1 - fog_level) + fog_level
        elif cond == 'shadow':
            H, W = img.shape[1], img.shape[2]
            if rng.random() > 0.5:
                img[:, :, :W // 2] *= (1.0 - severity * 0.7)
            else:
                img[:, :, W // 2:] *= (1.0 - severity * 0.7)

        return np.clip(img, 0, 1).astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        H = W = self.img_size
        oH, oW = self.out_h, self.out_w
        objs = self.object_params[idx]

        # Background image
        img = np.ones((3, H, W), dtype=np.float32) * 0.3
        noise = np.random.randn(3, H, W).astype(np.float32) * 0.02
        img = np.clip(img + noise, 0, 1)

        # Draw objects and build GT
        gt_heatmap = np.zeros((self.n_classes, oH, oW), dtype=np.float32)
        gt_offset = np.zeros((2, oH, oW), dtype=np.float32)
        gt_size = np.zeros((2, oH, oW), dtype=np.float32)
        gt_mask = np.zeros((1, oH, oW), dtype=np.float32)

        for cls, cx, cy, w, h in objs:
            # Draw rectangle on image
            x1_px = int((cx - w / 2) * W)
            y1_px = int((cy - h / 2) * H)
            x2_px = int((cx + w / 2) * W)
            y2_px = int((cy + h / 2) * H)
            x1_px = np.clip(x1_px, 0, W - 1)
            y1_px = np.clip(y1_px, 0, H - 1)
            x2_px = np.clip(x2_px, 0, W - 1)
            y2_px = np.clip(y2_px, 0, H - 1)

            # Color by class
            colors = [[0.8, 0.2, 0.2], [0.2, 0.2, 0.8], [0.8, 0.8, 0.2],
                      [0.2, 0.8, 0.2], [0.8, 0.5, 0.2]]
            color = colors[cls % len(colors)]
            for c in range(3):
                img[c, y1_px:y2_px, x1_px:x2_px] = color[c]

            # CenterNet GT: center in output (heatmap) coordinates
            cx_out = cx * oW
            cy_out = cy * oH
            cx_int = int(cx_out)
            cy_int = int(cy_out)

            if 0 <= cx_int < oW and 0 <= cy_int < oH:
                # Heatmap: Gaussian at center
                radius = max(1, int(min(w * oW, h * oH) / 2))
                self._draw_gaussian(gt_heatmap[cls], (cx_int, cy_int), radius)

                # Offset: sub-pixel center refinement
                gt_offset[0, cy_int, cx_int] = cx_out - cx_int
                gt_offset[1, cy_int, cx_int] = cy_out - cy_int

                # Size: log-space (w, h) in output coordinates
                gt_size[0, cy_int, cx_int] = np.log(max(w * oW, 1e-4))
                gt_size[1, cy_int, cx_int] = np.log(max(h * oH, 1e-4))

                # Mask: positive location
                gt_mask[0, cy_int, cx_int] = 1.0

        # Apply condition
        img = self._apply_condition(img, idx)

        return (
            torch.from_numpy(img),
            torch.from_numpy(gt_heatmap),
            torch.from_numpy(gt_offset),
            torch.from_numpy(gt_size),
            torch.from_numpy(gt_mask),
        )


# ============================================================================
# CULane Dataset
# ============================================================================

class CULaneDataset(Dataset):
    """CULane dataset loader.

    CULane: 133,235 images, 88,880 train / 9,675 val / 34,680 test
    Categories: Normal, Crowded, Dazzle, Shadow, No line, Arrow, Curve, Night, Cross

    Download: https://xingangpan.github.io/projects/CULane.html
    Structure:
      data_dir/
        driver_23_30frame/...
        driver_161_90frame/...
        laneseg_label_w16/...
        list/
          train_gt.txt
          val_gt.txt
          test.txt
    """

    def __init__(self, data_dir, split='train', img_size=288, n_lanes=4, n_anchors=18):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.n_lanes = n_lanes
        self.n_anchors = n_anchors
        self.anchor_ys = np.linspace(0.4, 1.0, n_anchors)

        # Load file list
        list_file = self.data_dir / 'list' / f'{split}_gt.txt'
        if not list_file.exists():
            list_file = self.data_dir / 'list' / f'{split}.txt'

        if not list_file.exists():
            raise FileNotFoundError(
                f"CULane list file not found: {list_file}\n"
                f"Download CULane from: https://xingangpan.github.io/projects/CULane.html\n"
                f"Expected structure: {data_dir}/list/train_gt.txt")

        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    img_path = parts[0].lstrip('/')
                    self.samples.append(img_path)

        print(f"  CULane {split}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.data_dir / self.samples[idx]

        # Load image
        try:
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            img = np.array(img, dtype=np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC -> CHW
        except Exception as e:
            # Fallback: random image
            img = np.random.rand(3, self.img_size, self.img_size).astype(np.float32)

        # Load lane annotations
        anno_path = img_path.with_suffix('.lines.txt')
        gt_x = np.zeros((self.n_lanes, len(self.anchor_ys)), dtype=np.float32)
        gt_conf = np.zeros((self.n_lanes, len(self.anchor_ys)), dtype=np.float32)

        if anno_path.exists():
            try:
                with open(anno_path, 'r') as f:
                    for lane_idx, line in enumerate(f):
                        if lane_idx >= self.n_lanes:
                            break
                        coords = list(map(float, line.strip().split()))
                        # CULane format: x1 y1 x2 y2 ...
                        points = [(coords[i], coords[i + 1])
                                  for i in range(0, len(coords) - 1, 2)]
                        if len(points) < 2:
                            continue
                        # Interpolate to anchor positions
                        for j, y_target in enumerate(self.anchor_ys):
                            y_abs = y_target * 590  # CULane original height
                            # Find closest points
                            for k in range(len(points) - 1):
                                x1, y1 = points[k]
                                x2, y2 = points[k + 1]
                                if (y1 <= y_abs <= y2) or (y2 <= y_abs <= y1):
                                    if abs(y2 - y1) > 1e-6:
                                        t = (y_abs - y1) / (y2 - y1)
                                        x_interp = x1 + t * (x2 - x1)
                                        gt_x[lane_idx, j] = x_interp / 1640  # normalize
                                        gt_conf[lane_idx, j] = 1.0
                                    break
            except Exception:
                pass

        return (
            torch.from_numpy(img),
            torch.from_numpy(gt_x),
            torch.from_numpy(gt_conf),
        )


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch,
                    task='lane'):
    """Train one epoch. Mirrors audio KWS training loop."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch[0].to(device)
        optimizer.zero_grad()

        preds = model(images)

        if task == 'detection':
            # Detection: batch = (img, gt_heatmap, gt_offset, gt_size, gt_mask)
            gt_heatmap = batch[1].to(device)
            gt_offset = batch[2].to(device)
            gt_size = batch[3].to(device)
            gt_mask = batch[4].to(device)
            heatmap, offset, size = preds
            loss = criterion(heatmap, offset, size,
                             gt_heatmap, gt_offset, gt_size, gt_mask)
        elif task == 'multitask':
            gt_x = batch[1].to(device)
            gt_conf = batch[2].to(device)
            lane_loss = criterion(preds['lanes'], gt_x, gt_conf)
            loss = lane_loss  # TODO: add det loss when det GT available
        else:
            # Lane: batch = (img, gt_x, gt_conf)
            gt_x = batch[1].to(device)
            gt_conf = batch[2].to(device)
            loss = criterion(preds, gt_x, gt_conf)

        # NaN safety (from audio training experience)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [WARN] NaN/Inf loss at batch {batch_idx}, skipping")
            continue

        loss.backward()

        # Gradient clipping (essential for SSM stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"loss={loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, task='lane'):
    """Evaluate model. Returns avg loss and accuracy metric."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_points = 0
    n_batches = 0

    for batch in dataloader:
        images = batch[0].to(device)
        preds = model(images)

        if task == 'detection':
            gt_heatmap = batch[1].to(device)
            gt_offset = batch[2].to(device)
            gt_size = batch[3].to(device)
            gt_mask = batch[4].to(device)
            heatmap, offset, size = preds
            loss = criterion(heatmap, offset, size,
                             gt_heatmap, gt_offset, gt_size, gt_mask)
            total_loss += loss.item()
            n_batches += 1

            # Detection accuracy: heatmap peak recall
            pred_scores = heatmap.sigmoid()
            pos_mask = gt_mask.squeeze(1)  # (B, H, W)
            for c in range(gt_heatmap.size(1)):
                gt_peaks = gt_heatmap[:, c] > 0.9  # (B, H, W)
                if gt_peaks.sum() > 0:
                    pred_peak = pred_scores[:, c] > 0.3
                    correct = (pred_peak & gt_peaks).sum()
                    total_correct += correct.item()
                    total_points += gt_peaks.sum().item()
        else:
            # Lane
            if isinstance(preds, dict):
                preds_lane = preds['lanes']
            else:
                preds_lane = preds

            gt_x = batch[1].to(device)
            gt_conf = batch[2].to(device)
            loss = criterion(preds_lane, gt_x, gt_conf)
            total_loss += loss.item()
            n_batches += 1

            pred_x = torch.sigmoid(preds_lane[..., 0])
            pred_conf = torch.sigmoid(preds_lane[..., 1])
            mask = gt_conf > 0.5

            if mask.sum() > 0:
                x_diff = (pred_x - gt_x).abs()
                correct = ((x_diff < 0.1) & (pred_conf > 0.3) & mask).sum()
                total_correct += correct.item()
                total_points += mask.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_points, 1) * 100

    return avg_loss, accuracy


def evaluate_conditions(model, img_size, n_anchors, device, task='lane'):
    """Evaluate under each adverse condition separately.

    Mirrors audio noise evaluation:
      Audio: evaluate at each noise type x SNR level
      Vision: evaluate at each condition x severity level
    """
    conditions = ['normal', 'dark', 'glare', 'fog', 'shadow']
    results = {}

    for cond in conditions:
        if task == 'detection':
            dataset = SyntheticDetectionDataset(
                n_samples=200, img_size=img_size,
                condition=cond, split='val')
            criterion = CriticalObjectLoss()
        else:
            dataset = SyntheticLaneDataset(
                n_samples=200, img_size=img_size, n_anchors=n_anchors,
                condition=cond, split='val')
            criterion = LaneDetectionLoss()

        loader = DataLoader(dataset, batch_size=32, shuffle=False,
                            num_workers=0)

        loss, acc = evaluate(model, loader, criterion, device, task=task)
        results[cond] = {'loss': loss, 'accuracy': acc}
        print(f"  {cond:<10}: loss={loss:.4f}, acc={acc:.1f}%")

    return results


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    """EMA of model parameters. Same as audio NanoMamba training."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='NC-SSM Vision Training')
    parser.add_argument('--mode', type=str, default='synthetic',
                        choices=['synthetic', 'culane'],
                        help='Dataset mode')
    parser.add_argument('--data_dir', type=str, default='./data/culane',
                        help='CULane data directory')
    parser.add_argument('--task', type=str, default='lane',
                        choices=['lane', 'detection', 'multitask'],
                        help='Task type')
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['nano', 'tiny'],
                        help='Model size variant')
    parser.add_argument('--img_size', type=int, default=288)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate (same as audio NanoMamba)')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_vision')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"  NC-SSM Vision Training")
    print(f"  Mode: {args.mode} | Task: {args.task} | Size: {args.model_size}")
    print(f"  Device: {device} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"{'='*70}\n")

    # ---- Dataset ----
    n_anchors = 18
    if args.mode == 'synthetic':
        if args.task == 'detection':
            train_dataset = SyntheticDetectionDataset(
                n_samples=2000, img_size=args.img_size,
                condition='mixed', split='train')
            val_dataset = SyntheticDetectionDataset(
                n_samples=400, img_size=args.img_size,
                condition='normal', split='val')
        else:
            train_dataset = SyntheticLaneDataset(
                n_samples=2000, img_size=args.img_size, n_anchors=n_anchors,
                condition='mixed', split='train')
            val_dataset = SyntheticLaneDataset(
                n_samples=400, img_size=args.img_size, n_anchors=n_anchors,
                condition='normal', split='val')
        print(f"  Synthetic data: {len(train_dataset)} train, "
              f"{len(val_dataset)} val")
    elif args.mode == 'culane':
        train_dataset = CULaneDataset(
            args.data_dir, split='train', img_size=args.img_size,
            n_anchors=n_anchors)
        val_dataset = CULaneDataset(
            args.data_dir, split='val', img_size=args.img_size,
            n_anchors=n_anchors)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == 'cuda'))

    # ---- Model ----
    if args.task == 'lane':
        if args.model_size == 'nano':
            model = create_lane_detector_nano(img_size=args.img_size)
        else:
            model = create_lane_detector_tiny(img_size=args.img_size)
        criterion = LaneDetectionLoss(sim_weight=1.0)
    elif args.task == 'detection':
        if args.model_size == 'nano':
            model = create_critical_detector_nano(img_size=args.img_size)
        else:
            model = create_critical_detector_tiny(img_size=args.img_size)
        criterion = CriticalObjectLoss()
    elif args.task == 'multitask':
        model = create_multitask_detector_tiny(img_size=args.img_size)
        criterion = LaneDetectionLoss(sim_weight=1.0)

    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    backbone_p = sum(p.numel() for p in model.backbone.parameters())
    print(f"\n  Model: {args.task}-{args.model_size}")
    print(f"  Total params:    {params:>8,}")
    print(f"  Backbone params: {backbone_p:>8,}")
    print(f"  Head params:     {params - backbone_p:>8,}")
    print(f"  INT8 size:       {params / 1024:>7.1f} KB")

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Loaded checkpoint: {args.checkpoint}")

    # ---- Eval only ----
    if args.eval_only:
        print("\n--- Evaluation ---\n")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device,
                                     task=args.task)
        print(f"  Val loss: {val_loss:.4f}, Val acc: {val_acc:.1f}%")
        if args.mode == 'synthetic':
            print("\n--- Per-condition evaluation ---\n")
            evaluate_conditions(model, args.img_size, n_anchors, device,
                                task=args.task)
        return

    # ---- Optimizer (same config as audio NanoMamba) ----
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=0.01)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # EMA
    ema = EMA(model, decay=0.999)

    # ---- Training ----
    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0
    history = []

    print(f"\n--- Training Start ---\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            task=args.task)

        # EMA update
        ema.update()

        # Evaluate with EMA weights
        ema.apply_shadow()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device,
                                     task=args.task)
        ema.restore()

        scheduler.step()

        dt = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch:>3}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.1f}% | "
              f"lr={lr_now:.2e} | "
              f"time={dt:.1f}s")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr_now,
        })

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            ema.apply_shadow()
            save_path = Path(args.save_dir) / f'{args.task}_{args.model_size}_best.pt'
            torch.save(model.state_dict(), save_path)
            ema.restore()
            print(f"  ** New best: {val_acc:.1f}% -> saved {save_path}")

    # ---- Final evaluation ----
    print(f"\n{'='*70}")
    print(f"  Training complete. Best val accuracy: {best_acc:.1f}%")
    print(f"{'='*70}")

    # Load best model and evaluate per-condition
    best_path = Path(args.save_dir) / f'{args.task}_{args.model_size}_best.pt'
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    if args.mode == 'synthetic':
        print("\n--- Per-Condition Evaluation (best model) ---\n")
        cond_results = evaluate_conditions(
            model, args.img_size, n_anchors, device, task=args.task)

        # Print summary table (mirrors audio noise robustness table)
        print(f"\n--- Summary (analogous to Audio Noise Table) ---\n")
        print(f"  {'Condition':<12} | {'Accuracy':>10} | Audio Analog")
        print(f"  {'-'*55}")
        analogs = {
            'normal': 'Clean audio',
            'dark': 'Factory noise (stationary)',
            'glare': 'Impulse noise (non-stationary)',
            'fog': 'White noise (broadband)',
            'shadow': 'Narrowband noise (partial)',
        }
        for cond, res in cond_results.items():
            analog = analogs.get(cond, '')
            print(f"  {cond:<12} | {res['accuracy']:>8.1f}% | {analog}")

    # Save history
    history_path = Path(args.save_dir) / f'{args.task}_{args.model_size}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n  History saved: {history_path}")


if __name__ == '__main__':
    main()
