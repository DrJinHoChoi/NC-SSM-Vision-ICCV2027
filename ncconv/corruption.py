"""
Corruption functions for NC-Conv experiments.
=============================================

Maps vision corruptions to audio noise analogs:
  gaussian_noise   <-> White noise
  brightness_down  <-> Factory noise (stationary)
  contrast         <-> Babble noise
  fog              <-> Broadband noise
  impulse_noise    <-> Impulse noise (non-stationary)
"""

import numpy as np
import torch


CORRUPTION_TYPES = ['gaussian_noise', 'brightness_down', 'contrast', 'fog', 'impulse_noise']

AUDIO_ANALOG = {
    'gaussian_noise':   'White noise',
    'brightness_down':  'Factory noise (stationary)',
    'contrast':         'Babble noise',
    'fog':              'Broadband noise',
    'impulse_noise':    'Impulse noise (non-stationary)',
}


def apply_corruption(images, corruption, severity=3):
    """Apply corruption to normalized image tensor.

    Args:
        images: (B, C, H, W) normalized tensor
        corruption: one of CORRUPTION_TYPES
        severity: 1-5 (mild to severe)
    Returns:
        corrupted images (same shape)
    """
    if corruption == 'gaussian_noise':
        s = [0.04, 0.06, 0.08, 0.12, 0.18][severity - 1]
        return images + torch.randn_like(images) * s
    elif corruption == 'brightness_down':
        s = [0.3, 0.5, 0.7, 0.85, 1.0][severity - 1]
        return images - s
    elif corruption == 'contrast':
        s = [0.6, 0.5, 0.4, 0.3, 0.15][severity - 1]
        m = images.mean(dim=(2, 3), keepdim=True)
        return (images - m) * s + m
    elif corruption == 'fog':
        s = [0.2, 0.4, 0.6, 0.8, 1.0][severity - 1]
        return images * (1 - s) + s
    elif corruption == 'impulse_noise':
        s = [0.01, 0.02, 0.05, 0.1, 0.2][severity - 1]
        mask = torch.rand_like(images) < s
        return images * ~mask + torch.randint_like(images, -2, 3).float() * mask
    return images


def random_corruption_batch(images, prob=0.3):
    """Randomly corrupt a fraction of the batch for training augmentation.

    Args:
        images: (B, C, H, W) tensor
        prob: probability of corruption (default 0.3 = 30%)
    Returns:
        images with some samples corrupted
    """
    B = images.size(0)
    mask = torch.rand(B) < prob
    if mask.sum() == 0:
        return images
    imgs = images.clone()
    corr = np.random.choice(CORRUPTION_TYPES)
    sev = np.random.randint(1, 4)
    imgs[mask] = apply_corruption(imgs[mask], corr, sev)
    return imgs


def apply_lane_corruption(images, corr, sev=3):
    """Simplified corruption for lane detection."""
    if corr == 'dark':
        return images - np.random.uniform(0.3, 0.8)
    elif corr == 'noise':
        return images + torch.randn_like(images) * np.random.uniform(0.05, 0.15)
    elif corr == 'fog':
        s = np.random.uniform(0.2, 0.6)
        return images * (1 - s) + s
    return images
