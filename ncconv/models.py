"""
NC-Conv Models: All architectures for reproducibility.
======================================================

Architecture hierarchy:
  StandardCNN          - Baseline (no NC)
  NCConvNet v7         - Per-sample sigma gate
  NCConvNet v8         - Per-spatial sigma map
  SpatialBackbone      - NC-Conv feature extractor for video
  VideoModelBiNC       - Bidirectional temporal NC-SSM + quality gate
  LaneDetector         - CULane lane detection (any backbone)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Building Blocks
# =====================================================================

class NCConvBlock(nn.Module):
    """NC-Conv v7: Per-sample sigma gate.

    sigma = sigmoid(FC(GAP(x))) in [0,1], shape (B, 1, 1, 1)
    h = sigma * h_dynamic + (1-sigma) * h_static

    Audio NC-SSM analog:
      Delta_t = sigma * Delta_sel(x) + (1-sigma) * Delta_base
    """
    def __init__(self, ch, ks=3):
        super().__init__()
        self.static_dw = nn.Conv2d(ch, ch, ks, padding=ks//2, groups=ch, bias=False)
        self.static_bn = nn.BatchNorm2d(ch)
        self.dynamic_dw = nn.Conv2d(ch, ch, ks, padding=ks//2, groups=ch, bias=False)
        self.dynamic_bn = nn.BatchNorm2d(ch)
        self.dyn_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch), nn.Sigmoid())
        self.sigma_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // 4), nn.SiLU(),
            nn.Linear(ch // 4, 1))
        self.pw = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.SiLU())

    def forward(self, x):
        sigma = torch.sigmoid(self.sigma_net(x)).unsqueeze(-1).unsqueeze(-1)
        h_s = self.static_bn(self.static_dw(x))
        h_d = self.dynamic_bn(self.dynamic_dw(x))
        h_d = h_d * self.dyn_gate(x).unsqueeze(-1).unsqueeze(-1)
        return x + self.pw(F.silu(sigma * h_d + (1 - sigma) * h_s))


class NCConvBlockSpatial(nn.Module):
    """NC-Conv v8: Per-spatial sigma map.

    sigma = sigmoid(Conv(x)) in [0,1], shape (B, 1, H, W)
    Each spatial location gets its own quality score.

    Solves impulse noise (local extreme pixels) and partial darkness.
    Audio analog: per-sub-band sigma (each frequency band has own sigma).
    """
    def __init__(self, ch, ks=3):
        super().__init__()
        self.static_dw = nn.Conv2d(ch, ch, ks, padding=ks//2, groups=ch, bias=False)
        self.static_bn = nn.BatchNorm2d(ch)
        self.dynamic_dw = nn.Conv2d(ch, ch, ks, padding=ks//2, groups=ch, bias=False)
        self.dynamic_bn = nn.BatchNorm2d(ch)
        self.dyn_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch), nn.Sigmoid())
        neck = max(ch // 4, 8)
        self.sigma_net = nn.Sequential(
            nn.Conv2d(ch, neck, 1, bias=False), nn.BatchNorm2d(neck), nn.SiLU(),
            nn.Conv2d(neck, neck, 3, padding=1, groups=neck, bias=False),
            nn.BatchNorm2d(neck), nn.SiLU(),
            nn.Conv2d(neck, 1, 1))
        nn.init.constant_(self.sigma_net[-1].bias, 2.0)  # sigmoid(2)~0.88
        self.pw = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.SiLU())

    def forward(self, x):
        sigma = torch.sigmoid(self.sigma_net(x))  # (B, 1, H, W)
        h_s = self.static_bn(self.static_dw(x))
        h_d = self.dynamic_bn(self.dynamic_dw(x))
        h_d = h_d * self.dyn_gate(x).unsqueeze(-1).unsqueeze(-1)
        return x + self.pw(F.silu(sigma * h_d + (1 - sigma) * h_s))


class StdBlock(nn.Module):
    """Standard CNN block (DWConv + PW, no NC)."""
    def __init__(self, ch, ks=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, ks, padding=ks//2, groups=ch, bias=False),
            nn.BatchNorm2d(ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.SiLU())

    def forward(self, x):
        return x + self.net(x)


# =====================================================================
# Classification Networks
# =====================================================================

class StandardCNN(nn.Module):
    """Standard CNN baseline for CIFAR-10."""
    def __init__(self, n_classes=10, c1=48, c2=96, c3=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.SiLU(),
            *[StdBlock(c1) for _ in range(3)],
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(),
            *[StdBlock(c2) for _ in range(3)],
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(c3, n_classes)

    def forward(self, x):
        return self.head(self.net(x))


def make_ncconv_net(block_class, c1=44, c2=88, c3=176, n_classes=10):
    """Factory for NC-Conv networks."""
    class NCConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.SiLU())
            self.s1 = nn.Sequential(*[block_class(c1) for _ in range(3)])
            self.down1 = nn.Sequential(
                nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())
            self.s2 = nn.Sequential(*[block_class(c2) for _ in range(3)])
            self.down2 = nn.Sequential(
                nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, n_classes))
            self.feat_dim = c3

        def forward(self, x):
            return self.head(self.down2(self.s2(self.down1(self.s1(self.stem(x))))))

        def extract(self, x):
            x = self.stem(x)
            x = self.s1(x)
            x = self.down1(x)
            x = self.s2(x)
            x = self.down2(x)
            return F.adaptive_avg_pool2d(x, 1).flatten(1)

    return NCConvNet()


# =====================================================================
# Temporal NC-SSM (Bidirectional)
# =====================================================================

class BiTemporalNCSBlock(nn.Module):
    """Bidirectional Temporal NC-SSM block.

    Forward: clean past helps degraded present
    Backward: clean future helps degraded present

    sigma_t * h_selective + (1-sigma_t) * h_fixed
    """
    def __init__(self, d_model, kernel_size=5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        neck = max(d_model // 8, 16)
        self.fwd_fixed = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model)
        self.fwd_sel = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model)
        self.fwd_gate = nn.Sequential(nn.Linear(d_model, neck), nn.SiLU(), nn.Linear(neck, d_model))
        self.bwd_fixed = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model)
        self.bwd_sel = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model)
        self.bwd_gate = nn.Sequential(nn.Linear(d_model, neck), nn.SiLU(), nn.Linear(neck, d_model))
        self.sigma_net = nn.Sequential(nn.Linear(d_model, neck), nn.SiLU(), nn.Linear(neck, 1))
        self.out = nn.Sequential(nn.Linear(d_model * 2, neck), nn.SiLU(), nn.Linear(neck, d_model))

    def _process(self, x, conv_f, conv_s, gate):
        T = x.size(1)
        xt = x.transpose(1, 2)
        h_f = conv_f(xt)[:, :, :T].transpose(1, 2)
        h_s = conv_s(xt)[:, :, :T].transpose(1, 2)
        h_s = h_s * torch.sigmoid(gate(x))
        sigma = torch.sigmoid(self.sigma_net(x))
        return sigma * h_s + (1 - sigma) * h_f

    def forward(self, x):
        r = x
        x = self.norm(x)
        h_fwd = self._process(x, self.fwd_fixed, self.fwd_sel, self.fwd_gate)
        h_bwd = self._process(x.flip(1), self.bwd_fixed, self.bwd_sel, self.bwd_gate).flip(1)
        return r + self.out(torch.cat([h_fwd, h_bwd], dim=-1))


class SpatialBackbone(nn.Module):
    """NC-Conv spatial backbone for video."""
    def __init__(self, c1=44, c2=88, c3=176):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.SiLU())
        self.s1 = nn.Sequential(*[NCConvBlock(c1) for _ in range(3)])
        self.down1 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())
        self.s2 = nn.Sequential(*[NCConvBlock(c2) for _ in range(3)])
        self.down2 = nn.Sequential(nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(c3, 10)
        self.feat_dim = c3

    def forward(self, x):
        return self.head(self.extract(x))

    def extract(self, x):
        return self.pool(self.down2(self.s2(self.down1(self.s1(self.stem(x))))))


class VideoModelBiNC(nn.Module):
    """Bidirectional NC-SSM with per-frame quality gate.

    Training: bidirectional (sees past + future)
    Deployment: causal forward-only (streaming, 1-frame latency)

    Quality gate: clean frames -> gate~0 (no correction)
                  degraded frames -> gate>0 (temporal correction)
    """
    def __init__(self, spatial_backbone, n_temporal=2):
        super().__init__()
        self.spatial = spatial_backbone
        for p in self.spatial.parameters():
            p.requires_grad = False
        self.spatial.eval()
        d = self.spatial.feat_dim
        neck = max(d // 8, 16)
        self.temporal = nn.Sequential(*[BiTemporalNCSBlock(d) for _ in range(n_temporal)])
        self.quality_gate = nn.Sequential(
            nn.Linear(d, neck), nn.SiLU(), nn.Linear(neck, 1), nn.Sigmoid())
        nn.init.constant_(self.quality_gate[2].bias, -3.0)
        self.head = nn.Linear(d, 10)

    def forward(self, video, return_details=False):
        B, T = video.shape[:2]
        with torch.no_grad():
            feats_sp = torch.stack([self.spatial.extract(video[:, t]) for t in range(T)], dim=1)
        feats_temp = self.temporal(feats_sp)
        gate = self.quality_gate(feats_sp)
        feats_out = feats_sp + gate * (feats_temp - feats_sp)
        output = self.head(feats_out.mean(dim=1))
        if return_details:
            return output, feats_sp, feats_out, gate
        return output

    def forward_per_frame(self, video):
        B, T = video.shape[:2]
        with torch.no_grad():
            feats_sp = torch.stack([self.spatial.extract(video[:, t]) for t in range(T)], dim=1)
        feats_temp = self.temporal(feats_sp)
        gate = self.quality_gate(feats_sp)
        feats_out = feats_sp + gate * (feats_temp - feats_sp)
        return torch.stack([self.head(feats_out[:, t]) for t in range(T)], dim=1), gate


# =====================================================================
# Lane Detection
# =====================================================================

class LaneBackbone(nn.Module):
    """Lane detection backbone (any block type)."""
    def __init__(self, c1, c2, c3, block_class):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c1), nn.SiLU())
        self.s1 = nn.Sequential(*[block_class(c1) for _ in range(3)])
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())
        self.s2 = nn.Sequential(*[block_class(c2) for _ in range(3)])
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())

    def forward(self, x):
        return self.down2(self.s2(self.down1(self.s1(self.stem(x)))))


class LaneHead(nn.Module):
    """Row-anchor lane detection head."""
    def __init__(self, in_ch, n_lanes=4, n_anchors=18):
        super().__init__()
        self.n_lanes, self.n_anchors = n_lanes, n_anchors
        self.pool = nn.AdaptiveAvgPool2d((n_anchors, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // 2), nn.SiLU(),
            nn.Linear(in_ch // 2, n_lanes * 2))

    def forward(self, feat):
        B = feat.size(0)
        x = self.pool(feat).squeeze(-1).permute(0, 2, 1)
        out = self.fc(x)
        return out.view(B, self.n_anchors, self.n_lanes, 2).permute(0, 2, 1, 3)


class LaneDetector(nn.Module):
    """Full lane detector: backbone + head."""
    def __init__(self, c1=48, c2=96, c3=192, block_class=StdBlock):
        super().__init__()
        self.backbone = LaneBackbone(c1, c2, c3, block_class)
        self.head = LaneHead(c3)

    def forward(self, x):
        return self.head(self.backbone(x))


class LaneLoss(nn.Module):
    """Lane detection loss: position L1 + confidence BCE + smoothness."""
    def forward(self, preds, gt_x, gt_conf):
        pred_x = torch.sigmoid(preds[..., 0])
        mask = gt_conf > 0.5
        n_pos = mask.sum().clamp(min=1)
        pos_loss = (F.l1_loss(pred_x, gt_x, reduction='none') * mask).sum() / n_pos
        conf_loss = F.binary_cross_entropy_with_logits(preds[..., 1], gt_conf)
        if pred_x.size(2) > 1:
            diff = (pred_x[:, :, 1:] - pred_x[:, :, :-1]).pow(2)
            sim_loss = (diff * mask[:, :, 1:]).sum() / mask[:, :, 1:].sum().clamp(1)
        else:
            sim_loss = 0
        return pos_loss + conf_loss + sim_loss
