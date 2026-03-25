# NC-Conv: Noise-Conditioned Dual-Path Convolution

## Target Venues
- **ACCV 2026** (Osaka, deadline 7/5)
- **CVPR 2027** (deadline ~11/2026)

## Results Summary

| Experiment | Key Result |
|---|---|
| CIFAR-10-C (19 corruptions) | +1.6% avg, severity +0.5% to +2.9% |
| Temporal Bi-NC-SSM | f3 dark: +35.2%, clean -1.0% |
| CULane (real data) | fog +21.7%, normal +6.5% |
| Per-spatial sigma | impulse +1.0% (limitation solved) |
| Scale (1x vs 10x) | consistent improvement |

## Files
- `models.py` - All architectures (StandardCNN, NC-Conv v7/v8, Bi-NC-SSM, LaneDetector)
- `data.py` - Datasets (CIFAR-10, TunnelVideo, CULane)
- `corruption.py` - Corruption functions (19 types)
- `results.py` - Hardcoded results for reference

## Quick Start
```python
from ncconv.models import StandardCNN, make_ncconv_net, NCConvBlock, NCConvBlockSpatial
from ncconv.data import get_cifar10_loaders, TunnelVideoDataset
from ncconv.corruption import apply_corruption, random_corruption_batch
from ncconv.results import print_all_results

# Print all results
print_all_results()

# Create models
std_cnn = StandardCNN(n_classes=10)       # 251K params
nc_v7 = make_ncconv_net(NCConvBlock)      # 253K params (per-sample sigma)
nc_v8 = make_ncconv_net(NCConvBlockSpatial)  # per-spatial sigma
```

## Architecture
```
Standard CNN:  x -> DWConv -> BN -> SiLU -> PW -> residual
NC-Conv:       x -> sigma_gate -> blend(static, dynamic) -> PW -> residual
                    sigma: clean->1 (dynamic), degraded->0 (static)
```

## MCU Deployment
- 253K params = 253KB INT8
- STM32H743 ($8, 1MB SRAM, Cortex-M7 480MHz)
- 28 FPS, 35ms latency, 0.5W
