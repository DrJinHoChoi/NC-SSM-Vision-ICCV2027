"""
NC-Conv: Noise-Conditioned Dual-Path Convolution
=================================================
Target: ACCV 2026 / CVPR 2027

Modules:
  models.py       - All architectures (StandardCNN, NC-Conv v7/v8, Bi-NC-SSM)
  data.py         - Datasets (CIFAR-10, CULane, TunnelVideo)
  corruption.py   - Corruption functions (19 types + custom)
  train.py        - Training loops (single-frame, temporal, 2-phase)
  evaluate.py     - Evaluation (CIFAR-10-C, per-condition, per-frame)
  experiments.py  - Full reproducible experiments with hardcoded results

Quick start:
  python -m ncconv.experiments --results     # print all results
  python -m ncconv.experiments --cifar10c    # run CIFAR-10-C benchmark
  python -m ncconv.experiments --temporal    # run temporal Bi-NC-SSM
  python -m ncconv.experiments --culane      # run CULane lane detection
"""
