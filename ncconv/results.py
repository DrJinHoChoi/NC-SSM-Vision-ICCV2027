"""
NC-Conv: Hardcoded experiment results for reference.
=====================================================

All results are reproducible using the experiment scripts.
"""

RESULTS = {
    # =========================================================
    # Experiment 1: CIFAR-10-C Official Benchmark
    # =========================================================
    'cifar10c': {
        'description': 'CIFAR-10-C (19 corruptions x 5 severities, Hendrycks & Dietterich 2019)',
        'std_cnn_aug': {
            'clean': 89.2, 'c10c_avg': 76.2,
            'severity': {1: 84.8, 2: 81.5, 3: 78.2, 4: 72.8, 5: 63.6},
            'params': 251434,
        },
        'ncconv_aug': {
            'clean': 88.9, 'c10c_avg': 77.7,
            'severity': {1: 85.3, 2: 82.4, 3: 79.6, 4: 74.9, 5: 66.6},
            'params': 253258,
            'per_corruption': {
                'brightness': 87.0, 'contrast': 71.7, 'defocus_blur': 77.3,
                'elastic_transform': 77.5, 'fog': 80.4, 'frost': 80.0,
                'gaussian_blur': 71.7, 'gaussian_noise': 80.1, 'glass_blur': 64.5,
                'impulse_noise': 81.7, 'jpeg_compression': 78.8, 'motion_blur': 71.9,
                'pixelate': 72.4, 'saturate': 83.1, 'shot_noise': 82.3,
                'snow': 77.4, 'spatter': 82.5, 'speckle_noise': 82.5, 'zoom_blur': 74.3,
            },
        },
        'key_finding': 'Severity scaling: gap grows +0.5% (s1) to +2.9% (s5)',
    },

    # =========================================================
    # Experiment 2: Temporal Bi-NC-SSM (Tunnel Video)
    # =========================================================
    'temporal': {
        'description': 'Tunnel video simulation (8 frames, bidirectional NC-SSM + quality gate)',
        'frame_labels': ['clean', 'approach', 'entry', 'inside', 'inside+n', 'exit', 'recover', 'clean'],
        'nc_conv_per_frame': [89.0, 85.1, 68.9, 52.6, 58.6, 72.9, 81.7, 89.0],
        'bi_ncssm_per_frame': [87.9, 84.3, 86.7, 87.8, 88.1, 84.7, 83.1, 88.1],
        'gate_values': [0.004, 0.016, 0.067, 0.101, 0.092, 0.040, 0.026, 0.004],
        'nc_conv_clean_avg': 89.0, 'nc_conv_degraded_avg': 63.2,
        'bi_ncssm_clean_avg': 88.0, 'bi_ncssm_degraded_avg': 86.8,
        'bi_ncssm_overall': 88.8,
        'key_finding': 'f3 inside: 52.6% -> 87.8% (+35.2%), clean preserved within -1.0%',
    },

    # =========================================================
    # Experiment 3: Scale Ablation (1x vs 10x)
    # =========================================================
    'scale': {
        'description': 'Scale comparison: 1x (253K) vs 10x (1.8M)',
        '1x': {
            'std': {'clean': 89.2, 'c10c': 76.2, 's5': 63.6, 'params': 251434},
            'nc': {'clean': 88.9, 'c10c': 77.7, 's5': 66.6, 'params': 253258},
        },
        '10x': {
            'std': {'clean': 91.4, 'c10c': 79.9, 's5': 68.3, 'params': 1745674},
            'nc': {'clean': 91.5, 'c10c': 81.2, 's5': 70.2, 'params': 1819276},
        },
        'key_finding': 'NC gap larger at small scale (1x: +2.9% s5 vs 10x: +1.9% s5)',
    },

    # =========================================================
    # Experiment 4: CULane Lane Detection (Real Data)
    # =========================================================
    'culane': {
        'description': 'CULane lane detection (driver_37, 3357 images)',
        'std_cnn': {'normal': 78.3, 'dark': 60.0, 'noise': 71.6, 'fog': 44.2, 'avg': 63.5},
        'ncconv':  {'normal': 84.7, 'dark': 59.0, 'noise': 77.8, 'fog': 65.8, 'avg': 71.8},
        'key_finding': 'normal +6.5%, fog +21.7% on real driving data',
    },

    # =========================================================
    # Experiment 5: Per-Spatial Sigma (v8)
    # =========================================================
    'per_spatial': {
        'description': 'Per-spatial vs per-sample sigma on CIFAR-10 corruptions',
        'std_cnn':  {'clean': 89.0, 'impulse': 76.2, 'brightness': 86.7, 'gaussian': 86.3, 'contrast': 72.8, 'fog': 70.0},
        'v7_sample': {'clean': 88.9, 'impulse': 77.1, 'brightness': 86.9, 'gaussian': 87.0, 'contrast': 86.1, 'fog': 85.7},
        'v8_spatial': {'clean': 89.0, 'impulse': 78.1, 'brightness': 87.2, 'gaussian': 87.0, 'contrast': 85.4, 'fog': 85.2},
        'key_finding': 'Per-spatial solves impulse (+1.0% over v7) and dark (+0.3%)',
    },
}


def print_all_results():
    """Print formatted summary of all experiments."""
    print('\n' + '=' * 80)
    print('  NC-Conv: Complete Experiment Results')
    print('  Target: ACCV 2026 / CVPR 2027')
    print('=' * 80)

    # 1. CIFAR-10-C
    r = RESULTS['cifar10c']
    print(f'\n  [1] {r["description"]}')
    s, n = r['std_cnn_aug'], r['ncconv_aug']
    print(f'      Std CNN: clean={s["clean"]}% | C10-C avg={s["c10c_avg"]}% | {s["params"]:,} params')
    print(f'      NC-Conv: clean={n["clean"]}% | C10-C avg={n["c10c_avg"]}% | {n["params"]:,} params')
    print(f'      Gap: C10-C {n["c10c_avg"]-s["c10c_avg"]:+.1f}%')
    print(f'      Severity: ', end='')
    for sv in range(1, 6):
        gap = n['severity'][sv] - s['severity'][sv]
        print(f's{sv}={gap:+.1f}% ', end='')
    print(f'\n      Key: {r["key_finding"]}')

    # 2. Temporal
    r = RESULTS['temporal']
    print(f'\n  [2] {r["description"]}')
    print(f'      NC-Conv degraded: {r["nc_conv_degraded_avg"]}%')
    print(f'      Bi-NC-SSM degraded: {r["bi_ncssm_degraded_avg"]}% (+{r["bi_ncssm_degraded_avg"]-r["nc_conv_degraded_avg"]:.1f}%)')
    print(f'      Clean: {r["bi_ncssm_clean_avg"]}% ({r["bi_ncssm_clean_avg"]-r["nc_conv_clean_avg"]:+.1f}%)')
    print(f'      Key: {r["key_finding"]}')

    # 3. Scale
    r = RESULTS['scale']
    print(f'\n  [3] {r["description"]}')
    for scale in ['1x', '10x']:
        gap = r[scale]['nc']['c10c'] - r[scale]['std']['c10c']
        gap5 = r[scale]['nc']['s5'] - r[scale]['std']['s5']
        print(f'      {scale}: C10-C {gap:+.1f}% | s5 {gap5:+.1f}%')
    print(f'      Key: {r["key_finding"]}')

    # 4. CULane
    r = RESULTS['culane']
    print(f'\n  [4] {r["description"]}')
    for cond in ['normal', 'dark', 'noise', 'fog']:
        gap = r['ncconv'][cond] - r['std_cnn'][cond]
        print(f'      {cond:<8}: Std={r["std_cnn"][cond]:.1f}% | NC={r["ncconv"][cond]:.1f}% | {gap:+.1f}%')
    print(f'      Key: {r["key_finding"]}')

    # 5. Per-spatial
    r = RESULTS['per_spatial']
    print(f'\n  [5] {r["description"]}')
    print(f'      impulse: v7={r["v7_sample"]["impulse"]}% -> v8={r["v8_spatial"]["impulse"]}% (+{r["v8_spatial"]["impulse"]-r["v7_sample"]["impulse"]:.1f}%)')
    print(f'      Key: {r["key_finding"]}')

    print('\n' + '=' * 80)


if __name__ == '__main__':
    print_all_results()
