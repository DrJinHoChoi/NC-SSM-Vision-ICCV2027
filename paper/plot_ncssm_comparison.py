#!/usr/bin/env python3
"""NC-SSM vs Baselines: Noise Robustness Comparison Charts"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('paper', exist_ok=True)

# ============================================================
# Data: [-15, -10, -5, 0, 5, 10, 15] dB
# ============================================================
snr = [-15, -10, -5, 0, 5, 10, 15]

nc_ssm = {
    'factory': [60.8, 66.6, 76.5, 86.7, 90.2, 92.2, 92.7],
    'white':   [43.8, 61.9, 74.8, 84.4, 90.0, 92.4, 93.7],
    'babble':  [72.6, 78.8, 84.7, 89.8, 92.2, 93.7, 94.3],
    'street':  [59.3, 63.1, 74.2, 82.5, 88.7, 92.2, 93.5],
    'pink':    [45.5, 60.3, 75.5, 86.0, 91.4, 93.4, 94.2],
    'clean': 94.8, 'params': 7443,
}

bc_resnet = {
    'factory': [64.5, 74.8, 83.5, 89.0, 91.9, 92.9, 93.9],
    'white':   [58.6, 64.6, 76.7, 85.9, 90.2, 92.6, 93.8],
    'babble':  [76.7, 84.3, 88.8, 92.1, 93.8, 94.4, 95.0],
    'street':  [60.7, 65.1, 77.7, 84.8, 90.1, 93.0, 94.1],
    'pink':    [58.9, 66.0, 79.7, 88.4, 91.8, 93.8, 94.5],
    'clean': 95.3, 'params': 7464,
}

nm_matched = {
    'factory': [61.1, 70.6, 82.3, 88.5, 91.4, 93.0, 93.8],
    'white':   [31.1, 59.1, 75.0, 85.5, 89.8, 92.8, 93.9],
    'babble':  [71.4, 79.0, 86.0, 91.0, 93.0, 93.9, 94.5],
    'street':  [57.6, 64.2, 73.9, 83.8, 88.4, 92.4, 94.2],
    'pink':    [23.6, 52.4, 76.9, 87.3, 91.7, 93.0, 94.4],
    'clean': 95.1, 'params': 7402,
}

ds_cnn = {
    'factory': [64.4, 74.6, 85.9, 91.5, 93.8, 95.1, 95.5],
    'white':   [61.7, 67.1, 81.6, 89.9, 93.1, 94.8, 95.6],
    'babble':  [79.2, 86.2, 91.4, 94.5, 95.5, 95.9, 96.5],
    'street':  [61.1, 69.4, 81.5, 88.9, 92.8, 95.0, 95.9],
    'pink':    [61.2, 66.4, 82.7, 90.6, 93.8, 95.1, 95.9],
    'clean': 96.8, 'params': 23756,
}

noises = ['factory', 'white', 'babble', 'street', 'pink']
noise_kr = {'factory':'Factory', 'white':'White', 'babble':'Babble',
            'street':'Street', 'pink':'Pink'}

models_all = [
    ('NC-SSM (7,443)', nc_ssm, '#E63946', 'o', '-'),
    ('BC-ResNet-1 (7,464)', bc_resnet, '#457B9D', 's', '-'),
    ('NM-Matched (7,402)', nm_matched, '#F4A261', '^', '--'),
    ('DS-CNN-S (23,756)', ds_cnn, '#2A9D8F', 'D', ':'),
]

# ============================================================
# Fig 1: 5 noise subplots
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(24, 5.5), sharey=True)
fig.suptitle('Noise Robustness Comparison: Accuracy (%) vs SNR (dB)',
             fontsize=16, fontweight='bold', y=1.02)

for i, n in enumerate(noises):
    ax = axes[i]
    for name, data, color, marker, ls in models_all:
        ax.plot(snr, data[n], color=color, marker=marker, markersize=7,
                linewidth=2.2, linestyle=ls, label=name)
    ax.set_title(noise_kr[n], fontsize=14, fontweight='bold')
    ax.set_xlabel('SNR (dB)', fontsize=11)
    if i == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(snr)
    ax.set_ylim(15, 100)
    ax.grid(True, alpha=0.25)
    ax.axhline(90, color='gray', ls='--', alpha=0.3)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=11,
           bbox_to_anchor=(0.5, -0.08), frameon=True)
plt.tight_layout()
plt.savefig('paper/fig_ncssm_all_noise.png', dpi=150, bbox_inches='tight')
plt.close()
print("[1/5] fig_ncssm_all_noise.png")


# ============================================================
# Fig 2: Average across 5 noises
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
for name, data, color, marker, ls in models_all:
    avg = [np.mean([data[n][j] for n in noises]) for j in range(7)]
    ax.plot(snr, avg, color=color, marker=marker, markersize=9,
            linewidth=2.8, linestyle=ls, label=name)
    # annotate 0dB value
    idx0 = 3
    ax.annotate(f'{avg[idx0]:.1f}', (snr[idx0], avg[idx0]),
                textcoords="offset points", xytext=(8, -12), fontsize=9,
                color=color, fontweight='bold')

ax.set_xlabel('SNR (dB)', fontsize=13)
ax.set_ylabel('Average Accuracy (%)', fontsize=13)
ax.set_title('Average Noise Robustness (5 Noise Types)', fontsize=15, fontweight='bold')
ax.set_xticks(snr)
ax.set_ylim(40, 100)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('paper/fig_ncssm_avg.png', dpi=150, bbox_inches='tight')
plt.close()
print("[2/5] fig_ncssm_avg.png")


# ============================================================
# Fig 3: NC-SSM vs BC-ResNet-1 gap (grouped bar)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(7)
w = 0.15
noise_colors = ['#264653','#2A9D8F','#E9C46A','#F4A261','#E76F51']

for i, n in enumerate(noises):
    gap = [nc_ssm[n][j] - bc_resnet[n][j] for j in range(7)]
    ax.bar(x + i*w, gap, w, label=noise_kr[n], color=noise_colors[i], alpha=0.85)

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('NC-SSM minus BC-ResNet-1 (%p)', fontsize=12)
ax.set_title('NC-SSM vs BC-ResNet-1: Per-Condition Accuracy Gap',
             fontsize=14, fontweight='bold')
ax.set_xticks(x + w*2)
ax.set_xticklabels([str(s) for s in snr])
ax.axhline(0, color='black', lw=1)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, axis='y', alpha=0.3)
ax.axhspan(ax.get_ylim()[0], 0, alpha=0.05, color='red')

plt.tight_layout()
plt.savefig('paper/fig_ncssm_vs_bc_gap.png', dpi=150, bbox_inches='tight')
plt.close()
print("[3/5] fig_ncssm_vs_bc_gap.png")


# ============================================================
# Fig 4: -15dB bar comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(5)
w = 0.2

model_list = [
    ('NC-SSM', nc_ssm, '#E63946'),
    ('BC-ResNet-1', bc_resnet, '#457B9D'),
    ('NM-Matched', nm_matched, '#F4A261'),
    ('DS-CNN-S', ds_cnn, '#2A9D8F'),
]

for mi, (mname, mdata, mc) in enumerate(model_list):
    vals = [mdata[n][0] for n in noises]
    bars = ax.bar(x + mi*w, vals, w, label=mname, color=mc, alpha=0.88)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.8,
                f'{v:.1f}', ha='center', fontsize=7.5, fontweight='bold')

ax.set_xlabel('Noise Type', fontsize=13)
ax.set_ylabel('Accuracy at -15 dB (%)', fontsize=13)
ax.set_title('Extreme Noise: -15 dB SNR Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + w*1.5)
ax.set_xticklabels([noise_kr[n] for n in noises], fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 85)

plt.tight_layout()
plt.savefig('paper/fig_ncssm_minus15.png', dpi=150, bbox_inches='tight')
plt.close()
print("[4/5] fig_ncssm_minus15.png")


# ============================================================
# Fig 5: NC-SSM improvement over NM-Matched at -15dB
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

improve = [nc_ssm[n][0] - nm_matched[n][0] for n in noises]
bar_c = ['#2A9D8F' if v >= 0 else '#E63946' for v in improve]

bars = ax.bar([noise_kr[n] for n in noises], improve,
              color=bar_c, alpha=0.85, edgecolor='black', lw=0.5)
ax.axhline(0, color='black', lw=1)

for bar, v in zip(bars, improve):
    label = f'+{v:.1f}' if v >= 0 else f'{v:.1f}'
    ypos = v + 0.4 if v >= 0 else v - 1.5
    ax.text(bar.get_x() + bar.get_width()/2, ypos, label,
            ha='center', fontsize=13, fontweight='bold')

ax.set_ylabel('NC-SSM minus NM-Matched (%p)', fontsize=13)
ax.set_title('NC-SSM vs NM-Matched: Improvement at -15 dB',
             fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

ax.annotate('Broadband noise:\nLSG + sub-band gate\neffective!',
            xy=(3.8, 21.9), fontsize=9, color='#264653',
            ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('paper/fig_ncssm_vs_nm_improve.png', dpi=150, bbox_inches='tight')
plt.close()
print("[5/5] fig_ncssm_vs_nm_improve.png")

print("\n=== All 5 charts generated in paper/ ===")
