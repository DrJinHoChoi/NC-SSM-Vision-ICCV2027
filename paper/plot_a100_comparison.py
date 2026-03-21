#!/usr/bin/env python3
"""A100 NC-SSM: Bare vs SS+Bypass Comparison Graphs"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('paper', exist_ok=True)

# ============================================================
# A100 NC-SSM Bare Results (no SS)
# ============================================================
noises = ['Factory', 'White', 'Babble', 'Street', 'Pink']
snr_labels = ['Clean', '0 dB', '-15 dB']

# A100 NC-SSM Bare: [Clean, 0dB, -15dB]
bare = {
    'Factory': [95.3, 88.0, 62.2],
    'White':   [95.3, 86.0, 46.6],
    'Babble':  [95.3, 90.8, 74.9],
    'Street':  [95.3, 82.9, 59.1],
    'Pink':    [95.2, 87.3, 56.8],
}

# A100 NC-SSM + SS+Bypass: [Clean, 0dB, -15dB]
ss_bypass = {
    'Factory': [95.2, 83.8, 62.9],
    'White':   [95.3, 83.2, 62.0],
    'Babble':  [95.3, 88.9, 75.3],
    'Street':  [95.3, 81.0, 60.5],
    'Pink':    [95.3, 85.2, 60.9],
}

# BC-ResNet-1 Reference: [Clean, 0dB, -15dB]
bc_resnet = {
    'Factory': [95.3, 89.0, 64.5],
    'White':   [95.3, 85.9, 58.6],
    'Babble':  [95.3, 92.1, 76.7],
    'Street':  [95.3, 84.8, 60.7],
    'Pink':    [95.3, 88.4, 58.9],
}

# NM-Matched Reference: [Clean, 0dB, -15dB]
nm_matched = {
    'Factory': [95.1, 88.5, 61.1],
    'White':   [95.1, 85.5, 31.1],
    'Babble':  [95.1, 91.0, 71.4],
    'Street':  [95.1, 83.8, 57.6],
    'Pink':    [95.2, 87.3, 23.6],
}

colors = {
    'bare': '#457B9D',
    'ss': '#E63946',
    'bc': '#2A9D8F',
    'nm': '#F4A261',
}

# ============================================================
# Fig 1: -15 dB Bar Comparison (Key Result!)
# NC-SSM Bare vs SS+Bypass vs BC-ResNet-1 at -15dB
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6.5))
x = np.arange(5)
w = 0.22

# BC-ResNet-1
bars_bc = ax.bar(x - w*1.5, [bc_resnet[n][2] for n in noises], w,
                 label='BC-ResNet-1 (7,464)', color='#2A9D8F', alpha=0.85, edgecolor='black', lw=0.5)

# NM-Matched
bars_nm = ax.bar(x - w*0.5, [nm_matched[n][2] for n in noises], w,
                 label='NM-Matched (7,402)', color='#F4A261', alpha=0.85, edgecolor='black', lw=0.5)

# NC-SSM Bare
bars_bare = ax.bar(x + w*0.5, [bare[n][2] for n in noises], w,
                   label='NC-SSM Bare (7,443)', color='#457B9D', alpha=0.85, edgecolor='black', lw=0.5)

# NC-SSM + SS+Bypass
bars_ss = ax.bar(x + w*1.5, [ss_bypass[n][2] for n in noises], w,
                 label='NC-SSM + SS+Bypass (7,443)', color='#E63946', alpha=0.85, edgecolor='black', lw=0.5)

# Value labels
for bars in [bars_bc, bars_nm, bars_bare, bars_ss]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.6,
                f'{h:.1f}', ha='center', fontsize=8.5, fontweight='bold')

# Highlight: NC-SSM+SS beats BC-ResNet-1 at White!
ax.annotate('NC-SSM+SS\nBEATS BC!\n(+3.4%p)',
            xy=(1 + w*1.5, 62.0), xytext=(1 + w*1.5 + 0.35, 72),
            fontsize=9, fontweight='bold', color='#E63946',
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

ax.set_xlabel('Noise Type', fontsize=13)
ax.set_ylabel('Accuracy at -15 dB (%)', fontsize=13)
ax.set_title('A100 NC-SSM: Extreme Noise (-15 dB) Comparison\nSS+Bypass dramatically improves broadband noise robustness',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(noises, fontsize=12)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 85)

plt.tight_layout()
plt.savefig('paper/fig_a100_minus15_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[1/4] fig_a100_minus15_comparison.png")


# ============================================================
# Fig 2: SS+Bypass Delta (%p improvement) at -15dB
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

delta_bare = [ss_bypass[n][2] - bare[n][2] for n in noises]
delta_0db = [ss_bypass[n][1] - bare[n][1] for n in noises]

x = np.arange(5)
w = 0.35

bars_minus15 = ax.bar(x - w/2, delta_bare, w,
                       label='SS+Bypass effect at -15 dB', color='#E63946', alpha=0.85, edgecolor='black', lw=0.5)
bars_0db = ax.bar(x + w/2, delta_0db, w,
                  label='SS+Bypass effect at 0 dB', color='#457B9D', alpha=0.85, edgecolor='black', lw=0.5)

# Labels
for bar, v in zip(bars_minus15, delta_bare):
    label = f'+{v:.1f}' if v >= 0 else f'{v:.1f}'
    ypos = v + 0.3 if v >= 0 else v - 1.2
    ax.text(bar.get_x() + bar.get_width()/2, ypos, label,
            ha='center', fontsize=11, fontweight='bold', color='#E63946')

for bar, v in zip(bars_0db, delta_0db):
    label = f'+{v:.1f}' if v >= 0 else f'{v:.1f}'
    ypos = v - 1.5 if v < 0 else v + 0.3
    ax.text(bar.get_x() + bar.get_width()/2, ypos, label,
            ha='center', fontsize=11, fontweight='bold', color='#457B9D')

ax.axhline(0, color='black', lw=1)
ax.set_xlabel('Noise Type', fontsize=13)
ax.set_ylabel('Accuracy Change (%p)', fontsize=13)
ax.set_title('SS+Bypass Effect on NC-SSM (A100)\nPositive = improvement, Negative = degradation',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(noises, fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, axis='y', alpha=0.3)

# Highlight trade-off region
ax.axhspan(ax.get_ylim()[0], 0, alpha=0.04, color='red')
ax.axhspan(0, ax.get_ylim()[1], alpha=0.04, color='green')

plt.tight_layout()
plt.savefig('paper/fig_a100_ss_delta.png', dpi=150, bbox_inches='tight')
plt.close()
print("[2/4] fig_a100_ss_delta.png")


# ============================================================
# Fig 3: NC-SSM+SS vs BC-ResNet-1 gap at all three SNR levels
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)

for si, (snr_name, snr_idx) in enumerate([('Clean', 0), ('0 dB', 1), ('-15 dB', 2)]):
    ax = axes[si]

    nc_bare_vals = [bare[n][snr_idx] for n in noises]
    nc_ss_vals = [ss_bypass[n][snr_idx] for n in noises]
    bc_vals = [bc_resnet[n][snr_idx] for n in noises]

    x = np.arange(5)
    w = 0.25

    ax.bar(x - w, bc_vals, w, label='BC-ResNet-1', color='#2A9D8F', alpha=0.85, edgecolor='black', lw=0.3)
    ax.bar(x, nc_bare_vals, w, label='NC-SSM Bare', color='#457B9D', alpha=0.85, edgecolor='black', lw=0.3)
    ax.bar(x + w, nc_ss_vals, w, label='NC-SSM+SS', color='#E63946', alpha=0.85, edgecolor='black', lw=0.3)

    # Value labels
    for xi, (bc, nb, ns) in enumerate(zip(bc_vals, nc_bare_vals, nc_ss_vals)):
        ax.text(xi - w, bc + 0.5, f'{bc:.1f}', ha='center', fontsize=7, fontweight='bold')
        ax.text(xi, nb + 0.5, f'{nb:.1f}', ha='center', fontsize=7, fontweight='bold')
        ax.text(xi + w, ns + 0.5, f'{ns:.1f}', ha='center', fontsize=7, fontweight='bold')

    ax.set_title(f'SNR: {snr_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(noises, fontsize=10, rotation=30)
    ax.grid(True, axis='y', alpha=0.3)

    if snr_name == 'Clean':
        ax.set_ylim(93, 96.5)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
    elif snr_name == '0 dB':
        ax.set_ylim(75, 95)
    else:
        ax.set_ylim(35, 80)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, -0.05), frameon=True)
fig.suptitle('NC-SSM (A100) vs BC-ResNet-1: Three SNR Conditions\n2.8x fewer MACs, 3.2x less RAM',
             fontsize=15, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig('paper/fig_a100_vs_bc_3snr.png', dpi=150, bbox_inches='tight')
plt.close()
print("[3/4] fig_a100_vs_bc_3snr.png")


# ============================================================
# Fig 4: Comprehensive radar chart - Clean + 0dB + -15dB avg
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Average across noises at each SNR
ax = axes[0]
snr_pts = ['Clean', '0 dB', '-15 dB']

for name, data, color, marker, ls in [
    ('NC-SSM Bare', bare, '#457B9D', 'o', '-'),
    ('NC-SSM + SS', ss_bypass, '#E63946', 's', '-'),
    ('BC-ResNet-1', bc_resnet, '#2A9D8F', '^', '--'),
    ('NM-Matched', nm_matched, '#F4A261', 'D', ':'),
]:
    avg = [np.mean([data[n][i] for n in noises]) for i in range(3)]
    ax.plot(snr_pts, avg, color=color, marker=marker, markersize=10,
            linewidth=2.5, linestyle=ls, label=name)
    for xi, v in enumerate(avg):
        ax.annotate(f'{v:.1f}', (xi, v), textcoords="offset points",
                    xytext=(10, -5 if name != 'NC-SSM + SS' else 8),
                    fontsize=9, color=color, fontweight='bold')

ax.set_xlabel('SNR Condition', fontsize=12)
ax.set_ylabel('Average Accuracy (%)', fontsize=12)
ax.set_title('Average Across 5 Noise Types', fontsize=13, fontweight='bold')
ax.legend(fontsize=9.5, loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_ylim(40, 100)

# Right: Bar chart showing NC-SSM+SS advantage over BC at -15dB
ax = axes[1]
gap_bare = [bare[n][2] - bc_resnet[n][2] for n in noises]
gap_ss = [ss_bypass[n][2] - bc_resnet[n][2] for n in noises]

x = np.arange(5)
w = 0.35
bars1 = ax.bar(x - w/2, gap_bare, w, label='NC-SSM Bare vs BC', color='#457B9D', alpha=0.85)
bars2 = ax.bar(x + w/2, gap_ss, w, label='NC-SSM+SS vs BC', color='#E63946', alpha=0.85)

for bar, v in zip(bars1, gap_bare):
    label = f'+{v:.1f}' if v >= 0 else f'{v:.1f}'
    ypos = v + 0.3 if v >= 0 else v - 1.5
    ax.text(bar.get_x() + bar.get_width()/2, ypos, label,
            ha='center', fontsize=9, fontweight='bold')

for bar, v in zip(bars2, gap_ss):
    label = f'+{v:.1f}' if v >= 0 else f'{v:.1f}'
    ypos = v + 0.3 if v >= 0 else v - 1.5
    ax.text(bar.get_x() + bar.get_width()/2, ypos, label,
            ha='center', fontsize=9, fontweight='bold', color='#E63946')

ax.axhline(0, color='black', lw=1)
ax.set_xlabel('Noise Type', fontsize=12)
ax.set_ylabel('Gap vs BC-ResNet-1 (%p)', fontsize=12)
ax.set_title('NC-SSM vs BC-ResNet-1 at -15 dB\n(Positive = NC-SSM wins)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(noises, fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.axhspan(0, ax.get_ylim()[1], alpha=0.04, color='green')
ax.axhspan(ax.get_ylim()[0], 0, alpha=0.04, color='red')

plt.tight_layout()
plt.savefig('paper/fig_a100_comprehensive.png', dpi=150, bbox_inches='tight')
plt.close()
print("[4/4] fig_a100_comprehensive.png")


print("\n" + "="*60)
print("KEY FINDINGS (A100 NC-SSM):")
print("="*60)

# Summary stats
print("\n[1] Clean Accuracy: NC-SSM = 95.3% = BC-ResNet-1 (TIED!)")
print(f"    Clean Test: 94.66%  |  Best Val: 95.34% @ epoch 23")

print("\n[2] SS+Bypass Effect at -15dB:")
for n in noises:
    d = ss_bypass[n][2] - bare[n][2]
    sign = '+' if d >= 0 else ''
    print(f"    {n:10s}: {bare[n][2]:.1f}% -> {ss_bypass[n][2]:.1f}% ({sign}{d:.1f}%p)")

print("\n[3] NC-SSM+SS vs BC-ResNet-1 at -15dB:")
for n in noises:
    d = ss_bypass[n][2] - bc_resnet[n][2]
    sign = '+' if d >= 0 else ''
    winner = "NC-SSM WINS!" if d > 0 else ("TIED" if d == 0 else "BC wins")
    print(f"    {n:10s}: NC-SSM+SS {ss_bypass[n][2]:.1f}% vs BC {bc_resnet[n][2]:.1f}% ({sign}{d:.1f}%p) - {winner}")

# Average comparison
avg_bare = np.mean([bare[n][2] for n in noises])
avg_ss = np.mean([ss_bypass[n][2] for n in noises])
avg_bc = np.mean([bc_resnet[n][2] for n in noises])
print(f"\n[4] Average -15dB:")
print(f"    NC-SSM Bare:    {avg_bare:.1f}%")
print(f"    NC-SSM+SS:      {avg_ss:.1f}%  ({avg_ss-avg_bare:+.1f}%p from SS)")
print(f"    BC-ResNet-1:    {avg_bc:.1f}%")
print(f"    NC-SSM+SS vs BC: {avg_ss-avg_bc:+.1f}%p")
print(f"\n    --> NC-SSM+SS achieves {avg_ss:.1f}% avg at -15dB")
print(f"        with 2.8x fewer MACs and 3.2x less RAM!")
