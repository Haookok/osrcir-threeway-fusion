"""Generate all thesis figures for OSrCIR Three-Way Fusion paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['TeX Gyre Termes', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

BLUE = '#2563EB'
RED = '#DC2626'
GREEN = '#16A34A'
ORANGE = '#EA580C'
PURPLE = '#7C3AED'
GRAY = '#6B7280'
LIGHT_BLUE = '#DBEAFE'
LIGHT_RED = '#FEE2E2'


# ============================================================
# Figure 1: Pipeline Flowchart
# ============================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(x, y, w, h, text, color='#E5E7EB', ec='#374151', fontsize=8, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor=ec, linewidth=1.2)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, weight=weight, wrap=True)
        return (x + w/2, y + h/2)

    def arrow(x1, y1, x2, y2, color='#374151'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.3))

    # Stage labels
    ax.text(0.3, 5.7, 'Stage 1', fontsize=9, weight='bold', color=BLUE, style='italic')
    ax.text(4.2, 5.7, 'Stage 2', fontsize=9, weight='bold', color=GREEN, style='italic')
    ax.text(7.2, 5.7, 'Retrieval', fontsize=9, weight='bold', color=RED, style='italic')

    # Input
    box(0.1, 4.5, 1.2, 0.7, 'Reference\nImage', '#DBEAFE', fontsize=7.5)
    box(0.1, 3.4, 1.2, 0.7, 'Modification\nText', '#DBEAFE', fontsize=7.5)

    # Stage 1: MLLM -> D1 -> T2I -> Proxy
    c1 = box(1.8, 3.8, 1.3, 0.9, 'MLLM\n(Qwen-VL)', '#FEF3C7', fontsize=7.5)
    arrow(1.3, 4.85, 1.8, 4.4)
    arrow(1.3, 3.75, 1.8, 4.1)

    c2 = box(3.5, 4.5, 0.9, 0.6, '$D_1$', '#DCFCE7', fontsize=9, bold=True)
    arrow(3.1, 4.3, 3.5, 4.8)

    c3 = box(3.5, 3.3, 0.9, 0.6, 'T2I\nModel', '#FEF3C7', fontsize=7.5)
    arrow(3.95, 4.5, 3.95, 3.9)

    c4 = box(3.5, 2.2, 0.9, 0.6, 'Proxy\nImage', '#E0E7FF', fontsize=7.5)
    arrow(3.95, 3.3, 3.95, 2.8)

    # Stage 2: MLLM V7 -> D2
    c5 = box(5.0, 3.5, 1.4, 0.9, 'MLLM (V7)\nAnti-Halluc.', '#DCFCE7', fontsize=7.5)
    arrow(1.3, 4.85, 5.0, 4.2)   # ref image -> MLLM2
    arrow(4.4, 2.5, 5.0, 3.7)    # proxy -> MLLM2
    arrow(1.3, 3.6, 5.0, 3.8)    # mod text -> MLLM2

    c6 = box(5.2, 2.2, 0.9, 0.6, '$D_2$', '#DCFCE7', fontsize=9, bold=True)
    arrow(5.7, 3.5, 5.65, 2.8)

    # CLIP Encoding
    clip_y = 1.0
    c7 = box(3.0, clip_y, 1.0, 0.6, 'CLIP\n($D_1$)', '#F3E8FF', fontsize=7.5)
    arrow(3.95, 4.5, 3.5, clip_y + 0.6)

    c8 = box(4.5, clip_y, 1.0, 0.6, 'CLIP\n($D_2$)', '#F3E8FF', fontsize=7.5)
    arrow(5.65, 2.2, 5.0, clip_y + 0.6)

    c9 = box(6.0, clip_y, 1.0, 0.6, 'CLIP\n(Proxy)', '#F3E8FF', fontsize=7.5)
    arrow(3.95, 2.2, 6.5, clip_y + 0.6)

    # Fusion
    fy = 0.1
    box(3.5, fy, 1.6, 0.5, r'$\beta \cdot D_1 + (1{-}\beta) \cdot D_2$',
        '#BBF7D0', fontsize=7.5)
    arrow(3.5, clip_y, 4.0, fy + 0.5)
    arrow(5.0, clip_y, 4.8, fy + 0.5)

    # Score
    box(7.2, 1.5, 2.3, 1.2,
        r'$\mathrm{score} = \alpha \cdot \mathrm{sim}(f_{text}, g)$'
        '\n' + r'$+ (1{-}\alpha) \cdot \mathrm{sim}(f_{proxy}, g)$',
        '#FEE2E2', fontsize=7.5)

    arrow(5.1, fy + 0.25, 7.2, 1.8)
    arrow(6.5, clip_y + 0.3, 7.2, 1.8)

    # Gallery
    box(7.5, 3.5, 1.5, 0.7, 'Gallery\n(CLIP features)', '#E5E7EB', fontsize=7.5)
    arrow(8.25, 3.5, 8.25, 2.7)

    # Result
    box(7.8, 0.2, 1.2, 0.7, 'Top-K\nResults', '#FEF3C7', fontsize=8, bold=True)
    arrow(8.35, 1.5, 8.4, 0.9)

    # Greek letter annotations
    ax.text(4.3, -0.3, r'$\beta = 0.7$', fontsize=8, color=GREEN, ha='center')
    ax.text(8.0, -0.3, r'$\alpha = 0.9$', fontsize=8, color=RED, ha='center')

    fig.savefig(os.path.join(OUT_DIR, 'fig_pipeline.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_pipeline.png'))
    plt.close(fig)
    print("  -> fig_pipeline.pdf/png")


# ============================================================
# Figure 2: Main Results Bar Chart (9 datasets)
# ============================================================
def fig_main_results():
    datasets = [
        'FIQ\ndress', 'FIQ\nshirt', 'FIQ\ntoptee',
        'CIRCO', 'CIRR',
        'GeneCIS\nch_obj', 'GeneCIS\nfo_obj', 'GeneCIS\nch_attr', 'GeneCIS\nfo_attr'
    ]
    metrics = ['R@10', 'R@10', 'R@10', 'mAP@10', 'R@1',
               'R@1', 'R@1', 'R@1', 'R@1']
    baseline = [15.80, 26.00, 23.09, 16.21, 22.96,
                13.83, 16.02, 12.65, 18.82]
    threeway = [19.29, 27.35, 27.35, 21.26, 25.90,
                25.51, 23.62, 21.79, 27.83]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline (OSrCIR)',
                   color=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.8)
    bars2 = ax.bar(x + width/2, threeway, width, label='Three-Way Fusion (Ours)',
                   color=LIGHT_RED, edgecolor=RED, linewidth=0.8)

    for i, (b, t) in enumerate(zip(baseline, threeway)):
        delta = t - b
        ax.text(x[i] + width/2, t + 0.5, f'+{delta:.1f}',
                ha='center', va='bottom', fontsize=6.5, color=RED, weight='bold')

    ax.set_ylabel('Primary Metric (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=7.5)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, max(threeway) * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    ax.axvline(x=4.5, color=GRAY, linestyle='--', alpha=0.4)
    ax.text(2.0, max(threeway) * 1.12, 'FashionIQ / CIRCO / CIRR',
            ha='center', fontsize=7, color=GRAY, style='italic')
    ax.text(6.75, max(threeway) * 1.12, 'GeneCIS (task-adaptive)',
            ha='center', fontsize=7, color=GRAY, style='italic')

    fig.savefig(os.path.join(OUT_DIR, 'fig_main_results.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_main_results.png'))
    plt.close(fig)
    print("  -> fig_main_results.pdf/png")


# ============================================================
# Figure 3: Prompt Evolution (CIRR 200-sample)
# ============================================================
def fig_prompt_evolution():
    prompts = ['Baseline\n(D1 only)', 'Original\nprompt', 'V5\n(CoT)',
               'V6\n(full ctx)', 'V7\n(anti-halluc.)', 'V7 +\nEnsemble']
    r10 = [67.0, 64.5, 65.5, 49.5, 73.5, 69.0]
    colors = [GRAY, ORANGE, ORANGE, RED, GREEN, BLUE]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(range(len(prompts)), r10, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.7, alpha=0.85)

    ax.axhline(y=67.0, color=GRAY, linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(len(prompts) - 0.5, 67.5, 'Baseline', fontsize=7, color=GRAY, ha='right')

    for i, (v, c) in enumerate(zip(r10, colors)):
        delta = v - 67.0
        sign = '+' if delta >= 0 else ''
        ax.text(i, v + 0.8, f'{v}\n({sign}{delta:.1f})',
                ha='center', va='bottom', fontsize=7, color=c, weight='bold')

    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels(prompts, fontsize=7.5)
    ax.set_ylabel('CIRR R@10 (200 samples)')
    ax.set_ylim(40, 80)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(os.path.join(OUT_DIR, 'fig_prompt_evolution.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_prompt_evolution.png'))
    plt.close(fig)
    print("  -> fig_prompt_evolution.pdf/png")


# ============================================================
# Figure 4: Alpha/Beta Heatmaps (GeneCIS grid search)
# ============================================================
def fig_heatmap():
    alphas = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    betas  = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    def parse_grid(lines):
        grid = np.full((len(betas), len(alphas)), np.nan)
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                b, a, r1 = float(parts[0]), float(parts[1]), float(parts[2])
                bi = betas.index(b) if b in betas else -1
                ai = alphas.index(a) if a in alphas else -1
                if bi >= 0 and ai >= 0:
                    grid[bi, ai] = r1
        return grid

    # change_object grid data (from log)
    co_lines = """   0.30  0.95    14.85
   0.40  0.95    14.80
   0.50  0.90    14.80
   0.50  0.95    14.69
   0.30  1.00    14.64
   0.60  0.90    14.59
   0.60  0.95    14.59
   0.30  0.90    14.49
   0.40  1.00    14.44
   0.50  0.85    14.44
   1.00  0.95    14.29
   0.40  0.85    14.18
   0.50  1.00    14.18
   0.80  0.90    14.18
   0.40  0.90    14.13
   0.60  0.85    14.13
   0.70  0.95    14.13
   0.30  0.80    14.08
   0.80  0.95    14.08
   0.30  0.85    13.98
   0.50  0.75    13.98
   0.90  0.95    13.98
   0.50  0.70    13.93
   0.70  0.90    13.93
   1.00  0.90    13.93
   0.60  1.00    13.88
   0.70  1.00    13.88
   0.80  1.00    13.78
   0.90  1.00    13.57
   0.30  0.75    13.52
   0.40  0.80    13.52
   0.60  0.80    13.52
   0.70  0.85    13.52
   0.80  0.85    13.47
   0.90  0.90    13.47
   0.30  0.70    13.42
   0.40  0.75    13.42
   0.60  0.75    13.42
   0.70  0.80    13.42
   0.80  0.80    13.42
   0.40  0.70    13.37
   0.50  0.80    13.37
   0.60  0.70    13.37
   0.70  0.75    13.37
   0.80  0.75    13.37
   0.90  0.85    13.37
   0.90  0.80    13.27
   0.90  0.75    13.16
   1.00  0.85    13.16
   0.80  0.70    13.11
   0.90  0.70    13.01
   1.00  0.80    12.96
   1.00  0.75    12.76
   1.00  0.70    12.55""".strip().split('\n')

    # focus_object grid data (from log)
    fo_lines = """   1.00  0.90    16.07
   0.90  0.95    16.02
   1.00  1.00    16.02
   0.80  0.95    15.97
   0.90  0.90    15.97
   0.90  1.00    15.97
   1.00  0.95    15.92
   1.00  0.85    15.82
   0.80  1.00    15.77
   1.00  0.70    15.61
   1.00  0.80    15.61
   0.70  1.00    15.56
   0.60  1.00    15.51
   0.80  0.90    15.51
   0.70  0.95    15.41
   0.90  0.85    15.41
   0.90  0.80    15.36
   0.60  0.70    15.31
   0.60  0.75    15.31
   0.70  0.80    15.31
   0.70  0.75    15.26
   0.40  0.70    15.20
   1.00  0.75    15.20
   0.40  1.00    15.15
   0.60  0.80    15.15
   0.70  0.90    15.15
   0.80  0.85    15.10
   0.50  1.00    15.00
   0.50  0.70    14.95
   0.60  0.85    14.95
   0.80  0.80    14.95
   0.50  0.75    14.90
   0.30  0.70    14.85
   0.60  0.90    14.85
   0.80  0.75    14.85
   0.50  0.80    14.80
   0.60  0.95    14.80
   0.70  0.85    14.80
   0.40  0.75    14.74
   0.50  0.95    14.74
   0.40  0.80    14.69
   0.70  0.70    14.64
   0.80  0.70    14.64
   0.30  0.75    14.59
   0.40  0.95    14.59
   0.40  0.85    14.54
   0.50  0.85    14.49
   0.30  0.80    14.44
   0.50  0.90    14.44
   0.40  0.90    14.39
   0.30  0.85    14.34
   0.30  0.90    14.13
   0.30  0.95    14.03
   0.30  1.00    14.90
   0.90  0.70    14.90
   0.90  0.75    15.05""".strip().split('\n')

    # focus_attribute grid data (from log)
    fa_lines = """   0.60  0.85    20.37
   0.60  0.75    20.27
   0.50  0.85    20.12
   0.60  0.70    20.12
   0.70  0.85    20.07
   0.70  0.80    20.02
   0.50  0.80    19.97
   0.60  0.80    19.97
   0.70  0.75    19.97
   0.40  0.80    19.92
   0.40  0.85    19.92
   0.60  0.90    19.92
   0.50  0.70    19.87
   0.50  1.00    19.87
   0.60  0.95    19.87
   0.60  1.00    19.87
   0.70  0.90    19.87
   0.80  0.80    19.87
   0.50  0.75    19.82
   0.50  0.90    19.82
   0.70  1.00    19.82
   0.50  0.95    19.77
   0.90  0.70    19.77
   0.90  0.85    19.77
   0.90  0.90    19.77
   0.70  0.95    19.72
   0.80  0.75    19.72
   0.80  0.85    19.72
   0.90  0.80    19.72
   0.30  0.85    19.67
   0.40  0.90    19.67
   0.40  0.75    19.62
   0.80  0.70    19.62
   0.30  0.80    19.57
   0.80  0.90    19.57
   0.30  0.70    19.52
   0.90  0.75    19.52
   0.30  0.75    19.47
   0.40  0.70    19.47
   0.40  0.95    19.42
   0.40  1.00    19.42
   0.80  0.95    19.42
   0.90  0.95    19.42
   0.90  1.00    19.32
   0.80  1.00    19.32
   0.30  0.90    19.27
   0.70  0.70    19.27
   1.00  0.80    19.22
   1.00  0.75    19.22
   1.00  0.70    19.17
   1.00  0.85    19.12
   0.30  0.95    19.07
   1.00  0.90    18.92
   0.30  1.00    18.72
   1.00  0.95    18.67
   1.00  1.00    18.82""".strip().split('\n')

    grids = {
        'change_object': parse_grid(co_lines),
        'focus_object': parse_grid(fo_lines),
        'focus_attribute': parse_grid(fa_lines),
    }
    baselines = {
        'change_object': 13.83,
        'focus_object': 16.02,
        'focus_attribute': 18.82,
    }
    best_points = {
        'change_object': (0.30, 0.95),
        'focus_object': (1.00, 0.90),
        'focus_attribute': (0.60, 0.85),
    }

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    titles = ['change_object', 'focus_object', 'focus_attribute']

    for idx, (title, ax) in enumerate(zip(titles, axes)):
        delta_grid = grids[title] - baselines[title]
        vmax = np.nanmax(np.abs(delta_grid))
        im = ax.imshow(delta_grid, cmap='RdYlGn', aspect='auto',
                       vmin=-vmax, vmax=vmax, origin='lower')

        best_b, best_a = best_points[title]
        bi = betas.index(best_b)
        ai = alphas.index(best_a)
        ax.plot(ai, bi, 'k*', markersize=12, markeredgecolor='white', markeredgewidth=0.8)

        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f'{a:.2f}' for a in alphas], fontsize=6.5)
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([f'{b:.1f}' for b in betas], fontsize=6.5)
        ax.set_xlabel(r'$\alpha$', fontsize=9)
        if idx == 0:
            ax.set_ylabel(r'$\beta$', fontsize=9)
        ax.set_title(title.replace('_', ' '), fontsize=9, weight='bold')

        for bi_ in range(len(betas)):
            for ai_ in range(len(alphas)):
                v = delta_grid[bi_, ai_]
                if not np.isnan(v):
                    color = 'white' if abs(v) > vmax * 0.6 else 'black'
                    ax.text(ai_, bi_, f'{v:+.1f}', ha='center', va='center',
                            fontsize=5, color=color)

    fig.colorbar(im, ax=axes, label=r'$\Delta$R@1 vs Baseline', shrink=0.8, pad=0.02)
    fig.suptitle(r'GeneCIS $\alpha/\beta$ Grid Search ($\bigstar$ = best)', fontsize=11, y=1.02)

    fig.savefig(os.path.join(OUT_DIR, 'fig_heatmap.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_heatmap.png'))
    plt.close(fig)
    print("  -> fig_heatmap.pdf/png")


# ============================================================
# Figure 5: FashionIQ Detailed Results
# ============================================================
def fig_fashioniq_detail():
    subsets = ['dress', 'shirt', 'toptee']
    metrics = ['R@1', 'R@5', 'R@10', 'R@50']

    baseline_data = {
        'dress':  [4.17, 10.79, 15.80, 32.74],
        'shirt':  [9.07, 19.64, 26.00, 42.94],
        'toptee': [6.81, 16.69, 23.09, 41.19],
    }
    threeway_data = {
        'dress':  [5.63, 13.56, 19.29, 38.16],
        'shirt':  [9.77, 21.44, 27.35, 44.84],
        'toptee': [8.68, 20.28, 27.35, 46.44],
    }

    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=False)

    for i, (subset, ax) in enumerate(zip(subsets, axes)):
        x = np.arange(len(metrics))
        width = 0.35
        b_vals = baseline_data[subset]
        t_vals = threeway_data[subset]

        ax.bar(x - width/2, b_vals, width, label='Baseline',
               color=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.8)
        ax.bar(x + width/2, t_vals, width, label='Three-Way',
               color=LIGHT_RED, edgecolor=RED, linewidth=0.8)

        for j, (b, t) in enumerate(zip(b_vals, t_vals)):
            ax.text(x[j] + width/2, t + 0.4, f'+{t-b:.1f}',
                    ha='center', va='bottom', fontsize=6, color=RED, weight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_title(subset, fontsize=10, weight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        if i == 0:
            ax.set_ylabel('Recall (%)')
            ax.legend(fontsize=7, loc='upper left')

    fig.savefig(os.path.join(OUT_DIR, 'fig_fashioniq.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_fashioniq.png'))
    plt.close(fig)
    print("  -> fig_fashioniq.pdf/png")


# ============================================================
# Figure 6: Method Ablation (module contribution)
# ============================================================
def fig_ablation():
    methods = [
        'Baseline\n(D1 only)',
        '+ Proxy\n(post-fusion)',
        '+ Refinement\n(V7, D2 only)',
        '+ Ensemble\n(D1+D2)',
        'Three-Way\n(full)',
    ]
    # Representative data from CIRR 200-sample and FIQ dress
    cirr_r10 =  [67.0, 69.0, 73.5, 69.0, 69.0]
    dress_r10 = [15.5, 17.0, 13.0, 18.0, 19.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    colors_list = [GRAY, BLUE, GREEN, ORANGE, RED]

    for ax, data, title in [(ax1, cirr_r10, 'CIRR R@10 (200 samples)'),
                             (ax2, dress_r10, 'FIQ dress R@10 (200 samples)')]:
        bars = ax.bar(range(len(methods)), data, color=colors_list, width=0.65, alpha=0.85,
                      edgecolor='white', linewidth=0.8)
        ax.axhline(y=data[0], color=GRAY, linestyle='--', alpha=0.4, linewidth=0.8)

        for i, v in enumerate(data):
            ax.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom',
                    fontsize=7, weight='bold', color=colors_list[i])

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=6.5)
        ax.set_title(title, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    fig.savefig(os.path.join(OUT_DIR, 'fig_ablation.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_ablation.png'))
    plt.close(fig)
    print("  -> fig_ablation.pdf/png")


# ============================================================
# Figure 7: Relative Improvement Radar/Bar
# ============================================================
def fig_relative_improvement():
    datasets = ['FIQ dress', 'FIQ shirt', 'FIQ toptee', 'CIRCO', 'CIRR',
                'ch_obj', 'fo_obj', 'ch_attr', 'fo_attr']
    relative = [22.1, 5.2, 18.4, 31.2, 12.8, 84.5, 47.4, 72.3, 47.9]

    fig, ax = plt.subplots(figsize=(7, 3))
    colors_map = [BLUE]*3 + [PURPLE] + [GREEN] + [ORANGE]*4
    bars = ax.barh(range(len(datasets)), relative, color=colors_map, alpha=0.8,
                   edgecolor='white', height=0.7)

    for i, v in enumerate(relative):
        ax.text(v + 1, i, f'+{v:.1f}%', va='center', fontsize=7.5, weight='bold',
                color=colors_map[i])

    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=8)
    ax.set_xlabel('Relative Improvement (%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    fig.savefig(os.path.join(OUT_DIR, 'fig_relative.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_relative.png'))
    plt.close(fig)
    print("  -> fig_relative.pdf/png")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating thesis figures...")
    fig_pipeline()
    fig_main_results()
    fig_prompt_evolution()
    fig_heatmap()
    fig_fashioniq_detail()
    fig_ablation()
    fig_relative_improvement()
    print(f"\nAll figures saved to: {OUT_DIR}")
