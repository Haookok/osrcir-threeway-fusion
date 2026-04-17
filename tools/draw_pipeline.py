"""Draw the Three-Way Fusion pipeline diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 7.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.5)
ax.axis('off')

font_cn = {'fontfamily': 'Noto Sans CJK SC', 'fontsize': 10}
font_cn_s = {'fontfamily': 'Noto Sans CJK SC', 'fontsize': 8.5}
font_cn_title = {'fontfamily': 'Noto Sans CJK SC', 'fontsize': 11, 'fontweight': 'bold'}
font_en = {'fontsize': 8, 'fontfamily': 'monospace', 'fontstyle': 'italic'}

# Colors
c_input = '#E8F0FE'
c_model = '#FFF3E0'
c_output = '#E8F5E9'
c_fusion = '#FCE4EC'
c_border = '#333333'
c_round1 = '#1565C0'
c_round2 = '#E65100'
c_fusion_c = '#AD1457'

def box(x, y, w, h, text, color, border='#555', fontdict=None, text2=None):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=border, linewidth=1.2)
    ax.add_patch(rect)
    fd = fontdict or font_cn
    if text2:
        ax.text(x + w/2, y + h/2 + 0.18, text, ha='center', va='center', **fd)
        ax.text(x + w/2, y + h/2 - 0.22, text2, ha='center', va='center', **font_en)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', **fd)

def arrow(x1, y1, x2, y2, color='#555', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle='arc3,rad=0'))

def arrow_curve(x1, y1, x2, y2, color='#555', rad=0.3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                connectionstyle=f'arc3,rad={rad}'))

# === Title ===
ax.text(7, 7.2, '三路融合方法流程', ha='center', va='center',
        fontfamily='Noto Sans CJK SC', fontsize=14, fontweight='bold')

# === Round 1 ===
ax.text(3.5, 6.55, '第一轮：生成初始描述与代理图', ha='center', va='center',
        fontfamily='Noto Sans CJK SC', fontsize=11, fontweight='bold', color=c_round1)

box(0.3, 5.5, 1.8, 0.8, '参考图像', c_input)
box(0.3, 4.3, 1.8, 0.8, '修改文本', c_input)

box(2.8, 4.8, 2.0, 1.0, 'MLLM', c_model, text2='(Qwen-VL-Max)')
arrow(2.1, 5.9, 2.8, 5.5)
arrow(2.1, 4.7, 2.8, 5.1)

box(5.5, 4.9, 2.0, 0.8, '初始描述 D1', c_output)
arrow(4.8, 5.3, 5.5, 5.3)

box(5.5, 3.6, 2.0, 0.8, '代理图像', c_output, text2='(MiniMax)')
arrow(6.5, 4.9, 6.5, 4.4)

# === Round 2 ===
ax.text(10.5, 6.55, '第二轮：反幻觉精炼', ha='center', va='center',
        fontfamily='Noto Sans CJK SC', fontsize=11, fontweight='bold', color=c_round2)

box(8.5, 4.8, 2.0, 1.0, 'MLLM', c_model, text2='(V7 Prompt)')

# Arrows: ref image, proxy, text → MLLM round 2
arrow_curve(1.2, 5.5, 8.5, 5.7, color=c_round2, rad=-0.15)
arrow(7.5, 4.0, 8.5, 4.9, color=c_round2)
arrow_curve(1.2, 4.3, 8.5, 4.9, color=c_round2, rad=0.2)

ax.text(4.2, 6.0, '参考图', ha='center', va='center', fontsize=7.5,
        fontfamily='Noto Sans CJK SC', color=c_round2, fontstyle='italic')
ax.text(8.0, 4.3, '代理图', ha='center', va='center', fontsize=7.5,
        fontfamily='Noto Sans CJK SC', color=c_round2, fontstyle='italic')

box(11.2, 4.9, 2.2, 0.8, '精炼描述 D2', c_output)
arrow(10.5, 5.3, 11.2, 5.3)

# === Fusion ===
ax.text(7, 2.7, '检索阶段：三路融合', ha='center', va='center',
        fontfamily='Noto Sans CJK SC', fontsize=11, fontweight='bold', color=c_fusion_c)

# CLIP boxes
box(2.5, 1.5, 1.8, 0.7, 'CLIP(D1)', '#E3F2FD')
box(5.0, 1.5, 1.8, 0.7, 'CLIP(D2)', '#E3F2FD')

arrow(6.5, 4.9, 3.4, 2.2, color='#666')
arrow(12.3, 4.9, 5.9, 2.2, color='#666')

# Ensemble
box(3.5, 0.3, 2.0, 0.7, '描述融合', c_fusion, fontdict=font_cn_s,
    text2='b*D1+(1-b)*D2')
arrow(3.4, 1.5, 4.0, 1.0)
arrow(5.9, 1.5, 5.0, 1.0)

# Proxy CLIP
box(7.5, 1.5, 2.0, 0.7, 'CLIP(代理图)', '#E3F2FD')
arrow(6.5, 3.6, 8.5, 2.2, color='#666')

# Final score
box(8.0, 0.3, 2.5, 0.7, '最终得分', c_fusion, fontdict=font_cn_s,
    text2='α·sim_text+(1-α)·sim_proxy')
arrow(4.8, 0.65, 8.0, 0.65, color=c_fusion_c)
arrow(8.5, 1.5, 8.8, 1.0, color=c_fusion_c)

# Result
box(11.3, 0.3, 2.0, 0.7, '检索结果', '#C8E6C9')
arrow(10.5, 0.65, 11.3, 0.65, color=c_fusion_c, lw=2)

# Parameter annotations
ax.text(4.5, 0.0, 'β=0.7', ha='center', fontsize=8, color='#888', fontstyle='italic')
ax.text(9.2, 0.0, 'α=0.9', ha='center', fontsize=8, color='#888', fontstyle='italic')

import os
out_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'pipeline_diagram.png')
plt.tight_layout()
plt.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'Saved: {out_path}')
