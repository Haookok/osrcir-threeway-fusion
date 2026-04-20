"""Regenerate the three-way fusion pipeline figure with a clean,
conference-paper-style orthogonal layout.

Layout replicates the manually verified design in Figma (file key
zSGUVyLU9o3nzPVszD3Xvy). Canvas 1600x1000 units, origin top-left.

Usage:
    python fig_pipeline_v2.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

INK = "#1F2937"
INK_MID = "#4B5563"
INK_LIGHT = "#6B7280"
ARROW = "#4B5563"
ARROW_LIGHT = "#9CA3AF"
DIVIDER = "#DCDEE5"

INPUT_FILL = "#F5F6F8"
INPUT_BORDER = "#7C8493"

MLLM1_FILL = "#FFF3D6"
MLLM1_BORDER = "#D99426"

T2I_FILL = "#E1EDFF"
T2I_BORDER = "#3D7FEA"

MLLM2_FILL = "#FFE5DB"
MLLM2_BORDER = "#D9674D"

BUBBLE_FILL = "#E8FAEE"
BUBBLE_BORDER = "#199E72"

PROXY_FILL = "#EBE0FF"
PROXY_BORDER = "#7A52E0"

CLIP_FILL = "#EDE0FF"
CLIP_BORDER = "#7A45D1"

SCORE_FILL = "#FDE7E5"
SCORE_BORDER = "#D64039"

GALLERY_FILL = "#F2F4F7"
GALLERY_BORDER = "#737A84"

TOPK_FILL = "#FDEB96"
TOPK_BORDER = "#C79318"

STAGE1_BAR = "#3B82F6"
STAGE2_BAR = "#10B981"
STAGE3_BAR = "#EF4444"


def add_box(ax, x, y, w, h, fill, border, radius=10, lw=1.5):
    """Rounded rectangle with fill + border. (x, y) is top-left in canvas coords."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=border,
        facecolor=fill,
        joinstyle="round",
    )
    ax.add_patch(patch)
    return patch


def add_text(ax, x, y, text, size=11, weight="normal", color=INK, ha="center", va="center", style="normal"):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        ha=ha,
        va=va,
        fontweight=weight,
        fontstyle=style,
    )


def box_center(x, y, w, h):
    return x + w / 2, y + h / 2


def draw_arrow(ax, points, color=ARROW, lw=1.8, dashed=False, head_size=10):
    """Draw an orthogonal polyline ending in an arrow head.

    points: list of (x, y) tuples, the last point is the arrow tip.
    """
    if len(points) < 2:
        return

    ls = (0, (5, 4)) if dashed else "-"

    # body: line segments up to (but not including) the final approach
    verts = list(points)
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    path = Path(verts[:-1] + [verts[-1]], codes)
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=lw,
            linestyle=ls,
            joinstyle="round",
            capstyle="round",
        )
    )

    # arrow head: a small FancyArrowPatch on the last segment
    p_prev = points[-2]
    p_end = points[-1]
    ax.add_patch(
        FancyArrowPatch(
            p_prev,
            p_end,
            arrowstyle="-|>",
            mutation_scale=head_size,
            color=color,
            linewidth=0,
            shrinkA=0,
            shrinkB=0,
        )
    )


def build_figure():
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1600)
    ax.set_ylim(1000, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # ===== Title =====
    add_text(
        ax,
        800,
        34,
        "Three-Way Fusion Pipeline for Zero-Shot Composed Image Retrieval",
        size=18,
        weight="bold",
        color=INK,
    )

    # ===== Stage title bars =====
    stages = [
        (40, 500, STAGE1_BAR, "Stage 1   Initial Reasoning"),
        (580, 500, STAGE2_BAR, "Stage 2   Visual Proxy & Refinement"),
        (1120, 440, STAGE3_BAR, "Stage 3   Three-Way Retrieval"),
    ]
    for x, w, color, name in stages:
        add_box(ax, x, 78, w, 36, color, color, radius=6, lw=0)
        add_text(ax, x + 20, 96, name, size=12.5, weight="bold", color="white", ha="left")

    # ===== Stage dividers =====
    for dx in (540, 1090):
        path = Path([(dx, 140), (dx, 940)], [Path.MOVETO, Path.LINETO])
        ax.add_patch(
            PathPatch(
                path,
                facecolor="none",
                edgecolor=DIVIDER,
                linewidth=1.2,
                linestyle=(0, (5, 5)),
            )
        )

    # ===== Row 1: Inference pipeline =====
    # Reference Image
    add_box(ax, 70, 180, 140, 64, INPUT_FILL, INPUT_BORDER, radius=8)
    add_text(ax, 140, 204, "Reference", size=12, weight="semibold")
    add_text(ax, 140, 224, "Image  I_r", size=11.5, color=INK_MID)

    # Modification Text
    add_box(ax, 70, 284, 140, 64, INPUT_FILL, INPUT_BORDER, radius=8)
    add_text(ax, 140, 308, "Modification", size=12, weight="semibold")
    add_text(ax, 140, 328, "Text  t", size=11.5, color=INK_MID)

    # MLLM #1 (Qwen-VL CoT)
    add_box(ax, 270, 208, 190, 108, MLLM1_FILL, MLLM1_BORDER, radius=12, lw=2)
    add_text(ax, 365, 232, "MLLM", size=14, weight="bold")
    add_text(ax, 365, 252, "(Qwen-VL-Max)", size=11.5, color=INK_MID)
    add_text(ax, 365, 278, "Reflective CoT", size=11, style="italic", color=INK_LIGHT)

    # D_1 pill
    add_box(ax, 488, 246, 90, 42, BUBBLE_FILL, BUBBLE_BORDER, radius=21, lw=1.5)
    add_text(ax, 533, 267, "D\u2081", size=17, weight="bold", color=BUBBLE_BORDER)

    # T2I
    add_box(ax, 612, 208, 180, 108, T2I_FILL, T2I_BORDER, radius=12, lw=2)
    add_text(ax, 702, 232, "T2I Model", size=14, weight="bold")
    add_text(ax, 702, 252, "(MiniMax image-01)", size=11.5, color=INK_MID)
    add_text(ax, 702, 278, "image synthesis", size=11, style="italic", color=INK_LIGHT)

    # Proxy Image P
    add_box(ax, 820, 246, 90, 42, PROXY_FILL, PROXY_BORDER, radius=21, lw=1.5)
    add_text(ax, 865, 267, "Proxy P", size=13, weight="bold", color=PROXY_BORDER)

    # MLLM #2 (V7)
    add_box(ax, 942, 208, 200, 108, MLLM2_FILL, MLLM2_BORDER, radius=12, lw=2)
    add_text(ax, 1042, 232, "MLLM", size=14, weight="bold")
    add_text(ax, 1042, 252, "(Qwen-VL-Max)", size=11.5, color=INK_MID)
    add_text(ax, 1042, 278, "V7 Anti-Hallucination CoT", size=11, style="italic", color=INK_LIGHT)

    # D_2 pill
    add_box(ax, 1170, 246, 90, 42, BUBBLE_FILL, BUBBLE_BORDER, radius=21, lw=1.5)
    add_text(ax, 1215, 267, "D\u2082", size=17, weight="bold", color=BUBBLE_BORDER)

    # ===== Row 1 arrows (horizontal chain) =====
    draw_arrow(ax, [(210, 212), (268, 212)])
    draw_arrow(ax, [(210, 316), (268, 316)])
    draw_arrow(ax, [(460, 262), (486, 262)])
    draw_arrow(ax, [(578, 262), (610, 262)])
    draw_arrow(ax, [(792, 262), (818, 262)])
    draw_arrow(ax, [(910, 262), (940, 262)])
    draw_arrow(ax, [(1142, 262), (1168, 262)])

    # Dashed skip: Reference Image reused by MLLM V7
    draw_arrow(
        ax,
        [(140, 180), (140, 150), (1042, 150), (1042, 206)],
        color=ARROW_LIGHT,
        lw=1.4,
        dashed=True,
        head_size=9,
    )
    # white backdrop under "I_r reused" label so dashed line doesn't overprint
    ax.add_patch(FancyBboxPatch(
        (556, 139), 92, 20,
        boxstyle="round,pad=0,rounding_size=3",
        facecolor="white", edgecolor="none", zorder=5,
    ))
    add_text(ax, 602, 149, "I_r  reused", size=10.5, style="italic", color=INK_LIGHT)

    # ===== Row 2: CLIP boxes =====
    # CLIP(D_1)
    add_box(ax, 468, 400, 130, 60, CLIP_FILL, CLIP_BORDER, radius=10, lw=1.5)
    add_text(ax, 533, 418, "CLIP", size=13, weight="bold", color=CLIP_BORDER)
    add_text(ax, 533, 434, "Text Encoder", size=10.5, color=INK_MID)
    add_text(ax, 533, 449, "\u2190 D\u2081", size=10.5, style="italic", color=INK_LIGHT)

    # CLIP(P)
    add_box(ax, 800, 400, 130, 60, CLIP_FILL, CLIP_BORDER, radius=10, lw=1.5)
    add_text(ax, 865, 418, "CLIP", size=13, weight="bold", color=CLIP_BORDER)
    add_text(ax, 865, 434, "Image Encoder", size=10.5, color=INK_MID)
    add_text(ax, 865, 449, "\u2190 Proxy P", size=10.5, style="italic", color=INK_LIGHT)

    # CLIP(D_2)
    add_box(ax, 1150, 400, 130, 60, CLIP_FILL, CLIP_BORDER, radius=10, lw=1.5)
    add_text(ax, 1215, 418, "CLIP", size=13, weight="bold", color=CLIP_BORDER)
    add_text(ax, 1215, 434, "Text Encoder", size=10.5, color=INK_MID)
    add_text(ax, 1215, 449, "\u2190 D\u2082", size=10.5, style="italic", color=INK_LIGHT)

    # ===== D/P -> CLIP vertical arrows =====
    draw_arrow(ax, [(533, 288), (533, 398)])
    draw_arrow(ax, [(865, 288), (865, 398)])
    draw_arrow(ax, [(1215, 288), (1215, 398)])

    # ===== Retrieval block =====
    add_box(ax, 300, 560, 980, 140, SCORE_FILL, SCORE_BORDER, radius=14, lw=2)
    add_text(ax, 790, 580, "Three-Way Fusion Scoring", size=13, weight="bold", color=SCORE_BORDER)
    add_text(
        ax, 790, 614,
        r"$f_{\mathrm{text}} \,=\, \mathrm{normalize}\left(\,\beta \cdot \mathrm{CLIP}(D_1) + (1-\beta)\cdot \mathrm{CLIP}(D_2)\,\right)$",
        size=15, color=INK,
    )
    add_text(
        ax, 790, 648,
        r"$\mathrm{score}(I_c) \,=\, \alpha \cdot \mathrm{sim}\left(f_{\mathrm{text}},\,\mathrm{CLIP}(I_c)\right) + (1-\alpha)\cdot \mathrm{sim}\left(\mathrm{CLIP}(P),\,\mathrm{CLIP}(I_c)\right)$",
        size=15, color=INK,
    )
    add_text(
        ax, 790, 680,
        r"defaults:  $\beta = 0.7$,   $\alpha = 0.9$      (task-adaptive on GeneCIS)",
        size=11, style="italic", color=INK_LIGHT,
    )

    # Gallery (right of retrieval block)
    add_box(ax, 1320, 595, 220, 70, GALLERY_FILL, GALLERY_BORDER, radius=10, lw=1.5)
    add_text(ax, 1430, 614, "Gallery  G", size=13, weight="bold")
    add_text(ax, 1430, 634, "precomputed CLIP features", size=10.5, color=INK_MID)
    add_text(ax, 1430, 650, "(candidate images  I_c)", size=10, style="italic", color=INK_LIGHT)

    # Top-K Results (below retrieval)
    add_box(ax, 670, 800, 260, 74, TOPK_FILL, TOPK_BORDER, radius=12, lw=2)
    add_text(ax, 800, 823, "Top-K Retrieval Results", size=15, weight="bold")
    add_text(ax, 800, 850, "ranked by  score(I_c)", size=11, color=INK_MID)

    # ===== Final arrows =====
    # CLIP row -> Retrieval block top
    draw_arrow(ax, [(533, 460), (533, 558)])
    draw_arrow(ax, [(865, 460), (865, 558)])
    draw_arrow(ax, [(1215, 460), (1215, 558)])

    # Gallery left -> Retrieval right
    draw_arrow(ax, [(1318, 630), (1282, 630)])

    # Retrieval bottom -> Top-K top
    draw_arrow(ax, [(800, 702), (800, 798)], lw=2.0, head_size=11)

    return fig


def main():
    fig = build_figure()
    png_path = os.path.join(OUT_DIR, "fig_pipeline.png")
    pdf_path = os.path.join(OUT_DIR, "fig_pipeline.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
