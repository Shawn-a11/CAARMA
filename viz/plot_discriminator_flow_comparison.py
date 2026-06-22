#!/usr/bin/env python3
"""One-off figure: SSL discriminator path vs MLP discriminator ablation."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-caarma")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT_DIR = Path("viz/outputs")
PNG_OUT = OUT_DIR / "discriminator_flow_comparison.png"
PDF_OUT = OUT_DIR / "discriminator_flow_comparison.pdf"


def add_box(ax, x, y, w, h, text, *, fc="#FFFFFF", ec="#263238",
            fontsize=11, weight="normal", color="#111827"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.7,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.22,
    )
    return box


def add_arrow(ax, x1, y1, x2, y2, *, color="#374151"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def draw_column(ax, x0, title, subtitle, boxes, *, title_color="#111827"):
    ax.text(
        x0 + 1.65,
        9.45,
        title,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x0 + 1.65,
        9.05,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4B5563",
    )

    w, h = 3.3, 0.78
    ys = [8.05, 6.85, 5.65, 4.45, 3.25, 2.05, 0.85]
    last_center = None
    for i, spec in enumerate(boxes):
        y = ys[i]
        add_box(ax, x0, y, w, h, spec["text"], fc=spec.get("fc", "#FFFFFF"),
                ec=spec.get("ec", "#263238"), fontsize=spec.get("fontsize", 10.8),
                weight=spec.get("weight", "normal"), color=spec.get("color", "#111827"))
        center_top = (x0 + w / 2, y + h)
        center_bottom = (x0 + w / 2, y)
        if last_center is not None:
            add_arrow(ax, last_center[0], last_center[1], center_top[0], center_top[1])
        last_center = center_bottom


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.5, 9), dpi=220)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 10)
    ax.axis("off")

    fig.patch.set_facecolor("#FAFAF7")
    ax.set_facecolor("#FAFAF7")

    ssl_boxes = [
        {
            "text": "input\nreal / synthetic embedding\n"
                    "e_real, e_syn in R^192",
            "fc": "#E8F4FF",
            "ec": "#2563EB",
            "weight": "bold",
        },
        {
            "text": "EnhancedAdapter\n192 → 1024",
            "fc": "#FFFFFF",
        },
        {
            "text": "reshape / pseudo sequence\n[B, 1024] → [B, 1, 1024]",
            "fc": "#FFF7ED",
            "ec": "#F97316",
            "weight": "bold",
        },
        {
            "text": "HuBERT / WavLM Transformer\ninput seq_len = 1",
            "fc": "#FEF2F2",
            "ec": "#DC2626",
            "weight": "bold",
        },
        {
            "text": "selected SSL hidden layers\ne.g., HuBERT: 7 / 9 / 11 / 12",
            "fc": "#FFFFFF",
        },
        {
            "text": "pooling + classifier head\nconcat projections → MLP head",
            "fc": "#FFFFFF",
        },
        {
            "text": "output\nreal / fake logit",
            "fc": "#ECFDF5",
            "ec": "#059669",
            "weight": "bold",
        },
    ]

    mlp_boxes = [
        {
            "text": "input\nreal / synthetic embedding\n"
                    "e_real, e_syn in R^192",
            "fc": "#E8F4FF",
            "ec": "#2563EB",
            "weight": "bold",
        },
        {
            "text": "MLP discriminator\nLinear(192, 256) -> LN -> LeakyReLU",
            "fc": "#F5F3FF",
            "ec": "#7C3AED",
            "weight": "bold",
        },
        {
            "text": "hidden layer\nDropout -> Linear(256, 128)\n-> LN -> LeakyReLU",
            "fc": "#FFFFFF",
            "fontsize": 10.2,
        },
        {
            "text": "output layer\nDropout -> Linear(128, 1)",
            "fc": "#FFFFFF",
        },
        {
            "text": "output\nreal / fake logit",
            "fc": "#ECFDF5",
            "ec": "#059669",
            "weight": "bold",
        },
    ]

    draw_column(
        ax,
        0.6,
        "Original SSL discriminator",
        "Adapter + length-1 HuBERT/WavLM pseudo-sequence",
        ssl_boxes,
        title_color="#1D4ED8",
    )
    draw_column(
        ax,
        7.0,
        "MLP discriminator ablation",
        "Direct embedding-space real/fake classifier",
        mlp_boxes,
        title_color="#6D28D9",
    )

    ax.plot([5.6, 5.6], [0.55, 9.45], color="#D1D5DB", linewidth=1.5, linestyle="--")

    ax.text(
        5.6,
        0.18,
        "Key contrast: SSL-D adapts each 192-d embedding into a length-1 pseudo-sequence; "
        "MLP-D directly discriminates in the native speaker embedding space.",
        ha="center",
        va="center",
        fontsize=10.7,
        fontweight="bold",
        color="#7C2D12",
    )

    plt.tight_layout(pad=0.8)
    fig.savefig(PNG_OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(PDF_OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(PNG_OUT.resolve())
    print(PDF_OUT.resolve())


if __name__ == "__main__":
    main()
