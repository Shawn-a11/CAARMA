"""Corrected CAARMA + MLP-Discriminator architecture flow diagram.

Fixes three errors in the hand-drawn version:
  1. The discriminator consumes the *embedding* e_syn, NOT the loss scalar L_syn.
  2. e_real ALSO feeds the discriminator (real/fake game needs positives).
  3. Embeddings are L2-normalized before entering D (shown as explicit nodes).
L_real / L_syn are AM-Softmax branches that run in PARALLEL with D; all three
losses join at L_total.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

C_AUDIO = "#A8E6A1"   # green — raw input
C_OP    = "#9DC3E6"   # blue — operation / module
C_DATA  = "#FFFFFF"   # white — tensor / data
C_NORM  = "#FCE7E4"   # light coral — normalize step
C_LOSS  = "#FFE08A"   # amber — loss term
C_TOTAL = "#F4A98C"   # coral — total loss
C_EDGE  = "#2C2C2C"
C_TEXT  = "#1A1A1A"


def box(ax, cx, cy, w, h, text, color, fontsize=10, weight="normal", edge=C_EDGE):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        linewidth=1.4, edgecolor=edge, facecolor=color, zorder=3,
    )
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=C_TEXT, weight=weight, zorder=4)


def op(ax, cx, cy, text, w=2.6, fontsize=10):
    box(ax, cx, cy, w, 0.62, text, C_OP, fontsize=fontsize, weight="bold")


def arrow(ax, x1, y1, x2, y2, color=C_EDGE, lw=1.6, rad=0.0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=lw, zorder=2,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(13, 15.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 15.6); ax.axis("off")

CX = 6.5   # central spine

# ── Top spine: audio → embedding ────────────────────────────────────
box(ax, CX, 14.6, 3.0, 0.7, "Audio waveform\n[B, samples]", C_AUDIO, weight="bold")
arrow(ax, CX, 14.25, CX, 13.85)
op(ax, CX + 2.3, 14.05, "FBank", w=1.8, fontsize=9.5)

box(ax, CX, 13.5, 3.2, 0.7, "Mel feature\n[B, 1, T, 80]", C_DATA)
arrow(ax, CX, 13.15, CX, 12.75)
op(ax, CX + 2.5, 12.95, "MFA-Conformer\nEncoder", fontsize=9)

box(ax, CX, 12.4, 3.5, 0.7, "Pre-pooled hidden states\n[B, T', 1536]", C_DATA, fontsize=9.5)
arrow(ax, CX, 12.05, CX, 11.65)
op(ax, CX + 2.7, 11.85, "Attentive Stat\nPooling + FC", fontsize=9)

box(ax, CX, 11.3, 4.0, 0.8,
    "Post-pooled speaker embedding\n$e_{real}\\in\\mathbb{R}^{B\\times192}$",
    C_DATA, fontsize=10, weight="bold")

# ── e_real fans out to THREE consumers ──────────────────────────────
# (a) LEFT — AM-Softmax real loss
arrow(ax, CX - 2.0, 11.15, 2.5, 10.3, rad=0.12)
op(ax, 2.5, 10.0, "AM-Softmax", fontsize=9.5)
arrow(ax, 2.5, 9.68, 2.5, 9.22)
box(ax, 2.5, 8.9, 2.0, 0.6, "$L_{real}$", C_LOSS, fontsize=12, weight="bold")

# (b) RIGHT — SL-Mixup → e_syn
arrow(ax, CX + 2.0, 11.15, 10.5, 10.3, rad=-0.12)
op(ax, 10.5, 10.0, "SL-Mixup", fontsize=9.5)
arrow(ax, 10.5, 9.68, 10.5, 9.22)
box(ax, 10.5, 8.9, 3.0, 0.7, "$e_{syn}\\in\\mathbb{R}^{B\\times192}$", C_DATA, fontsize=10, weight="bold")
# e_syn → AM-Softmax synthetic → L_syn
arrow(ax, 10.5, 8.55, 10.5, 8.12)
op(ax, 10.5, 7.8, "AM-Softmax (syn)", fontsize=9)
arrow(ax, 10.5, 7.48, 10.5, 7.02)
box(ax, 10.5, 6.7, 2.0, 0.6, "$L_{syn}$", C_LOSS, fontsize=12, weight="bold")

# (c) DISCRIMINATOR path — explicit normalize nodes, then D
# e_real → norm-real  (down, slightly left)
arrow(ax, CX, 10.9, 5.0, 8.05, rad=0.0)
box(ax, 5.0, 7.7, 2.0, 0.55, "L2-normalize", C_NORM, fontsize=9, weight="bold")
# e_syn → norm-syn  (down-left)
arrow(ax, 9.6, 8.85, 8.2, 8.05, rad=-0.10)
box(ax, 8.0, 7.7, 2.0, 0.55, "L2-normalize", C_NORM, fontsize=9, weight="bold")

# both norm nodes → MLP-D
arrow(ax, 5.0, 7.42, CX - 0.8, 6.55, rad=0.0)
arrow(ax, 8.0, 7.42, CX + 0.8, 6.55, rad=0.0)
op(ax, CX, 6.2, "MLP Discriminator", w=3.0, fontsize=10)
arrow(ax, CX, 5.88, CX, 5.42)
box(ax, CX, 5.1, 4.2, 0.7,
    "$D(e_{real}),\\ D(e_{syn})\\in\\mathbb{R}^{B\\times1}$", C_DATA, fontsize=10)
arrow(ax, CX, 4.75, CX, 4.3)
op(ax, CX, 4.0, "BCEWithLogits", fontsize=9.5)
arrow(ax, CX, 3.68, CX, 3.22)
box(ax, CX, 2.9, 2.4, 0.6, "$L_D\\ /\\ L_G$", C_LOSS, fontsize=11, weight="bold")

# ── L_total — joins L_real, L_syn, L_G with three clean vertical drops ──
box(ax, CX, 1.0, 10.4, 0.9,
    "$L_{total} = L_{real} + \\dfrac{1}{N_{spk}}L_{syn} + \\lambda_{adv}\\, L_G$",
    C_TOTAL, fontsize=13, weight="bold")
TOP = 1.45
arrow(ax, 2.5,  8.6, 2.5,  TOP)    # L_real  (far-left column)
arrow(ax, 10.5, 6.4, 10.5, TOP)    # L_syn   (far-right column)
arrow(ax, CX,   2.6, CX,   TOP)    # L_G     (center)

# ── Legend ──────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_AUDIO, edgecolor=C_EDGE, label="Raw input"),
    mpatches.Patch(facecolor=C_OP,    edgecolor=C_EDGE, label="Operation / module"),
    mpatches.Patch(facecolor=C_DATA,  edgecolor=C_EDGE, label="Tensor"),
    mpatches.Patch(facecolor=C_NORM,  edgecolor=C_EDGE, label="L2-normalize"),
    mpatches.Patch(facecolor=C_LOSS,  edgecolor=C_EDGE, label="Loss term"),
    mpatches.Patch(facecolor=C_TOTAL, edgecolor=C_EDGE, label="Total loss"),
]
ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(0.0, 0.99),
          fontsize=9.5, framealpha=0.95)

fig.suptitle("CAARMA + MLP-Discriminator — training flow",
             fontsize=16, weight="bold", y=0.995)
fig.savefig(f"{OUT}/arch_caarma_mlpdisc_corrected.png", bbox_inches="tight")
fig.savefig(f"{OUT}/arch_caarma_mlpdisc_corrected.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}/arch_caarma_mlpdisc_corrected.{{png,pdf}}")
