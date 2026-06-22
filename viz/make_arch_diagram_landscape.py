"""Landscape CAARMA architecture flow + Projection-D conditioning overlay.

Left->right flow for a 16:9 slide. Three lanes fan out from e_real:
  Lane A (top)    — AM-Softmax -> L_real
  Lane B (middle) — L2-normalize -> Projection-D -> BCE -> L_D / L_G
  Lane C (bottom) — SL-Mixup -> e_syn -> AM-Softmax(syn) -> L_syn
All losses converge into a tall L_total box on the far right.

Projection-D overlay (purple): D is conditioned on the class prototype w.
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
    "font.size": 12,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

C_AUDIO = "#A8E6A1"
C_OP    = "#9DC3E6"
C_DATA  = "#FFFFFF"
C_NORM  = "#FCE7E4"
C_LOSS  = "#FFE08A"
C_TOTAL = "#F4A98C"
C_PROJ  = "#9B59B6"   # purple — projection conditioning (NEW)
C_PROJBG= "#F0E6F7"
C_EDGE  = "#2C2C2C"
C_TEXT  = "#1A1A1A"


def box(ax, cx, cy, w, h, text, color, fontsize=11, weight="normal",
        edge=C_EDGE, lw=1.4, tcolor=C_TEXT):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=lw, edgecolor=edge, facecolor=color, zorder=3,
    )
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=tcolor, weight=weight, zorder=4)


def op(ax, cx, cy, text, w=2.7, fontsize=11, edge=C_EDGE, lw=1.4):
    box(ax, cx, cy, w, 0.66, text, C_OP, fontsize=fontsize, weight="bold",
        edge=edge, lw=lw)


def arrow(ax, x1, y1, x2, y2, color=C_EDGE, lw=1.7, rad=0.0, ls="-"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=lw, zorder=2, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlim(0, 18); ax.set_ylim(0, 9); ax.axis("off")

YA, YB, YC = 7.3, 4.5, 1.7   # lane heights

# ── Encoder (far-left vertical stack) ───────────────────────────────
EX = 2.5
box(ax, EX, 8.25, 2.3, 0.6, "Audio waveform\n[B, samples]", C_AUDIO, fontsize=10, weight="bold")
arrow(ax, EX, 7.94, EX, 7.55)
box(ax, EX, 7.25, 2.3, 0.55, "Mel feature  [B,1,T,80]", C_DATA, fontsize=9.5)
arrow(ax, EX, 6.96, EX, 6.57)
box(ax, EX, 6.27, 2.3, 0.55, "Pre-pooled  [B,T',1536]", C_DATA, fontsize=9.5)
arrow(ax, EX, 5.98, EX, 5.45)
box(ax, EX, 5.05, 2.5, 0.75, "Speaker embedding\n$e_{real}\\in\\mathbb{R}^{B\\times192}$",
    C_DATA, fontsize=10, weight="bold")

# encoder process labels — to the LEFT, right-aligned, clear of everything
for ly, lt in [(7.745, "FBank"), (6.765, "MFA-Conformer"), (5.715, "Attn pooling+FC")]:
    ax.text(EX - 1.35, ly, lt, fontsize=9, style="italic", color="#666",
            ha="right", va="center")

# ── Lane A (top) — real classification ──────────────────────────────
arrow(ax, EX + 1.05, 5.3, 4.6, YA - 0.1, rad=0.10)
op(ax, 5.7, YA, "AM-Softmax", w=2.6, fontsize=11)
arrow(ax, 7.0, YA, 7.75, YA)
box(ax, 8.4, YA, 1.7, 0.62, "$L_{real}$", C_LOSS, fontsize=13, weight="bold")

# ── Lane C (bottom) — synthetic path ────────────────────────────────
arrow(ax, EX + 1.05, 4.8, 4.3, YC + 0.1, rad=-0.10)
op(ax, 5.4, YC, "SL-Mixup", w=2.2, fontsize=11)
arrow(ax, 6.55, YC, 7.3, YC)
box(ax, 8.3, YC, 2.5, 0.64, "$e_{syn}\\in\\mathbb{R}^{B\\times192}$", C_DATA, fontsize=10, weight="bold")
arrow(ax, 9.55, YC, 10.3, YC)
op(ax, 11.5, YC, "AM-Softmax (syn)", w=2.8, fontsize=10)
arrow(ax, 12.95, YC, 13.7, YC)
box(ax, 14.35, YC, 1.7, 0.62, "$L_{syn}$", C_LOSS, fontsize=13, weight="bold")

# ── Lane B (middle) — discriminator ─────────────────────────────────
arrow(ax, EX + 1.3, 5.05, 4.2, YB + 0.08, rad=0.0)
box(ax, 5.1, YB, 2.0, 0.58, "L2-normalize", C_NORM, fontsize=9.5, weight="bold")
# e_syn -> norm-syn (up from lane C)
arrow(ax, 8.3, YC + 0.34, 8.3, 3.5, rad=0.0)
box(ax, 8.3, 3.2, 2.0, 0.58, "L2-normalize", C_NORM, fontsize=9.5, weight="bold")
# norm nodes -> Projection-D
arrow(ax, 6.1, YB, 8.0, YB - 0.05)
arrow(ax, 8.3, 3.49, 9.0, YB - 0.28)
op(ax, 9.8, YB, "Projection MLP-D", w=3.1, fontsize=11, edge=C_PROJ, lw=2.6)
arrow(ax, 11.35, YB, 11.95, YB)
box(ax, 13.2, YB, 2.5, 0.64, "$D(e_{real}),\\,D(e_{syn})$", C_DATA, fontsize=10)
arrow(ax, 14.45, YB, 15.0, YB)
op(ax, 15.55, YB, "BCE", w=1.0, fontsize=10)
arrow(ax, 16.05, YB, 16.5, YB)
box(ax, 17.05, YB, 1.2, 0.6, "$L_D/L_G$", C_LOSS, fontsize=10, weight="bold")

# ── Projection conditioning overlay (purple) ────────────────────────
box(ax, 9.8, 6.45, 5.0, 1.35,
    "Class prototypes  $W$  (AM-Softmax)\n"
    "$w_{real}=W[:,y]$   ·   $w_{syn}=\\mathrm{mix}(W_i,W_j)$\n"
    "$D(e\\,|\\,w)=\\psi(\\phi(e))+\\langle\\phi(e),Vw\\rangle$",
    C_PROJBG, fontsize=10, weight="bold", edge=C_PROJ, lw=2.2, tcolor="#5B2C6F")
arrow(ax, 9.8, 5.77, 9.8, YB + 0.36, color=C_PROJ, lw=2.2, ls="--")
ax.text(10.05, 5.5, "condition", fontsize=9.5, color=C_PROJ, style="italic", va="center")

# ── L_total (far-right tall box) — three clean horizontal feeds ─────
LT_X = 17.1
box(ax, LT_X, 4.5, 1.5, 6.5, "$L_{total}$", C_TOTAL, fontsize=15, weight="bold")
arrow(ax, 9.25, YA, LT_X - 0.78, YA)     # L_real  (top)
arrow(ax, 17.65, YB, 17.65, YA - 0.2, color=C_EDGE, lw=1.4)  # tie L_D/L_G up (short, inside box edge)
arrow(ax, 15.2, YC, LT_X - 0.78, YC)     # L_syn   (bottom)
ax.text(LT_X, 0.95,
        "$L_{total}=L_{real}+\\dfrac{1}{N_{spk}}L_{syn}+\\lambda_{adv}L_G$",
        fontsize=10.5, ha="center", va="center", color=C_TEXT, weight="bold")

# ── Legend ──────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_AUDIO,  edgecolor=C_EDGE, label="Raw input"),
    mpatches.Patch(facecolor=C_OP,     edgecolor=C_EDGE, label="Operation / module"),
    mpatches.Patch(facecolor=C_DATA,   edgecolor=C_EDGE, label="Tensor"),
    mpatches.Patch(facecolor=C_NORM,   edgecolor=C_EDGE, label="L2-normalize"),
    mpatches.Patch(facecolor=C_LOSS,   edgecolor=C_EDGE, label="Loss term"),
    mpatches.Patch(facecolor=C_PROJBG, edgecolor=C_PROJ, label="Projection-D (NEW)"),
]
ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.0, 0.0),
          fontsize=10, framealpha=0.95, ncol=3)

fig.suptitle("CAARMA + Projection-Discriminator — training flow",
             fontsize=17, weight="bold", y=0.98)
fig.savefig(f"{OUT}/arch_caarma_projd_landscape.png", bbox_inches="tight")
fig.savefig(f"{OUT}/arch_caarma_projd_landscape.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}/arch_caarma_projd_landscape.{{png,pdf}}")
