"""Hand-authored schematic figures for the CAARMA discriminator deck.

Generates publication-style figures (PNG + PDF), each ~8 boxes max,
explaining how a pretrained SSL model (HuBERT / WavLM) is used as the
adversarial discriminator in CAARMA:

  fig_6_1_input_contradiction.{png,pdf}  → 标准 HuBERT vs CAARMA 用法对照
  fig_6_2_d_architecture.{png,pdf}        → D 内部结构（HuBERT 折叠成 1 个盒子）
  fig_6_3_training_mechanism.{png,pdf}    → D step / M step 交替更新
  fig_6_4_gradient_flow.{png,pdf}         → 前向 vs 反向梯度路径（含 frozen 标识）
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Color palette
C_INPUT   = "#FFE5B4"   # peach — input embedding
C_TRAIN   = "#D4E6D4"   # sage — trainable
C_FROZEN  = "#E0E0E0"   # gray — frozen / pretrained
C_OUTPUT  = "#FFD6D6"   # rose — output logit
C_AUDIO   = "#D6E4F0"   # washed blue — raw audio
C_NOISE   = "#E6DFF0"   # lavender — noise / latent
C_LABEL   = "#FFFACD"   # lemon — labels / metadata
C_TEXT    = "#2C2C2C"


def box(ax, x, y, w, h, text, color, fontsize=10, edge="#444", lw=1.5, weight="normal"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=lw, edgecolor=edge, facecolor=color, zorder=2,
    )
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=C_TEXT, weight=weight, zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#444", lw=1.6, style="->", linestyle="-",
          label=None, label_offset=(0.05, 0)):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=15,
        color=color, linewidth=lw, linestyle=linestyle, zorder=1,
    )
    ax.add_patch(a)
    if label is not None:
        ax.text((x1+x2)/2 + label_offset[0], (y1+y2)/2 + label_offset[1],
                label, fontsize=8, color=color, ha="left", va="center")


# ============================================================================
# FIG 6.0 — GAN recap: standard GAN vs CAARMA's adversarial setup
# ============================================================================
# Tall canvas, large fonts, simpler mapping layout (only the 4 entries that
# matter, with big arrow chips). 16x9 keeps the slide aspect.
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

# Title
fig.suptitle("Standard GAN vs CAARMA — D's role is conceptually identical",
             fontsize=18, weight="bold", y=0.97)

# Vertical divider
ax.plot([8.0, 8.0], [0.4, 8.3], color="#bbb", linestyle="--", linewidth=1.2)

# ─── LEFT: Standard GAN ─────────────────────────────────────────────
ax.text(4.0, 8.15, "Standard GAN  (Goodfellow 2014)",
        fontsize=16, weight="bold", ha="center", color=C_TEXT)

# Top row: z → G → fake
box(ax, 0.5, 6.3, 2.0, 0.9, "Noise z\n~ N(0, I)",   C_NOISE,  fontsize=12)
arrow(ax, 2.55, 6.75, 3.0, 6.75)
box(ax, 3.0, 6.3, 2.0, 0.9, "Generator G",          C_TRAIN,  fontsize=13, weight="bold")
arrow(ax, 5.05, 6.75, 5.5, 6.75)
box(ax, 5.5, 6.3, 2.0, 0.9, "fake sample",          C_INPUT,  fontsize=12)

# Middle row: real x feeds D
box(ax, 3.0, 4.4, 2.0, 0.9, "Real data x",          C_INPUT,  fontsize=12)
arrow(ax, 5.05, 4.85, 5.6, 5.6)  # real → D
arrow(ax, 6.5, 6.25, 6.5, 5.6)   # fake → D

# D box
box(ax, 5.5, 4.0, 2.0, 1.0, "Discriminator D",      C_TRAIN,  fontsize=14, weight="bold")
arrow(ax, 6.5, 3.95, 6.5, 3.3)
box(ax, 5.5, 2.5, 2.0, 0.7, "real / fake ?",        C_OUTPUT, fontsize=12)

# Goals (left column, bottom)
ax.text(0.7, 1.7, "G:  fool D into saying \"real\"",
        fontsize=12, color="#1976D2", style="italic", weight="bold")
ax.text(0.7, 1.1, "D:  correctly label real vs fake",
        fontsize=12, color="#2A7", style="italic", weight="bold")
ax.text(0.7, 0.5, "→  D is discarded after training",
        fontsize=11, color="#666")

# ─── RIGHT: CAARMA mapping ──────────────────────────────────────────
ax.text(12.0, 8.15, "CAARMA mapping",
        fontsize=16, weight="bold", ha="center", color=C_TEXT)
ax.text(12.0, 7.65, "(same alternating game, different inputs)",
        fontsize=12, ha="center", color="#666", style="italic")

# Six concise mapping rows with clear visual separation
mappings = [
    ("z  (random noise)",        "real audio waveform"),
    ("G  (generator)",            "MFA-Conformer encoder M"),
    ("G's output",                "e  (encoder embedding)"),
    ("(no analog in GAN)",        "+  e_syn = mixup(e_i, e_j)"),
    ("D",                         "MixupDiscriminator (HuBERT-backed)"),
    ("D's job",                   "tell real e from synthetic e_syn"),
]
y = 6.7
for left, right in mappings:
    # Light background row for readability
    ax.add_patch(mpatches.FancyBboxPatch(
        (8.4, y - 0.28), 7.4, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0, facecolor="#F5F2EC", zorder=0,
    ))
    ax.text(8.6,  y, left,  fontsize=12.5, ha="left", va="center",
            color="#1976D2", weight="bold")
    ax.text(11.45, y, "→", fontsize=14, ha="center", va="center", color="#666")
    ax.text(11.7, y, right, fontsize=12.5, ha="left", va="center",
            color="#2A7", weight="bold")
    y -= 0.75

# Bottom footnote on right
ax.text(12.0, 0.7,
        "→  D is also discarded at inference (same as standard GAN)",
        fontsize=12, ha="center", color="#666", style="italic", weight="bold")

fig.savefig(f"{OUT}/fig_6_0_gan_recap.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_6_0_gan_recap.pdf", bbox_inches="tight")
plt.close(fig)
print("  6.0 fig_6_0_gan_recap.{png,pdf}")


# ============================================================================
# FIG 6.1 — Input-type contradiction resolved
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")

# LEFT: standard HuBERT usage
ax.text(2.5, 4.7, "Standard HuBERT usage", fontsize=13, weight="bold", ha="center")
box(ax, 0.8, 3.7, 3.4, 0.6, "Raw waveform  (T samples @ 16 kHz)", C_AUDIO)
arrow(ax, 2.5, 3.65, 2.5, 3.3)
box(ax, 0.8, 2.6, 3.4, 0.6, "Conv feature extractor (7 layers)", C_FROZEN)
arrow(ax, 2.5, 2.55, 2.5, 2.2)
box(ax, 0.8, 1.5, 3.4, 0.6, "Transformer encoder (24 L)", C_FROZEN)
arrow(ax, 2.5, 1.45, 2.5, 1.1)
box(ax, 0.8, 0.4, 3.4, 0.6, "Hidden states  (T/320, 1024)", C_OUTPUT)

# Divider
ax.plot([5.5, 5.5], [0.2, 4.5], color="#bbb", linestyle="--", linewidth=1)

# RIGHT: CAARMA usage
ax.text(8.5, 4.7, "CAARMA discriminator usage", fontsize=13, weight="bold", ha="center")
box(ax, 6.8, 3.7, 3.4, 0.6, "Speaker embedding  (B, 192)", C_INPUT)
arrow(ax, 8.5, 3.65, 8.5, 3.3)
box(ax, 6.8, 2.6, 3.4, 0.6, "EnhancedAdapter   192 → 1024", C_TRAIN, weight="bold")
arrow(ax, 8.5, 2.55, 8.5, 2.2)
box(ax, 6.8, 1.5, 3.4, 0.6, "Transformer encoder (24 L)\nfrom HuBERT — pretrained", C_FROZEN)
arrow(ax, 8.5, 1.45, 8.5, 1.1)
box(ax, 6.8, 0.4, 3.4, 0.6, "Hidden states  (B, 1, 1024) × 24", C_OUTPUT)

# Annotation: conv extractor crossed out
ax.text(8.5, 5.0, "[X] conv extractor BYPASSED", fontsize=10, color="#c62828",
        weight="bold", ha="center", style="italic")
ax.annotate("", xy=(6.6, 1.8), xytext=(5.7, 2.6),
            arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.4, linestyle="--"))

ax.text(5.5, -0.1, "Same Transformer stack, different input pipeline",
        ha="center", fontsize=10, style="italic", color="#555")
fig.suptitle("Resolving 'embedding ≠ speech': the adapter trick", fontsize=14, weight="bold")
fig.savefig(f"{OUT}/fig_6_1_input_contradiction.png")
fig.savefig(f"{OUT}/fig_6_1_input_contradiction.pdf")
plt.close(fig)
print("  6.1 fig_6_1_input_contradiction.{png,pdf}")


# ============================================================================
# FIG 6.2 — Discriminator internal architecture
# ============================================================================
# Wider canvas + extra top space for a horizontal legend strip so it can't
# collide with the e/e_syn input boxes.
fig, ax = plt.subplots(figsize=(12, 10.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 11); ax.axis("off")

# Column layout: param labels (left) — boxes (center) — legend (top, horizontal)
BOX_X = 3.6
BOX_W = 4.8
CX    = BOX_X + BOX_W / 2

# Inputs (real + fake) — placed above the box column
box(ax, 1.6, 9.0, 3.0, 0.7, "e   (real, 192)",          C_INPUT, fontsize=11)
box(ax, 7.4, 9.0, 3.0, 0.7, "e_syn  (synthetic, 192)",  C_INPUT, fontsize=11)
arrow(ax, 3.1, 8.95, CX, 8.4)
arrow(ax, 8.9, 8.95, CX, 8.4)

# Inside D (single column, wider)
box(ax, BOX_X, 7.5, BOX_W, 0.7,
    "EnhancedAdapter      192 → 1024\n(4 × Linear + GELU + LayerNorm)",
    C_TRAIN, fontsize=11)
ax.text(BOX_X - 0.1, 7.85, "~5M\ntrainable",
        fontsize=10, color="#2A7", ha="right", va="center", weight="bold")
arrow(ax, CX, 7.45, CX, 7.0)

box(ax, BOX_X, 5.9, BOX_W, 1.1,
    "HuBERT / WavLM Transformer encoder\n24 layers · hidden 1024 · pretrained\n(conv extractor unused)",
    C_FROZEN, fontsize=11, weight="bold")
ax.text(BOX_X - 0.1, 6.55,
        "315M params\n— frozen in our code\n— trainable in source",
        fontsize=10, color="#555", ha="right", va="center")
arrow(ax, CX, 5.85, CX, 5.4,
      label="  layer 7, 9, 11, 12 hidden states", label_offset=(0.0, 0))

box(ax, BOX_X, 4.4, BOX_W, 0.7,
    "LayerNorm + MultiHead Attn Pool   → (B, 1024) × 4",
    C_TRAIN, fontsize=10)
arrow(ax, CX, 4.35, CX, 3.9)

box(ax, BOX_X, 2.9, BOX_W, 0.7,
    "Linear projections 1024 → 256 × 4\n+ softmax(layer_weights) + concat",
    C_TRAIN, fontsize=10)
ax.text(BOX_X - 0.1, 3.25, "~3M\ntrainable",
        fontsize=10, color="#2A7", ha="right", va="center", weight="bold")
arrow(ax, CX, 2.85, CX, 2.4)

box(ax, BOX_X, 1.4, BOX_W, 0.7,
    "ResidualBlock × 2   →   Linear(256, 1)",
    C_TRAIN, fontsize=10)
ax.text(BOX_X - 0.1, 1.75, "~0.7M\ntrainable",
        fontsize=10, color="#2A7", ha="right", va="center", weight="bold")
arrow(ax, CX, 1.35, CX, 0.9)

box(ax, BOX_X + 0.6, 0.1, BOX_W - 1.2, 0.7,
    "real / fake logit   (B, 1)",
    C_OUTPUT, weight="bold", fontsize=12)

# Side bracket label — a thin vertical line + label, replacing the rotated text
ax.plot([2.5, 2.5], [0.4, 8.0], color="#888", lw=1.5, zorder=1)
ax.plot([2.5, 2.9], [8.0, 8.0], color="#888", lw=1.5, zorder=1)
ax.plot([2.5, 2.9], [0.4, 0.4], color="#888", lw=1.5, zorder=1)
ax.text(2.3, 4.2, "Discriminator D\n(MixupDiscriminator)",
        fontsize=11, weight="bold", color="#444",
        rotation=90, ha="center", va="center")

# Legend — horizontal, placed at the very top below the title
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,  edgecolor="#444", label="Input embedding"),
    mpatches.Patch(facecolor=C_TRAIN,  edgecolor="#444", label="Trainable (our code)"),
    mpatches.Patch(facecolor=C_FROZEN, edgecolor="#444", label="HuBERT backbone (frozen)"),
    mpatches.Patch(facecolor=C_OUTPUT, edgecolor="#444", label="Output logit"),
]
ax.legend(handles=legend_items, loc="upper center",
          bbox_to_anchor=(0.5, 1.04), ncol=4,
          fontsize=10, framealpha=0.95)

fig.suptitle("Where HuBERT lives inside the discriminator",
             fontsize=14, weight="bold", y=0.99)
fig.savefig(f"{OUT}/fig_6_2_d_architecture.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_6_2_d_architecture.pdf", bbox_inches="tight")
plt.close(fig)
print("  6.2 fig_6_2_d_architecture.{png,pdf}")


# ============================================================================
# FIG 6.3 — Alternating training mechanism (D step vs M step)
# ============================================================================
fig, ax = plt.subplots(figsize=(11.5, 6.5))
ax.set_xlim(0, 11.5); ax.set_ylim(0, 6.5); ax.axis("off")

# Common header
ax.text(5.75, 6.2, "Each batch — Algorithm 2: update D then M", fontsize=13,
        weight="bold", ha="center")

# LEFT column: D step
ax.text(2.75, 5.6, "Step 1 — Update D",
        fontsize=12, weight="bold", ha="center", color="#2A7")
box(ax, 0.5, 4.5, 4.5, 0.7, "encoder forward inside  torch.no_grad()", C_FROZEN, fontsize=9)
arrow(ax, 2.75, 4.45, 2.75, 4.1)
box(ax, 0.5, 3.3, 4.5, 0.7, "mixup(e, W)  →  e_syn", C_FROZEN, fontsize=9)
arrow(ax, 2.75, 3.25, 2.75, 2.9)
box(ax, 0.5, 2.1, 4.5, 0.7, "L_D = BCE(D(e), 1)  +  BCE(D(e_syn), 0)", C_INPUT, fontsize=10)
arrow(ax, 2.75, 2.05, 2.75, 1.7)
box(ax, 0.5, 0.9, 4.5, 0.7, "opt_D.step()  ←  updates D only", C_TRAIN, fontsize=10, weight="bold")

# Divider
ax.plot([5.75, 5.75], [0.3, 5.4], color="#bbb", linestyle="--", linewidth=1)

# RIGHT column: M step
ax.text(8.75, 5.6, "Step 2 — Update M (encoder + W)",
        fontsize=12, weight="bold", ha="center", color="#1976D2")
box(ax, 6.5, 4.5, 4.5, 0.7, "encoder forward  (grad enabled)", C_TRAIN, fontsize=9)
arrow(ax, 8.75, 4.45, 8.75, 4.1)
box(ax, 6.5, 3.3, 4.5, 0.7,
    "D forward — D frozen via toggle_optimizer",
    C_FROZEN, fontsize=9)
arrow(ax, 8.75, 3.25, 8.75, 2.9)
box(ax, 6.5, 2.1, 4.5, 0.7,
    "L_total = L_real + (1/N)·L_syn + λ_adv · L_G",
    C_INPUT, fontsize=10)
arrow(ax, 8.75, 2.05, 8.75, 1.7)
box(ax, 6.5, 0.9, 4.5, 0.7,
    "opt_M.step()  ←  updates encoder + W only",
    C_TRAIN, fontsize=10, weight="bold")

# Big arrow connecting bottom
arrow(ax, 5.0, 1.25, 6.5, 1.25, lw=2.2, color="#444")
ax.text(5.75, 1.1, "next batch", fontsize=8, ha="center", color="#666", style="italic")

# Annotation
ax.text(5.75, 0.15,
        "Standard GAN alternation — D and M never updated in the same backward pass.",
        ha="center", fontsize=9.5, style="italic", color="#555")

fig.suptitle("How the SSL backbone is trained (or kept frozen) in CAARMA",
             fontsize=14, weight="bold")
fig.savefig(f"{OUT}/fig_6_3_training_mechanism.png")
fig.savefig(f"{OUT}/fig_6_3_training_mechanism.pdf")
plt.close(fig)
print("  6.3 fig_6_3_training_mechanism.{png,pdf}")


# ============================================================================
# FIG 6.4 — Gradient flow during M step (encoder update)
# ============================================================================
# Wide layout: left = annotation column, right = D stack column. Plenty of
# whitespace between to avoid overlap. Title at very top with extra clearance.
fig, ax = plt.subplots(figsize=(14, 10.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 11); ax.axis("off")

# Title — placed in figure suptitle area only, leaving the canvas clear
fig.suptitle("Gradient flow during the M step (encoder update)",
             fontsize=15, weight="bold", y=0.98)
ax.text(7.0, 10.3, "All of D is frozen — but the gradient still flows through it",
        fontsize=12, ha="center", color="#666", style="italic")

# Right column: D blocks (frozen) — vertical stack with generous spacing
COL_X = 8.5
COL_W = 4.8
ROW_H = 0.75
# y-positions chosen so blocks are well-separated, backward arrows fit cleanly
D_BLOCKS = [
    (8.7, "Discriminator output  (B, 1) logit",        C_OUTPUT),
    (7.6, "ResidualBlock x 2  +  Linear(256, 1)",      C_FROZEN),
    (6.5, "Multi-layer pool + projections + softmax",  C_FROZEN),
    (5.4, "HuBERT Transformer encoder  (24 layers)",   C_FROZEN),
    (4.3, "EnhancedAdapter   192 → 1024",              C_FROZEN),
]
for y, t, c in D_BLOCKS:
    box(ax, COL_X, y, COL_W, ROW_H, t, c, fontsize=10.5)

# Input embeddings — below adapter, with clear vertical gap
box(ax, COL_X,         3.0, 2.1, ROW_H, "e   (real)",  C_INPUT, fontsize=10.5)
box(ax, COL_X + 2.7,   3.0, 2.1, ROW_H, "e_syn",       C_INPUT, fontsize=10.5)

# Encoder + mixup row (trainable) — bottom
box(ax, COL_X,         1.7, 2.1, 0.85,
    "MFA-Conformer\n(encoder)", C_TRAIN, weight="bold", fontsize=10)
box(ax, COL_X + 2.7,   1.7, 2.1, 0.85,
    "mixup (SLERP)\ndifferentiable", C_TRAIN, weight="bold", fontsize=10)

# Backward arrows
def back_arrow(ax, x1, y1, x2, y2, lw=2.0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=15,
        color="#c62828", linewidth=lw, linestyle="--", zorder=4,
    )
    ax.add_patch(a)

# Loss → D output (place L_G label well clear of the box)
ax.text(COL_X + COL_W + 0.7, 9.05, "L_G  (BCE)",
        fontsize=12, weight="bold", color="#c62828", va="center")
back_arrow(ax, COL_X + COL_W + 0.65, 9.05, COL_X + COL_W + 0.02, 9.05)

# Backward chain through D — arrows down the right inside edge
RIGHT = COL_X + COL_W - 0.5
for (y_dst, _, _), (y_src, _, _) in zip(D_BLOCKS[1:], D_BLOCKS[:-1]):
    back_arrow(ax, RIGHT, y_src - 0.02, RIGHT, y_dst + ROW_H + 0.02)
# adapter → embedding-row boundary
back_arrow(ax, RIGHT, 4.28, RIGHT, 3.80)
# split to e and e_syn
back_arrow(ax, RIGHT, 3.65, COL_X + 1.05,         3.78)   # to e
back_arrow(ax, RIGHT, 3.65, COL_X + 2.7 + 1.05,   3.78)   # to e_syn
# e_syn → mixup
back_arrow(ax, COL_X + 2.7 + 1.05, 2.97, COL_X + 2.7 + 1.05, 2.58)
# e → MFA-Conformer
back_arrow(ax, COL_X + 1.05,       2.97, COL_X + 1.05,       2.58)
# mixup → MFA (left arrow)
back_arrow(ax, COL_X + 2.7, 2.12, COL_X + 2.1, 2.12)

# LEFT annotation column — well-separated text blocks, never crossing 7.5 in x
# Annotation A: frozen but grad passes (mid-height, aligned with D backbone)
ax.text(3.7, 6.3,
        "These layers are FROZEN\n(requires_grad = False)\n\n"
        "Parameters are NOT updated by\n"
        "opt_M, but the gradient STILL\n"
        "TRAVERSES the computation\n"
        "graph all the way to the inputs.",
        fontsize=11, color="#c62828", ha="center", va="center", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#fff5f5",
                  edgecolor="#c62828", linewidth=1.6))
ax.annotate("", xy=(COL_X - 0.05, 5.75), xytext=(5.7, 6.2),
            arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.6,
                            linestyle="--"))

# Annotation B: gradient reaches encoder — bottom of left column
ax.text(3.7, 1.95,
        "Gradient arrives here:\n"
        "only MFA-Conformer params\n"
        "and AM-Softmax W are updated\n"
        "by opt_M.step()",
        fontsize=11, color="#2A7", weight="bold",
        ha="center", va="center", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f5fff5",
                  edgecolor="#2A7", linewidth=1.6))
ax.annotate("", xy=(COL_X - 0.05, 2.12), xytext=(5.7, 1.95),
            arrowprops=dict(arrowstyle="->", color="#2A7", lw=1.6))

# Bottom key insight — placed at very bottom with generous margin
ax.text(7.0, 0.45,
        "Key insight:  `requires_grad = False` blocks parameter UPDATES, NOT gradient propagation.\n"
        "This is precisely how a frozen HuBERT still trains the encoder through D.",
        ha="center", fontsize=12, weight="bold", color="#2C2C2C", style="italic",
        linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#fffde7",
                  edgecolor="#e65100", linewidth=1.8))

# Legend
legend_items = [
    mpatches.Patch(facecolor=C_FROZEN, edgecolor="#444", label="Frozen during M step"),
    mpatches.Patch(facecolor=C_TRAIN,  edgecolor="#444", label="Updated by opt_M"),
    mpatches.Patch(facecolor=C_INPUT,  edgecolor="#444", label="Embedding (e, e_syn)"),
    plt.Line2D([], [], color="#c62828", linestyle="--", label="Backward gradient"),
]
ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(0.0, 0.96),
          fontsize=9, framealpha=0.95)

fig.savefig(f"{OUT}/fig_6_4_gradient_flow.png")
fig.savefig(f"{OUT}/fig_6_4_gradient_flow.pdf")
plt.close(fig)
print("  6.4 fig_6_4_gradient_flow.{png,pdf}")


# ============================================================================
# FIG 6.5 — WavLM vs HuBERT — what each pretraining objective gives us
# ============================================================================
# Bigger canvas + bigger fonts for boxes (all 12-13pt body, 16pt headers)
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off")

# Title
fig.suptitle("Why WavLM (vs HuBERT) for the CAARMA critic?",
             fontsize=18, weight="bold", y=0.97)

# Common base block — both share architecture
ax.text(8.0, 9.1, "Both: 24-layer Transformer encoder  ·  hidden 1024  ·  315M params",
        fontsize=13, ha="center", color="#444", style="italic")

# Column widths — give each side 7 inches of breathing room
COL_W = 7.0

# LEFT: HuBERT (x = 0.5 to 7.5)
ax.text(4.0, 8.35, "HuBERT  (Hsu et al. 2021)",
        fontsize=16, weight="bold", ha="center", color="#1976D2")

box(ax, 0.5, 7.0, COL_W, 0.9,
    "Pretrain corpus:\nLibriLight 60kh (English read speech)",
    C_FROZEN, fontsize=12)

box(ax, 0.5, 5.3, COL_W, 1.4,
    "Pretraining objective:\n"
    "Masked prediction of k-means cluster IDs\n"
    "(focus = phoneme-level acoustic content)",
    C_FROZEN, fontsize=12)

box(ax, 0.5, 3.4, COL_W, 1.6,
    "What hidden states encode:\n"
    "  low layers    :  acoustic features\n"
    "  mid layers    :  phonemes\n"
    "  high layers   :  phoneme + some speaker info",
    C_AUDIO, fontsize=11.5)

box(ax, 0.5, 1.9, COL_W, 1.1,
    "CAARMA layer choice  {h7, h9, h11, h12}\n"
    "→  reaches for the SPEAKER-rich mid/high layers",
    C_INPUT, fontsize=12, weight="bold")

ax.text(4.0, 1.3, "SUPERB SID: 81.4%   ASV EER: 5.98%",
        fontsize=12, ha="center", color="#666", weight="bold")

# Divider
ax.plot([8.0, 8.0], [0.5, 8.6], color="#bbb", linestyle="--", linewidth=1.2)

# RIGHT: WavLM (x = 8.5 to 15.5)
ax.text(12.0, 8.35, "WavLM  (Chen et al. 2022)",
        fontsize=16, weight="bold", ha="center", color="#2A7")

box(ax, 8.5, 7.0, COL_W, 0.9,
    "Pretrain corpus:\n94kh (LibriLight + GigaSpeech + VoxPopuli)",
    C_FROZEN, fontsize=12)

box(ax, 8.5, 5.3, COL_W, 1.4,
    "Pretraining objective:\n"
    "Masked prediction + utterance MIXING\n"
    "+ denoising  (designed for speaker tasks)",
    C_TRAIN, fontsize=12, weight="bold")

box(ax, 8.5, 3.4, COL_W, 1.6,
    "What hidden states encode:\n"
    "  low layers    :  acoustic + noise robustness\n"
    "  mid layers    :  phonemes + speaker identity\n"
    "  high layers   :  strong speaker discriminative",
    C_TRAIN, fontsize=11.5)

box(ax, 8.5, 1.9, COL_W, 1.1,
    "Same layer choice  {h7, h9, h11, h12}\n"
    "→  but layers are inherently MORE speaker-discriminative",
    C_INPUT, fontsize=12, weight="bold")

ax.text(12.0, 1.3, "SUPERB SID: 95.5%   ASV EER: 3.77%",
        fontsize=12, ha="center", color="#666", weight="bold")

# Bottom takeaway — empirically corrected
ax.text(8.0, 0.45,
        "Empirical finding (CAARMA, this work):  WavLM 3.71%  vs  HuBERT 3.48%  EER\n"
        "→  The seq_len = 1 input collapses WavLM's sequence-level advantages.",
        ha="center", fontsize=13, weight="bold", color="#2C2C2C", style="italic",
        linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fffde7",
                  edgecolor="#e65100", linewidth=1.8))

fig.savefig(f"{OUT}/fig_6_5_wavlm_vs_hubert.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_6_5_wavlm_vs_hubert.pdf", bbox_inches="tight")
plt.close(fig)
print("  6.5 fig_6_5_wavlm_vs_hubert.{png,pdf}")


# ============================================================================
# FIG 6.6 — Empirical diagnosis: D is too strong, not too weak
# ============================================================================
import csv as _csv
import numpy as np

EXPS = [
    ("03_ddp_baseline",       "Baseline",         "#2A7FBF"),
    ("07_ddp_specaug",        "+ SpecAug",        "#F57C00"),
    ("09_ddp_aug_per_sample", "+ Per-sample aug", "#2A9D8F"),
    ("05_ddp_slerp",          "+ SLERP",          "#7CB342"),
    ("06_ddp_wavlm",          "+ WavLM",          "#8E24AA"),
    ("04_ddp_source_code",    "Source-faithful",  "#C2185B"),
    ("08_ddp_full_aug",       "Full-aug (killed)","#888888"),
    ("01_singlegpu_baseline", "Single-GPU",       "#1565C0"),
]
NASH = 1.386
CSV_DIR = os.path.join(REPO_ROOT, "output", "csv")

# Clean 2-row layout: top row has g/d_loss panels at full half-width each;
# bottom row is the experiment table, full-width. No figure title (the slide
# title carries that), no redundant right-side callout (bullets on the slide
# already say it).
fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.0, 1.0],
    height_ratios=[1.05, 0.95],
    hspace=0.40, wspace=0.22,
    top=0.95, bottom=0.05, left=0.06, right=0.97,
)
ax_g     = fig.add_subplot(gs[0, 0])
ax_d     = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[1, :]);  ax_table.axis("off")

# Plot g_loss and d_loss
rows_for_table = []
for tag, label, color in EXPS:
    rs = list(_csv.DictReader(open(os.path.join(CSV_DIR, tag + ".csv"))))
    ep = np.array([int(r["epoch"]) for r in rs])
    g  = np.array([float(r["g_loss"]) for r in rs])
    d  = np.array([float(r["d_loss"]) for r in rs])
    eers = np.array([float(r["eer"]) for r in rs])
    best_eer = float(eers.min())
    best_ep  = int(ep[int(eers.argmin())])
    last5_g  = float(g[-5:].mean())
    last5_d  = float(d[-5:].mean())
    reaches_nash = (abs(last5_g - NASH) < 0.5) and (abs(last5_d - NASH) < 0.3)
    rows_for_table.append((label, best_eer, best_ep, last5_g, last5_d, reaches_nash, color))

    lw = 2.2 if "killed" not in label and "Source" not in label else 1.4
    alpha = 0.55 if "killed" in label or "Source" in label else 1.0
    ax_g.plot(ep, g, color=color, lw=lw, alpha=alpha, label=label,
              marker="o", markersize=3)
    ax_d.plot(ep, d, color=color, lw=lw, alpha=alpha, label=label,
              marker="o", markersize=3)

# Nash reference lines on both
for ax in (ax_g, ax_d):
    ax.axhline(NASH, color="#c62828", linestyle="--", lw=1.6, zorder=1)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.5, 31.5)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.tick_params(labelsize=10)

# g_loss panel
ax_g.set_ylabel("g_loss", fontsize=12)
ax_g.set_title("Generator loss  (encoder's adversarial loss)",
               fontsize=12, weight="bold")
ax_g.set_ylim(0, 7.2)
ax_g.axhspan(2.5, 7.2, alpha=0.07, color="#c62828")
ax_g.text(16, 6.7, "g_loss >> Nash  →  G is dominated",
          color="#c62828", fontsize=10, ha="center", style="italic", weight="bold")
ax_g.text(1.3, NASH + 0.20, "Nash ≈ 1.39",
          color="#c62828", fontsize=10, weight="bold")
# Single shared legend pinned under the figure, spanning the whole width
ax_g.legend(loc="upper left", bbox_to_anchor=(0.0, -0.20),
            fontsize=9, ncol=4, framealpha=0.95, handlelength=2)

# d_loss panel
ax_d.set_ylabel("d_loss", fontsize=12)
ax_d.set_title("Discriminator loss", fontsize=12, weight="bold")
ax_d.set_ylim(0.5, 1.7)
ax_d.axhspan(0.5, 1.1, alpha=0.07, color="#c62828")
ax_d.text(16, 0.6, "d_loss < Nash  →  D classifies confidently",
          color="#c62828", fontsize=10, ha="center", style="italic", weight="bold")
ax_d.text(1.3, NASH + 0.05, "Nash ≈ 1.39",
          color="#c62828", fontsize=10, weight="bold")

# ─── Table (bottom row, full width) ───────────────────────────────
ax_table.set_xlim(0, 1); ax_table.set_ylim(0, 1)
ax_table.text(0.5, 0.94, "Steady-state values (epoch 26-30 mean)",
              fontsize=13, weight="bold", ha="center", color="#2C2C2C")

# Column x positions — wider spread for a balanced full-width table
COLS = {
    "label":   0.10,
    "best":    0.39,
    "g":       0.51,
    "d":       0.62,
    "nash":    0.74,
}
hdr_y = 0.78
ax_table.text(COLS["label"], hdr_y, "Experiment",   fontsize=11.5, weight="bold")
ax_table.text(COLS["best"],  hdr_y, "best EER",     fontsize=11.5, weight="bold", ha="center")
ax_table.text(COLS["g"],     hdr_y, "g_loss",       fontsize=11.5, weight="bold", ha="center")
ax_table.text(COLS["d"],     hdr_y, "d_loss",       fontsize=11.5, weight="bold", ha="center")
ax_table.text(COLS["nash"],  hdr_y, "near Nash?",   fontsize=11.5, weight="bold", ha="center")
ax_table.plot([COLS["label"] - 0.02, COLS["nash"] + 0.06],
              [hdr_y - 0.05, hdr_y - 0.05],
              color="#888", lw=1.0, transform=ax_table.transAxes)

row_pitch = 0.085
y = hdr_y - 0.10
for label, eer, _bep, g, d, nash_ok, color in rows_for_table:
    ax_table.text(COLS["label"], y, label, fontsize=11, color=color, weight="bold")
    ax_table.text(COLS["best"],  y, f"{eer:.2f}%", fontsize=11, ha="center",
                  color="#2C2C2C")
    ax_table.text(COLS["g"], y, f"{g:.2f}", fontsize=11, ha="center",
                  color="#c62828" if g > 2.0 else "#2A7", weight="bold")
    ax_table.text(COLS["d"], y, f"{d:.2f}", fontsize=11, ha="center",
                  color="#c62828" if d < 1.1 else "#2A7", weight="bold")
    ax_table.text(COLS["nash"], y, "yes" if nash_ok else "no",
                  fontsize=11.5, ha="center", weight="bold",
                  color="#2A7" if nash_ok else "#c62828")
    y -= row_pitch

# No figure suptitle — the slide title carries it.
# No bottom callout — the slide bullets carry the takeaway.
fig.savefig(f"{OUT}/fig_6_6_empirical_diagnosis.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_6_6_empirical_diagnosis.pdf", bbox_inches="tight")
plt.close(fig)
print("  6.6 fig_6_6_empirical_diagnosis.{png,pdf}")


# ============================================================================
# FIG 6.7 — Proposed action: fix the lambda_adv control law
# ============================================================================
# Tall canvas — 16x11 — so each of the three rows gets ~3 inches of height
fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")

fig.suptitle("Proposed next experiment — only one untested code deviation remains",
             fontsize=18, weight="bold", y=0.975)

# ─── Row 1: audit result, three columns ────────────────────────────
ax.text(8.0, 10.05, "What 6+ weeks of audit found",
        fontsize=14, weight="bold", ha="center", color="#444")

box(ax, 0.4, 8.0, 4.8, 1.6,
    "MATCHES source code:\n\n"
    "LR schedule  ·  D head\n"
    "AMP precision  ·  L2-normalize\n"
    "BCE losses",
    C_TRAIN, fontsize=12)
box(ax, 5.6, 8.0, 4.8, 1.6,
    "Engineering tradeoffs:\n\n"
    "HuBERT freeze (DDP deadlock fix)\n"
    "→  single-GPU run shows 3.51%\n"
    "   ruling this out as gap source",
    C_FROZEN, fontsize=12)
box(ax, 10.8, 8.0, 4.8, 1.6,
    "REAL untested deviation:\n\n"
    "λ_adv control law\n"
    "(cap, floor, per-batch reset)",
    "#FFD6D6", fontsize=13, weight="bold")

# ─── Row 2: diff (source vs ours) ──────────────────────────────────
ax.text(8.0, 7.2, "The diff",
        fontsize=14, weight="bold", ha="center", color="#444")

# Source side
ax.text(4.0, 6.75, "Source code (paper)",
        fontsize=13, weight="bold", ha="center", color="#2A7")
ax.text(4.0, 6.3,
        "every G step:\n"
        "    λ_adv = 0.25       # reset!\n"
        "    if ratio > 1.5:  λ = min(λ·1.1, 0.01)\n"
        "    if ratio < 0.5:  λ = max(λ·0.9, 0.0001)\n\n"
        "Effective range:  {0.01, 0.225, 0.25}",
        fontsize=12, ha="center", va="top", family="monospace", color="#2A7",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5fff5",
                  edgecolor="#2A7", linewidth=1.8),
        linespacing=1.4)

# Ours side
ax.text(12.0, 6.75, "Our code (current)",
        fontsize=13, weight="bold", ha="center", color="#c62828")
ax.text(12.0, 6.3,
        "init: λ_adv = 0.25  (NO per-batch reset)\n"
        "    if ratio > 1.5:  λ = min(λ·1.1, 0.5)\n"
        "    if ratio < 0.5:  λ = max(λ·0.9, 0.01)\n\n"
        "Effective range:  [0.01, 0.5] with memory\n"
        "Drift: G crushed → λ decays to floor",
        fontsize=12, ha="center", va="top", family="monospace", color="#c62828",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff5f5",
                  edgecolor="#c62828", linewidth=1.8),
        linespacing=1.4)

# ─── Row 3: experiment + expected outcomes ─────────────────────────
ax.text(8.0, 3.3, "Proposed experiment",
        fontsize=14, weight="bold", ha="center", color="#444")

box(ax, 0.4, 1.2, 7.4, 1.9,
    "exp/fix-lambda-adv-source-control-law-ddp\n"
    "(branched from  exp/innovation-specaug-ddp,  current best 3.27%)\n\n"
    "train.py — 3 changes:\n"
    "   (a) every G step:  self.lambda_adv = 0.25      # reset\n"
    "   (b) cap:    0.5    →   0.01\n"
    "   (c) floor:  0.01   →   0.0001",
    C_TRAIN, fontsize=12)

box(ax, 8.2, 1.2, 7.4, 1.9,
    "Expected outcomes (both informative):\n\n"
    "If EER drops 3.27 → ~3.10:\n"
    "    λ_adv control law IS the gap.\n"
    "    Lock as new default. Write into paper.\n\n"
    "If EER stays ~3.27:\n"
    "    Gap is seed noise.  Run multi-seed next.",
    C_INPUT, fontsize=12)

# Final headline
ax.text(8.0, 0.5,
        "Cost:  1 branch  ·  ~8h on 4-GPU DDP  ·  single decisive experiment",
        ha="center", fontsize=13, weight="bold", color="#2C2C2C", style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffde7",
                  edgecolor="#e65100", linewidth=1.8))

fig.savefig(f"{OUT}/fig_6_7_proposed_action.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_6_7_proposed_action.pdf", bbox_inches="tight")
plt.close(fig)
print("  6.7 fig_6_7_proposed_action.{png,pdf}")


print(f"\nAll 8 slide-ready figures written to {OUT}/")
