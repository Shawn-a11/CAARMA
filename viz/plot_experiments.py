#!/usr/bin/env python3
"""Consolidated experiment figures (merged from three one-off plot scripts).

Subcommands:
    python viz/plot_experiments.py all       # regenerate every figure
    python viz/plot_experiments.py specaug   # 11_ddp_baseline_vs_specaug_vs_fullaug
    python viz/plot_experiments.py ladder    # 12_plain_baseline_vs_caarma_ladder
    python viz/plot_experiments.py mlpd      # viz/figures/mlp_discriminator_training

Reads epoch/EER CSVs from output/csv/ and writes figures to output/plots/ and
viz/figures/. Paths are resolved relative to the repo root, so it runs from any
cwd. Merged from _plot_baseline_specaug_fullaug.py, _plot_plain_baseline_vs_caarma.py
and plot_mlp_disc_eer.py.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def p(*parts):
    return os.path.join(ROOT, *parts)


LADDER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.edgecolor": "#B0B0B0",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
}


def load(path):
    rows = list(csv.DictReader(open(path)))
    return (np.array([int(r["epoch"]) for r in rows]),
            np.array([float(r["eer"]) for r in rows]))


def best(x, y):
    i = int(np.argmin(y))
    return int(x[i]), float(y[i])


# ──────────────────────────────────────────────────────────────────────────
def plot_specaug_fullaug():
    """3-way: baseline vs SpecAug-only vs naive Full-aug (figure 11)."""
    plt.rcParams.update(LADDER_STYLE)
    baseline = load(p("output/csv/03_ddp_baseline.csv"))
    specaug = load(p("output/csv/07_ddp_specaug.csv"))
    fullaug = load(p("output/csv/08_ddp_full_aug.csv"))  # 26 epochs (killed)

    PAPER_SOTA = 3.09
    C_BASELINE, C_SPECAUG, C_FULLAUG, C_SOTA = "#2A7FBF", "#F57C00", "#888888", "#E76F51"
    START = 5
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    series = [
        ("baseline", baseline, C_BASELINE, "Baseline (Right Algorithm 2)"),
        ("specaug", specaug, C_SPECAUG, "+ SpecAugment (mel-spec freq/time mask)"),
        ("fullaug", fullaug, C_FULLAUG, "+ Naive 100% noise+reverb stacking (killed at ep26)"),
    ]
    for name, (x, y), color, label in series:
        mask = x >= START
        alpha = 1.0 if name != "fullaug" else 0.85
        ax.plot(x[mask], y[mask], marker="o", color=color, markerfacecolor="white",
                markeredgewidth=1.5, label=label, zorder=3, alpha=alpha)
    ax.axhline(PAPER_SOTA, ls="--", color=C_SOTA, lw=2.0,
               label=f"Paper SOTA ({PAPER_SOTA:.2f}%)", zorder=2)
    best_info = []
    for name, (x, y), color, _ in series:
        be, bv = best(x, y)
        ax.scatter([be], [bv], marker="*", s=520, color=color,
                   edgecolors="white", linewidths=1.6, zorder=6)
        best_info.append((name, be, bv, color))
    label_specs = {
        "baseline": dict(xytext=(11.5, 2.95), ha="right"),
        "specaug": dict(xytext=(14, 2.85), ha="right"),
        "fullaug": dict(xytext=(15, 5.40), ha="center"),
    }
    pretty = {"baseline": "Baseline", "specaug": "+ SpecAugment", "fullaug": "+ Full-Aug (naive)"}
    for name, be, bv, color in best_info:
        spec = label_specs[name]
        ax.annotate(f"{pretty[name]} best\n{bv:.2f}% @ ep {be}", xy=(be, bv),
                    xytext=spec["xytext"], fontsize=10, color=color, ha=spec["ha"],
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=color, linewidth=1.4),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                                    connectionstyle="arc3,rad=0.2"), zorder=7)
    ax.annotate("", xy=(29.5, PAPER_SOTA), xytext=(29.5, 3.27),
                arrowprops=dict(arrowstyle="<->", color=C_SPECAUG, lw=1.3), zorder=4)
    ax.text(29.7, (PAPER_SOTA + 3.27) / 2, "Δ 0.18%", color=C_SPECAUG,
            fontsize=9, fontweight="bold", va="center", ha="left")
    ax.annotate("", xy=(8.5, 3.27), xytext=(8.5, 3.48),
                arrowprops=dict(arrowstyle="<->", color=C_SPECAUG, lw=1.3), zorder=4)
    ax.text(8.3, (3.27 + 3.48) / 2, "−0.21%\n(54% of gap)", color=C_SPECAUG,
            fontsize=9, fontweight="bold", va="center", ha="right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("EER (%)  ↓ lower is better")
    ax.set_title("DDP Baseline vs SpecAugment vs Naive Full-Augmentation (Zoomed from epoch 5)")
    ax.set_xticks(np.arange(START, 31, 2))
    ax.set_xlim(START - 0.5, 30.5)
    ax.set_ylim(2.7, 7.5)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.text(0.01, 0.02,
            "Curves shown from epoch 5 onward. Full-Aug was killed at epoch 26 due to training divergence.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#888888")
    fig.tight_layout()
    out = p("output/plots/11_ddp_baseline_vs_specaug_vs_fullaug")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)
    print("Saved: output/plots/11_ddp_baseline_vs_specaug_vs_fullaug.{png,pdf}")
    for name, (x, y), _, _ in series:
        be, bv = best(x, y)
        print(f"  {name:10s}: best {bv:.2f}% @ ep {be:>2d}   "
              f"vs baseline 3.48%: {bv - 3.48:+.2f}%   vs paper 3.09%: {bv - PAPER_SOTA:+.2f}%")


# ──────────────────────────────────────────────────────────────────────────
def plot_plain_vs_caarma():
    """Ablation ladder: plain MFA vs CAARMA components vs paper (figure 12)."""
    plt.rcParams.update(LADDER_STYLE)
    plain = load(p("output/csv/12_plain_baseline_no_gan.csv"))
    caarma = load(p("output/csv/03_ddp_baseline.csv"))
    specaug = load(p("output/csv/07_ddp_specaug.csv"))

    PAPER_BASELINE, PAPER_SOTA = 3.33, 3.09
    C_PLAIN, C_CAARMA, C_SPECAUG, C_PBASE, C_SOTA = \
        "#6A4C93", "#2A7FBF", "#F57C00", "#9E9E9E", "#E76F51"
    START = 5
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    series = [
        ("plain", plain, C_PLAIN, "Plain MFA baseline (no GAN, no aug)"),
        ("caarma", caarma, C_CAARMA, "+ CAARMA GAN (frozen HuBERT-D)"),
        ("specaug", specaug, C_SPECAUG, "+ CAARMA GAN + SpecAugment"),
    ]
    for name, (x, y), color, label in series:
        mask = x >= START
        ax.plot(x[mask], y[mask], marker="o", color=color, markerfacecolor="white",
                markeredgewidth=1.5, label=label, zorder=3)
    ax.axhline(PAPER_BASELINE, ls=":", color=C_PBASE, lw=2.0,
               label=f"Paper plain baseline ({PAPER_BASELINE:.2f}%)", zorder=2)
    ax.axhline(PAPER_SOTA, ls="--", color=C_SOTA, lw=2.0,
               label=f"Paper SOTA / full CAARMA ({PAPER_SOTA:.2f}%)", zorder=2)
    best_info = []
    for name, (x, y), color, _ in series:
        be, bv = best(x, y)
        ax.scatter([be], [bv], marker="*", s=520, color=color,
                   edgecolors="white", linewidths=1.6, zorder=6)
        best_info.append((name, be, bv, color))
    label_specs = {
        "plain": dict(xytext=(20.0, 4.15), ha="left"),
        "caarma": dict(xytext=(22.5, 3.70), ha="left"),
        "specaug": dict(xytext=(24.5, 2.95), ha="left"),
    }
    pretty = {"plain": "Plain baseline", "caarma": "+ GAN", "specaug": "+ GAN + SpecAug"}
    for name, be, bv, color in best_info:
        spec = label_specs[name]
        ax.annotate(f"{pretty[name]}\n{bv:.2f}% @ ep {be}", xy=(be, bv),
                    xytext=spec["xytext"], fontsize=10, color=color, ha=spec["ha"],
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=color, linewidth=1.4),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                                    connectionstyle="arc3,rad=0.2"), zorder=7)
    ax.annotate("", xy=(6.2, PAPER_BASELINE), xytext=(6.2, 3.61),
                arrowprops=dict(arrowstyle="<->", color=C_PLAIN, lw=1.3), zorder=4)
    ax.text(6.0, (PAPER_BASELINE + 3.61) / 2, "recipe offset\nΔ0.28%", color=C_PLAIN,
            fontsize=9, fontweight="bold", va="center", ha="right")
    ax.annotate("", xy=(9.0, 3.48), xytext=(9.0, 3.61),
                arrowprops=dict(arrowstyle="<->", color=C_CAARMA, lw=1.3), zorder=4)
    ax.text(9.2, (3.48 + 3.61) / 2, "frozen GAN\n−0.13%", color=C_CAARMA,
            fontsize=9, fontweight="bold", va="center", ha="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("EER (%)   ↓ lower is better")
    ax.set_title("Plain MFA baseline vs CAARMA components (VoxCeleb1-O, zoomed from epoch 5)")
    ax.set_xticks(np.arange(START, 31, 2))
    ax.set_xlim(START - 0.5, 30.5)
    ax.set_ylim(2.8, 5.0)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.text(0.01, 0.02,
            "Plain baseline overfits after ep15 (train acc 100%, EER rebounds): "
            "CAARMA's synthetic classes act partly as regularisation.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#888888")
    fig.tight_layout()
    out = p("output/plots/12_plain_baseline_vs_caarma_ladder")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)
    print("Saved: output/plots/12_plain_baseline_vs_caarma_ladder.{png,pdf}")
    print(f"Plain baseline best : {best(*plain)[1]:.2f}% @ ep {best(*plain)[0]}")
    print(f"+ GAN (03) best     : {best(*caarma)[1]:.2f}% @ ep {best(*caarma)[0]}")
    print(f"+ GAN+SpecAug (07)  : {best(*specaug)[1]:.2f}% @ ep {best(*specaug)[0]}")


# ──────────────────────────────────────────────────────────────────────────
MLPD_DATA = """epoch,am_loss,am_loss_syn,acc,g_loss,d_loss,total_loss,eer,mindcf_2,mindcf_3
1,5.74,3.0,19.0,1.53,1.31,6.51,9.36,0.6335,0.7577
2,4.68,2.83,33.3,1.54,1.33,5.46,6.68,0.5612,0.6682
3,2.29,0.906,38.1,1.57,1.43,3.07,5.54,0.5018,0.667
4,2.04,0.766,57.1,1.6,1.25,2.84,5.19,0.4677,0.6359
5,1.44,0.9,76.2,1.63,1.2,2.03,4.32,0.4145,0.5195
6,1.34,0.809,71.4,1.73,1.33,1.36,4.15,0.3819,0.53
7,0.912,1.03,76.2,2.07,1.21,0.933,4.05,0.3874,0.5023
8,0.894,1.09,76.2,2.22,1.16,0.917,4.0,0.3775,0.4983
9,0.51,1.55,95.2,2.25,1.11,0.534,3.76,0.3697,0.4176
10,0.543,0.543,85.7,2.82,1.07,0.572,3.57,0.3413,0.4108
11,0.236,0.905,100.0,2.25,1.21,0.259,3.59,0.3544,0.4036
12,0.943,0.251,76.2,2.07,1.25,0.964,3.56,0.3648,0.4045
13,0.44,0.346,95.2,2.99,1.08,0.47,3.57,0.3634,0.4454
14,0.205,0.515,95.2,2.08,1.25,0.226,3.48,0.3475,0.4066
15,0.563,0.899,85.7,2.41,1.13,0.588,3.45,0.3472,0.4312
16,0.765,1.23,81.0,1.91,1.27,0.785,3.43,0.3725,0.4785
17,0.437,1.35,90.5,2.71,1.13,0.465,3.41,0.3562,0.4494
18,0.834,0.645,81.0,2.57,1.25,0.861,3.39,0.3567,0.4279
19,0.426,0.391,90.5,2.82,1.07,0.455,3.52,0.3529,0.4531
20,0.154,0.363,100.0,3.15,0.968,0.185,3.52,0.3554,0.447
21,0.554,1.37,85.7,2.22,1.14,0.578,3.44,0.3426,0.4458
22,0.444,0.437,85.7,2.27,1.09,0.467,3.37,0.3506,0.4791
23,0.399,0.915,90.5,2.2,1.12,0.422,3.45,0.3459,0.4652
24,0.387,0.424,95.2,2.62,1.07,0.413,3.44,0.3374,0.4986
25,0.415,0.264,95.2,2.65,1.21,0.442,3.44,0.345,0.4772
26,0.18,0.316,100.0,3.01,1.08,0.21,3.38,0.3381,0.4806
27,0.196,0.183,100.0,2.01,1.17,0.217,3.46,0.3541,0.4913
28,0.191,0.886,100.0,2.24,1.12,0.214,3.4,0.3495,0.5029
29,0.325,0.508,95.2,2.33,1.27,0.348,3.44,0.3398,0.4931
30,0.295,0.523,96.0,2.58,1.16,0.321,,,"""


def plot_mlp_disc_eer():
    """MLP-discriminator 2x2 training dashboard with best-EER annotation."""
    rows = []
    for line in MLPD_DATA.strip().split("\n")[1:]:
        rows.append([0.0 if f == "" else float(f) for f in line.split(",")])
    epochs = [int(r[0]) for r in rows]
    am_loss = [r[1] for r in rows]; am_loss_syn = [r[2] for r in rows]
    acc = [r[3] for r in rows]; g_loss = [r[4] for r in rows]
    d_loss = [r[5] for r in rows]; total_loss = [r[6] for r in rows]
    eer = [r[7] if r[7] != 0 else None for r in rows[:29]]
    mindcf_2 = [r[8] if r[8] != 0 else None for r in rows[:29]]
    mindcf_3 = [r[9] if r[9] != 0 else None for r in rows[:29]]
    epochs_29 = epochs[:29]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("MLP Discriminator Training Metrics (DDP Baseline Ablation)",
                 fontsize=15, fontweight="bold", y=0.995)

    ax = axes[0, 0]
    ax.plot(epochs, am_loss, "o-", label="AM Loss", linewidth=2.5, markersize=5, color="#1f77b4")
    ax.plot(epochs, am_loss_syn, "s-", label="AM Loss (Syn)", linewidth=2.5, markersize=5, color="#ff7f0e")
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold"); ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax.set_title("Augmented Mixup Losses", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10); ax.grid(True, alpha=0.3, linestyle="--"); ax.set_xlim([0, 31])

    ax = axes[0, 1]
    ax.plot(epochs, g_loss, "o-", label="Generator Loss", linewidth=2.5, markersize=5, color="#2ca02c")
    ax.plot(epochs, d_loss, "s-", label="Discriminator Loss", linewidth=2.5, markersize=5, color="#d62728")
    ax.plot(epochs, total_loss, "^-", label="Total Loss", linewidth=2.5, markersize=5, color="#9467bd")
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold"); ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax.set_title("GAN Losses", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10); ax.grid(True, alpha=0.3, linestyle="--"); ax.set_xlim([0, 31])

    ax = axes[1, 0]
    ax.plot(epochs, acc, "o-", linewidth=2.5, markersize=5, color="#17becf")
    ax.fill_between(epochs, 0, acc, alpha=0.2, color="#17becf")
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold"); ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Training Accuracy", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--"); ax.set_ylim([0, 105]); ax.set_xlim([0, 31])

    ax = axes[1, 1]
    ax.plot(epochs_29, eer, "o-", label="EER", linewidth=2.5, markersize=5, color="#e377c2")
    ax.plot(epochs_29, mindcf_2, "s-", label="minDCF (P=0.01)", linewidth=2.5, markersize=5, color="#7f7f7f")
    ax.plot(epochs_29, mindcf_3, "^-", label="minDCF (P=0.001)", linewidth=2.5, markersize=5, color="#bcbd22")
    ax.scatter([22], [3.37], s=350, color="red", marker="*", zorder=5, edgecolors="darkred", linewidth=2.5)
    ax.annotate("BEST EER\n3.37%\nEpoch 22", xy=(22, 3.37), xytext=(22 - 4.5, 3.37 + 1.2),
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="darkred", alpha=0.85, edgecolor="red", linewidth=2.5),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=2.5, connectionstyle="arc3,rad=0.3"))
    ax.scatter([26], [3.38], s=250, color="green", marker="D", zorder=5, edgecolors="darkgreen", linewidth=2)
    ax.annotate("Optimal\n3.38%\nEp 26", xy=(26, 3.38), xytext=(26 + 3, 3.38 - 1.0),
                fontsize=9, fontweight="bold", color="darkgreen",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8, edgecolor="darkgreen", linewidth=1.5),
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5))
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold"); ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Speaker Verification Metrics (Lower is Better)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10); ax.grid(True, alpha=0.3, linestyle="--"); ax.set_xlim([0, 31])

    plt.tight_layout()
    os.makedirs(p("viz/figures"), exist_ok=True)
    plt.savefig(p("viz/figures/mlp_discriminator_training.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: viz/figures/mlp_discriminator_training.png")
    print("   Best EER 3.37% @ ep22 | Optimal 3.38% @ ep26 | 9.36% -> 3.37% (-64%)")


FIGS = {"specaug": plot_specaug_fullaug, "ladder": plot_plain_vs_caarma, "mlpd": plot_mlp_disc_eer}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = FIGS.values() if which == "all" else [FIGS[which]] if which in FIGS else None
    if targets is None:
        sys.exit(f"unknown target '{which}'. choose from: all, {', '.join(FIGS)}")
    for fn in targets:
        fn()


if __name__ == "__main__":
    main()
