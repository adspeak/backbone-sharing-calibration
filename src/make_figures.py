#!/usr/bin/env python3
"""
Generate the manuscript figures.

Specification (Springer/Elsevier house style; most journals are more permissive):
  - Column widths: 84 mm single, 174 mm double
  - Vector PDF, plus a 600 dpi TIFF for production
  - Sans-serif, 8 pt base, embedded as TrueType (fonttype=42). Type 3 fonts are
    rejected by most publishers' preflight checks.
  - Legible in greyscale: line styles and marker shapes differ, so colour is
    never the only channel carrying information
  - Okabe-Ito colour-blind-safe palette

Note on physical size: `savefig.bbox = "tight"` trims surrounding whitespace, so
the saved figure is slightly narrower than the nominal width set here. The
manuscript scales every figure through \\includegraphics anyway. Final sizing for
production should be done once the target journal's template is fixed; the widths
below are the ones that produced the submitted figures and should not be changed
without regenerating all four and re-checking the manuscript.

Usage:
    cd src
    python make_figures.py

Reads the cached records resolved by paths.py (the copy bundled under
data/raw_records/ when present), and writes to
<workspace>/calibration_results/figures/ as .pdf + .tif.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sst

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RECORDS_DIR, CALIB  # noqa: E402

R = Path(RECORDS_DIR)
OUT = Path(CALIB) / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
W1, W2 = 84 * MM, 174 * MM          # single / double column
DPI = 600

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "pdf.fonttype": 42,             # TrueType, not Type 3
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Okabe-Ito colour-blind-safe palette
CB = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "purple": "#CC79A7", "orange": "#E69F00", "sky": "#56B4E9",
      "yellow": "#F0E442", "black": "#000000"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    try:
        fig.savefig(OUT / f"{name}.tif", dpi=DPI,
                    pil_kwargs={"compression": "tiff_lzw"})
    except Exception as e:
        fig.savefig(OUT / f"{name}.png", dpi=DPI)
        print(f"    [note] TIFF output failed ({type(e).__name__}); "
              f"wrote PNG at {DPI} dpi instead")
    plt.close(fig)


# ----------------------------------------------------------------- data
def dece(c, y, M=10):
    """Detection ECE over M equal-width confidence bins."""
    N = len(c)
    e = 0.0
    for m in range(M):
        lo, hi = m / M, (m + 1) / M
        k = (c >= lo) & (c <= hi) if m == M - 1 else (c >= lo) & (c < hi)
        if k.sum():
            e += k.sum() / N * abs(c[k].mean() - y[k].mean())
    return e


rec = {}
for f in sorted(R.glob("*.npz")):
    m = re.match(r"kvasir_(.+)_seed(\d+)_records\.npz", f.name)
    if not m:
        continue
    z = np.load(f)
    rec[(m.group(1), int(m.group(2)))] = (z["confidence"].astype(float),
                                          z["is_tp"].astype(float),
                                          z["image_idx"])
if not rec:
    raise SystemExit(f"[fatal] no record files found under {R}")


def top1(cfg, s):
    """Highest-confidence detection per image: the primary prediction set."""
    a, b, i = rec[(cfg, s)]
    C, Y = [], []
    for im in np.unique(i):
        k = i == im
        o = np.argmax(a[k])
        C.append(a[k][o])
        Y.append(b[k][o])
    return np.array(C), np.array(Y)


S8 = list(range(42, 50))
S6 = list(range(42, 48))
EIGHT = ("single_task", "A_independent", "B_shared",
         "gapfc_A_independent", "gapfc_B_shared")
seeds_of = lambda c: S8 if c in EIGHT else S6

DEPTH = [("A_independent", "A ($k$=0)"), ("C_shallow", "C-shallow ($k$=5)"),
         ("C_deep", "C-deep ($k$=9)"), ("B_shared", "B ($k$=11)")]


# --------------------------------------------- Figure: reliability (double)
fig, axes = plt.subplots(2, 4, figsize=(W2, 78 * MM), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12,
                                      "wspace": 0.22})
x = np.arange(10) / 10 + 0.05
for j, (cfg, lab) in enumerate(DEPTH):
    sd = seeds_of(cfg)
    conf = np.full(10, np.nan); acc = np.full(10, np.nan); cnt = np.zeros(10)
    cs, as_, ns = np.zeros(10), np.zeros(10), np.zeros(10)
    for s in sd:
        c, y = top1(cfg, s)
        for m in range(10):
            lo, hi = m / 10, (m + 1) / 10
            k = (c >= lo) & (c <= hi) if m == 9 else (c >= lo) & (c < hi)
            if k.sum():
                cs[m] += c[k].mean(); as_[m] += y[k].mean(); ns[m] += 1
            cnt[m] += k.sum()
    ok = ns > 0
    conf[ok] = cs[ok] / ns[ok]; acc[ok] = as_[ok] / ns[ok]; cnt /= len(sd)

    ax = axes[0, j]
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), c="0.35", lw=0.8, zorder=1)
    ax.bar(x, np.abs(conf - acc), bottom=np.fmin(conf, acc), width=0.085,
           facecolor="0.82", edgecolor="0.45", lw=0.4, zorder=2)
    ax.plot(x, acc, marker="o", c=CB["vermillion"], mfc="w", mew=0.9, zorder=3)
    d = np.mean([dece(*top1(cfg, s)) for s in sd])
    ax.set_title(f"{lab}\nD-ECE = {d:.3f}", pad=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    if j == 0:
        ax.set_ylabel("Observed precision")
    else:
        ax.set_yticklabels([])

    ax2 = axes[1, j]
    ax2.bar(x, np.maximum(cnt, 0.5), width=0.085, facecolor="0.55",
            edgecolor="none")
    ax2.set_yscale("log"); ax2.set_xlim(0, 1); ax2.set_ylim(0.5, 300)
    ax2.set_xlabel("Confidence")
    if j == 0:
        ax2.set_ylabel("Count")
    else:
        ax2.set_yticklabels([])
h = [plt.Line2D([], [], ls=(0, (4, 3)), c="0.35", lw=0.8),
     plt.Line2D([], [], marker="o", c=CB["vermillion"], mfc="w", mew=0.9),
     plt.Rectangle((0, 0), 1, 1, fc="0.82", ec="0.45", lw=0.4)]
axes[0, 0].legend(h, ["Perfect calibration", "Observed", "Gap"],
                  loc="upper left", frameon=False, handlelength=1.6)
save(fig, "fig_reliability")

# ------------------------------------- Figure: confidence histogram (single)
fig, ax = plt.subplots(figsize=(W1, 55 * MM))
bins = np.linspace(0, 1, 26)
for cfg, lab, col, hs in [("A_independent", "Independent ($k$=0)", CB["blue"], "///"),
                          ("B_shared", "Fully shared ($k$=11)", CB["vermillion"], "\\\\\\")]:
    allc = np.concatenate([top1(cfg, s)[0] for s in S8])
    ax.hist(allc, bins=bins, density=True, histtype="stepfilled",
            facecolor="none", edgecolor=col, hatch=hs, lw=0.8, label=lab)
ax.set_xlabel("Confidence of the top-ranked detection per image")
ax.set_ylabel("Density")
ax.set_yscale("log"); ax.set_xlim(0, 1)
ax.legend(frameon=False, loc="upper center")
save(fig, "fig_confidence_hist")

# -------------------------------------------- Figure: D-ECE against N (129 mm)
#
# This panel covers the fifty JOINTLY TRAINED models only, matching the figure
# caption and the correlation reported in the abstract (r = -0.74). The
# detection-only baseline is not jointly trained and is deliberately absent from
# both the scatter and the fit: including its eight seeds would silently change
# the reported correlation once its records are present in the cache.
fig, ax = plt.subplots(figsize=(129 * MM, 72 * MM))
style = {"A_independent": ("Independent", CB["blue"], "o"),
         "C_shallow": ("C-shallow", CB["sky"], "^"),
         "C_deep": ("C-deep", CB["purple"], "v"),
         "B_shared": ("Fully shared", CB["vermillion"], "s"),
         "gapfc_A_independent": ("GAP+FC indep.", CB["orange"], "D"),
         "gapfc_B_shared": ("GAP+FC shared", CB["green"], "P"),
         "F_frozen": ("Frozen backbone", "0.45", "X")}
aN, aD = [], []
for cfg, (lab, col, mk) in style.items():
    sd = seeds_of(cfg)
    if (cfg, sd[0]) not in rec:
        print(f"    [warn] no records for {cfg}; omitted from the D-ECE/N panel")
        continue
    N = [len(rec[(cfg, s)][0]) for s in sd]
    D = [dece(rec[(cfg, s)][0], rec[(cfg, s)][1]) for s in sd]
    ax.scatter(N, D, c=col, marker=mk, s=22, lw=0.5, edgecolors="w",
               label=lab, zorder=3)
    aN += N; aD += D
if len(aN) != 50:
    print(f"    [warn] the D-ECE/N panel covers {len(aN)} models, not 50; "
          f"the correlation will not match the manuscript")
r, p = sst.pearsonr(aN, aD)
b, a0 = np.polyfit(aN, aD, 1)
xs = np.linspace(min(aN), max(aN), 50)
ax.plot(xs, a0 + b * xs, ls=(0, (4, 3)), c="0.3", lw=0.9, zorder=2)
ax.set_xlabel("Predictions retained per model ($N$)")
ax.set_ylabel("D-ECE (permissive protocol)")
p_txt = "$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"
ax.text(0.97, 0.95, f"$r$ = {r:.2f}, {p_txt}", transform=ax.transAxes,
        ha="right", va="top")
ax.legend(frameon=False, ncol=2, loc="lower left", handletextpad=0.3,
          columnspacing=0.8)
save(fig, "fig_dece_vs_n")
print(f"    D-ECE/N panel: n = {len(aN)} models, r = {r:.4f}, p = {p:.3g}")

# --------------------------------------- Figure: D-ECE against depth (single)
fig, ax = plt.subplots(figsize=(W1, 58 * MM))
ks = [0, 5, 9, 11]
mu, sd_ = [], []
for cfg, _ in DEPTH:
    v = [dece(*top1(cfg, s)) for s in seeds_of(cfg)]
    mu.append(np.mean(v)); sd_.append(np.std(v))
ax.errorbar(ks, mu, yerr=sd_, fmt="o-", capsize=2.5, capthick=0.7, lw=1.0,
            c=CB["vermillion"], mfc="w", mew=0.9)
ax.set_xlabel("Sharing depth $k$ (backbone modules shared)")
ax.set_ylabel("D-ECE")
ax.set_xticks(ks)
ax.set_xlim(-0.8, 11.8)
save(fig, "fig_dece_vs_depth")
print("    depth panel: " + ", ".join(f"k={k}: {m:.4f}" for k, m in zip(ks, mu)))

print(f"\nWritten to {OUT} (vector PDF + 600 dpi TIFF per figure):")
for f in sorted(os.listdir(OUT)):
    print(f"   {f:32s} {os.path.getsize(OUT / f) // 1024:5d} KB")
print("\nCheck that fonts are embedded (Type 3 is rejected at preflight):")
print(f"   pdffonts {OUT / 'fig_reliability.pdf'}")
