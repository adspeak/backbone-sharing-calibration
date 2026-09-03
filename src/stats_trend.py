"""
Linear trend test across sharing depths, on a unified seed basis.

Only seeds present at all four depths can contribute a four-point regression, so
this script restricts the analysis to those six seeds for every metric. Mixing
four-point slopes with two-point slopes -- which is what happens if seeds missing
the intermediate depths are allowed through -- combines two different quantities
in one test.

Reads the cached records; no inference, no GPU.

Usage:
    python stats_trend.py
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from paths import CALIB, RECORDS_DIR

OUTPUT_DIR = CALIB

SEEDS = [42, 43, 44, 45, 46, 47]      # seeds present at all four depths
DEPTHS = [0, 5, 9, 11]
CONFIGS = ["A_independent", "C_shallow", "C_deep", "B_shared"]
N_BINS, EPS = 10, 1e-12


def load(run):
    p = RECORDS_DIR / f"kvasir_{run}_records.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    return d["confidence"].astype(float), d["is_tp"].astype(int)


def dece(c, l, nb=N_BINS):
    n = len(c); e = 0.0
    edges = np.linspace(0, 1, nb + 1)
    for i in range(nb):
        lo, hi = edges[i], edges[i+1]
        m = (c >= lo) & (c <= hi) if i == nb-1 else (c >= lo) & (c < hi)
        if m.sum() == 0: continue
        e += (m.sum()/n) * abs(c[m].mean() - l[m].mean())
    return e


def aece(c, l, nb=N_BINS):
    n = len(c); o = np.argsort(c); cs, ls = c[o], l[o]; e = 0.0
    for idx in np.array_split(np.arange(n), nb):
        if len(idx) == 0: continue
        e += (len(idx)/n) * abs(cs[idx].mean() - ls[idx].mean())
    return e


def brier(c, l):
    return np.mean((c - l) ** 2)


def nll(c, l):
    p = np.clip(c, EPS, 1-EPS)
    return -np.mean(l*np.log(p) + (1-l)*np.log(1-p))


METRICS = [("D-ECE", dece), ("Adaptive ECE", aece), ("Brier", brier), ("NLL", nll)]

print("=" * 64)
print(f"Trend test on a unified basis: {len(SEEDS)} seeds present at all four depths")
print("=" * 64)

for mname, mfn in METRICS:
    slopes = []
    for s in SEEDS:
        ys = []
        for cfg in CONFIGS:
            c, l = load(f"{cfg}_seed{s}")
            if c is None:
                ys = None
                break
            ys.append(mfn(c, l))
        if ys is None:
            print(f"  [warn] seed{s} incomplete data, skipping")
            continue
        slope, _, _, _, _ = scipy_stats.linregress(DEPTHS, ys)
        slopes.append(slope)

    slopes = np.array(slopes)
    t, p = scipy_stats.ttest_1samp(slopes, 0.0)
    print(f"\n{mname}:")
    print(f"  per-seed slopes: {np.round(slopes, 5).tolist()}")
    print(f"  mean slope = {slopes.mean():.5f} ± {slopes.std(ddof=1):.5f} per module")
    print(f"  t = {t:.3f}, p = {p:.4f}  (n={len(slopes)} seeds)")
    print(f"  {'sig.' if p < 0.05 else 'n.s.'}")

print("\n" + "=" * 64)
print("These are the figures to report: all four metrics on the same 6-seed basis")
print("=" * 64)
