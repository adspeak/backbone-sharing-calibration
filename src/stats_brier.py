"""
Statistical tests on the Brier score, matching the D-ECE analysis exactly.
=======================================================
Repeats every D-ECE test on a second, independent calibration metric, so the
conclusions can be checked for dependence on the choice of metric.

Reads the cached records written by compute_brier_reliability.py; no inference,
no GPU.

Tests, one-to-one with those in stats_dece.py:
  1. independent vs fully shared, paired t-test with Cohen's d_z
  2. linear trend across the four sharing depths
  3. gapfc_A vs gapfc_B, paired t-test
  4. interaction (primary test): the sharing-induced shift under the LAG-LGFF
     head against the corresponding shift under the GAP+FC head
  5. frozen-backbone control vs fully shared

Usage:
    python stats_brier.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from paths import CALIB, RECORDS_DIR

OUTPUT_DIR = CALIB

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]


def load_brier_by_seed(config_prefix, dataset="kvasir"):
    """
    Read back the per-seed Brier scores for one configuration.

    config_prefix is e.g. "A_independent"; run names are built by appending
    "_seed42" and so on. Returns a list in seed order, with None where the cache
    file is missing.
    """
    briers = []
    for seed in SEEDS:
        run_name = f"{config_prefix}_seed{seed}"
        path = RECORDS_DIR / f"{dataset}_{run_name}_records.npz"
        if not path.exists():
            print(f"  [warn] missing cache {path.name}")
            briers.append(None)
            continue
        data = np.load(path)
        confs = data["confidence"]
        labels = data["is_tp"]
        briers.append(float(np.mean((confs - labels) ** 2)))
    return briers


def paired_test(name_a, vals_a, name_b, vals_b):
    """Paired t-test with Cohen's d_z (mean difference over its SD)."""
    pairs = [(a, b) for a, b in zip(vals_a, vals_b) if a is not None and b is not None]
    if len(pairs) < 2:
        print(f"  {name_a} vs {name_b}: too few valid pairs, skipping")
        return
    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])
    diffs = b_arr - a_arr

    t_stat, p_val = scipy_stats.ttest_rel(b_arr, a_arr)
    d = float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else float("nan")

    rel_change = (np.mean(b_arr) - np.mean(a_arr)) / np.mean(a_arr) * 100

    print(f"  {name_a} = {np.mean(a_arr):.4f} ± {np.std(a_arr):.4f}")
    print(f"  {name_b} = {np.mean(b_arr):.4f} ± {np.std(b_arr):.4f}")
    print(f"  difference (B-A) = {np.mean(diffs):.4f}, relative change = {rel_change:+.1f}%")
    print(f"  paired t-test: t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f} (n={len(pairs)} seeds)")
    print(f"  {'>>> significant (p<0.05)' if p_val < 0.05 else '>>> not significant'}")


def trend_test(config_names, config_vals, depths):
    """Linear trend across sharing depths: fit a slope per seed, test against zero."""
    slopes = []
    n_seeds = len(config_vals[0])
    for si in range(n_seeds):
        ys, xs = [], []
        for ci, vals in enumerate(config_vals):
            if vals[si] is not None:
                ys.append(vals[si])
                xs.append(depths[ci])
        if len(ys) == len(config_vals):  # only seeds present at every depth
            slope, _, _, _, _ = scipy_stats.linregress(xs, ys)
            slopes.append(slope)

    if len(slopes) < 2:
        print("  too few valid slopes, skipping the trend test")
        return
    t_stat, p_val = scipy_stats.ttest_1samp(slopes, 0.0)
    print(f"  configuration means: " + ", ".join(
        f"{n}={np.mean([v for v in vals if v is not None]):.4f}"
        for n, vals in zip(config_names, config_vals)))
    print(f"  mean per-seed slope = {np.mean(slopes):.6f} ± {np.std(slopes):.6f}")
    print(f"  trend test (slope vs 0): t={t_stat:.3f}, p={p_val:.4f} (n={len(slopes)} seeds)")
    print(f"  {'>>> significant trend' if p_val < 0.05 else '>>> trend not significant'}")


def interaction_test(lag_a, lag_b, gapfc_a, gapfc_b):
    """
    Primary interaction test.

    Compares the sharing-induced cost (B - A) under the LAG-LGFF head against the
    corresponding cost under the GAP+FC head, which is the predefined hypothesis
    that classification-head capacity modulates the calibration cost of sharing.
    """
    lag_diffs, gapfc_diffs = [], []
    for i in range(len(SEEDS)):
        if all(x[i] is not None for x in [lag_a, lag_b, gapfc_a, gapfc_b]):
            lag_diffs.append(lag_b[i] - lag_a[i])
            gapfc_diffs.append(gapfc_b[i] - gapfc_a[i])

    if len(lag_diffs) < 2:
        print("  too few valid pairs, skipping the interaction test")
        return

    lag_diffs = np.array(lag_diffs)
    gapfc_diffs = np.array(gapfc_diffs)
    t_stat, p_val = scipy_stats.ttest_rel(lag_diffs, gapfc_diffs)

    interaction_diffs = lag_diffs - gapfc_diffs
    d = (float(np.mean(interaction_diffs) / np.std(interaction_diffs, ddof=1))
         if np.std(interaction_diffs, ddof=1) > 0 else float("nan"))

    print(f"  LAG-LGFF sharing cost (B-A) = {np.mean(lag_diffs):.4f} ± {np.std(lag_diffs):.4f}")
    print(f"  GAP+FC  sharing cost (B-A) = {np.mean(gapfc_diffs):.4f} ± {np.std(gapfc_diffs):.4f}")
    reduction = ((np.mean(lag_diffs) - np.mean(gapfc_diffs)) / np.mean(lag_diffs) * 100
                 if np.mean(lag_diffs) != 0 else float("nan"))
    print(f"  narrowing under the low-capacity head: {reduction:.1f}%")
    print(f"  interaction test: t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f} (n={len(lag_diffs)} seeds)")
    print(f"  {'>>> significant (p<0.05)' if p_val < 0.05 else '>>> not significant'}")


if __name__ == "__main__":
    print("=" * 70)
    print("Brier score statistical tests (matching the D-ECE analysis)")
    print("Source: cached prediction records; no inference required")
    print("=" * 70)

    print("\nReading per-configuration Brier scores ...")
    A = load_brier_by_seed("A_independent")
    C_shallow = load_brier_by_seed("C_shallow")
    C_deep = load_brier_by_seed("C_deep")
    B = load_brier_by_seed("B_shared")
    gapfc_A = load_brier_by_seed("gapfc_A_independent")
    gapfc_B = load_brier_by_seed("gapfc_B_shared")
    F_frozen = load_brier_by_seed("F_frozen")

    print("\n" + "=" * 70)
    print("Test 1: independent vs fully shared")
    print("=" * 70)
    paired_test("A_independent", A, "B_shared", B)

    print("\n" + "=" * 70)
    print("Test 2: linear trend across sharing depths (k = 0, 5, 9, 11)")
    print("=" * 70)
    trend_test(["A(k=0)", "C_shallow(k=5)", "C_deep(k=9)", "B_shared(k=11)"],
               [A, C_shallow, C_deep, B],
               depths=[0, 5, 9, 11])

    print("\n" + "=" * 70)
    print("Test 3: gapfc_A vs gapfc_B (residual cost under the low-capacity head)")
    print("=" * 70)
    paired_test("gapfc_A_independent", gapfc_A, "gapfc_B_shared", gapfc_B)

    print("\n" + "=" * 70)
    print("Test 4 (primary): interaction between head capacity and sharing")
    print("=" * 70)
    interaction_test(A, B, gapfc_A, gapfc_B)

    print("\n" + "=" * 70)
    print("Test 5: frozen-backbone control vs fully shared")
    print("=" * 70)
    paired_test("F_frozen", F_frozen, "B_shared", B)

    print("\n" + "=" * 70)
    print("These are the Brier-score results; compare with the")
    print("D-ECE output of stats_dece.py to check for metric dependence.")
    print("=" * 70)
