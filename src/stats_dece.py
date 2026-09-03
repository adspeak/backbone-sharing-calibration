"""
Statistical tests on the D-ECE results.

Computes D-ECE directly from the cached prediction records and reports, for
every comparison in the paper: paired t-tests with Cohen's d_z, TOST equivalence
tests at three margins, the linear trend across sharing depths, and the
interaction contrast that constitutes the primary test.

The Kvasir-SEG analyses (E1, E2, E3) read the cached records shipped with this
repository and therefore run without a GPU and without any training. The
additional-dataset comparison (E5) needs models retrained on CVC-ClinicDB and
ETIS, whose records are not bundled; that section is skipped unless they are
present.

Usage:
    python stats_dece.py
"""

import json
import re
from pathlib import Path

import numpy as np
from paths import CALIB, RECORDS_DIR

try:
    from scipy import stats as sstats
except ImportError:
    raise ImportError("scipy is required: pip install scipy")


RESULTS_DIR = CALIB
TOST_MARGINS = [0.02, 0.03, 0.05]  # equivalence margins, as in the companion study

# Seeds available per configuration. The four configurations entering the
# primary analysis were extended to eight seeds; the rest remain at six.
SEEDS_8 = [42, 43, 44, 45, 46, 47, 48, 49]
SEEDS_6 = [42, 43, 44, 45, 46, 47]

CONFIG_SEEDS = {
    "A_independent": SEEDS_8,
    "B_shared": SEEDS_8,
    "gapfc_A_independent": SEEDS_8,
    "gapfc_B_shared": SEEDS_8,
    "C_shallow": SEEDS_6,
    "C_deep": SEEDS_6,
    "F_frozen": SEEDS_6,
}

N_BINS = 10  # equal-width bins, matching compute_calibration.py


def compute_dece(conf, labels, n_bins=N_BINS):
    """
    Detection expected calibration error with equal-width bins.

    Weighted mean over non-empty bins of |mean confidence - fraction of true
    positives|. Empty bins contribute nothing, which matters here because the
    confidence distribution is strongly bimodal.
    """
    n = len(conf)
    if n == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        count = mask.sum()
        if count == 0:
            continue
        ece += (count / n) * abs(conf[mask].mean() - labels[mask].mean())
    return float(ece)


def load_dece_by_seed(config_prefix, dataset="kvasir"):
    """
    Compute per-seed D-ECE for one configuration from the cached records.

    Returns {seed: dece}, omitting any seed whose cache file is missing, so that
    downstream pairing by seed works unchanged.
    """
    out = {}
    for seed in CONFIG_SEEDS.get(config_prefix, SEEDS_8):
        path = RECORDS_DIR / f"{dataset}_{config_prefix}_seed{seed}_records.npz"
        if not path.exists():
            print(f"  [warn] missing cache {path.name}")
            continue
        data = np.load(path)
        out[seed] = compute_dece(
            data["confidence"].astype(np.float64),
            data["is_tp"].astype(np.int64),
        )
    return out


def load_results(filename):
    """Read a summary JSON written by compute_calibration.py, or None if absent."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_seed(run_name):
    """Extract the seed from a run folder name, e.g. A_independent_seed42 -> 42."""
    m = re.search(r"seed(\d+)", run_name)
    return int(m.group(1)) if m else None


def get_dece_by_seed(config_results):
    """Collect one configuration's results as {seed: value}, for seed pairing."""
    return {extract_seed(r["run"]): r["dece_point_estimate"] for r in config_results}


def paired_compare(dece_a, dece_b, label_a="A", label_b="B", margins=TOST_MARGINS):
    """
    Paired comparison between two configurations, aligned by seed.

    Returns the mean difference, paired t statistic and p-value, Cohen's d_z,
    and the TOST outcome at each equivalence margin.
    """
    common_seeds = sorted(set(dece_a.keys()) & set(dece_b.keys()))
    if len(common_seeds) < 2:
        print(f"  [warn] too few shared seeds ({len(common_seeds)}) for a paired test")
        return None
    a_vals = np.array([dece_a[s] for s in common_seeds])
    b_vals = np.array([dece_b[s] for s in common_seeds])
    diffs = b_vals - a_vals  # positive means B is worse calibrated than A
    mean_diff = diffs.mean()
    n = len(diffs)
    sd_diff = diffs.std(ddof=1) if n > 1 else np.nan
    t_stat, p_val = sstats.ttest_rel(b_vals, a_vals)
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else np.nan

    # TOST: two one-sided tests at each margin
    tost_results = {}
    for margin in margins:
        se = sd_diff / np.sqrt(n) if n > 1 else np.nan
        if se > 0 and not np.isnan(se):
            t_lower = (mean_diff + margin) / se
            t_upper = (mean_diff - margin) / se
            p_lower = 1 - sstats.t.cdf(t_lower, df=n - 1)
            p_upper = sstats.t.cdf(t_upper, df=n - 1)
            tost_p = max(p_lower, p_upper)  # TOST takes the larger one-sided p
            equivalent = tost_p < 0.05
        else:
            tost_p, equivalent = np.nan, False
        tost_results[margin] = {
            "p": float(tost_p),
            "equivalent_at_margin": bool(equivalent),
        }

    return {
        "n_seeds": n,
        "common_seeds": common_seeds,
        f"{label_a}_mean": float(a_vals.mean()),
        f"{label_b}_mean": float(b_vals.mean()),
        "mean_diff_B_minus_A": float(mean_diff),
        "paired_t": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
        "tost": tost_results,
    }


def print_comparison(title, result):
    print(f"\n--- {title} ---")
    if result is None:
        print("  (insufficient data, skipped)")
        return
    print(f"  n = {result['n_seeds']} seeds (shared seeds: {result['common_seeds']})")
    for k in [k for k in result if k.endswith("_mean")]:
        print(f"  {k} = {result[k]:.4f}")
    print(f"  mean difference (B-A) = {result['mean_diff_B_minus_A']:.4f}")
    print(f"  paired t = {result['paired_t']:.3f}, p = {result['p_value']:.4f}")
    d = result["cohens_d"]
    print(f"  Cohen's d_z = {d:.3f}" if d is not None else "  Cohen's d_z = N/A")
    print(f"  verdict: {'significant (p<0.05)' if result['p_value'] < 0.05 else 'not significant (p>=0.05)'}")
    for margin, tost in result["tost"].items():
        status = "equivalent" if tost["equivalent_at_margin"] else "not equivalent"
        print(f"  TOST at +/-{margin}: p = {tost['p']:.4f} -> {status}")


def main():
    print("=" * 60)
    print("E1: calibration across the four sharing depths")
    print("=" * 60)
    dece_a = load_dece_by_seed("A_independent")
    dece_cshallow = load_dece_by_seed("C_shallow")
    dece_cdeep = load_dece_by_seed("C_deep")
    dece_b = load_dece_by_seed("B_shared")

    r_ab = paired_compare(dece_a, dece_b, "A", "B")
    print_comparison("A_independent vs B_shared (endpoint comparison)", r_ab)

    # Trend: fit D-ECE against sharing depth per seed, test the slopes.
    # Only seeds present at every depth can contribute a four-point regression.
    print("\n--- linear trend: D-ECE against sharing depth k ---")
    k_values = np.array([0, 5, 9, 11])
    common_seeds_trend = sorted(
        set(dece_a) & set(dece_cshallow) & set(dece_cdeep) & set(dece_b)
    )
    slopes = []
    for s in common_seeds_trend:
        y = np.array([dece_a[s], dece_cshallow[s], dece_cdeep[s], dece_b[s]])
        slope, intercept, r, p, se = sstats.linregress(k_values, y)
        slopes.append(slope)
    slopes = np.array(slopes)
    t_slope, p_slope = sstats.ttest_1samp(slopes, 0)
    print(f"  n={len(slopes)} seeds, per-seed slopes: {np.round(slopes, 5).tolist()}")
    print(f"  mean slope = {slopes.mean():.5f} per module, t={t_slope:.3f}, p={p_slope:.4f}")
    print(f"  {'significant trend (p<0.05)' if p_slope < 0.05 else 'trend not significant'}")

    print("\n" + "=" * 60)
    print("E2 (primary): does the sharing cost narrow under the low-capacity head?")
    print("=" * 60)
    dece_gapfc_a = load_dece_by_seed("gapfc_A_independent")
    dece_gapfc_b = load_dece_by_seed("gapfc_B_shared")
    r_gapfc = paired_compare(dece_gapfc_a, dece_gapfc_b, "gapfc_A", "gapfc_B")
    print_comparison("gapfc_A vs gapfc_B (low-capacity head)", r_gapfc)

    # Interaction: the sharing-induced shift under one head against the other.
    # This is the primary test of the paper.
    print("\n--- interaction (primary test): does head type modulate the cost? ---")
    common_seeds_interact = sorted(
        set(dece_a) & set(dece_b) & set(dece_gapfc_a) & set(dece_gapfc_b)
    )
    delta_lag = np.array([dece_b[s] - dece_a[s] for s in common_seeds_interact])
    delta_gap = np.array([dece_gapfc_b[s] - dece_gapfc_a[s] for s in common_seeds_interact])
    interact_diff = delta_lag - delta_gap
    t_int, p_int = sstats.ttest_1samp(interact_diff, 0)
    d_int = interact_diff.mean() / interact_diff.std(ddof=1)
    print(f"  n={len(common_seeds_interact)} seeds")
    print(f"  LAG-LGFF (B-A) per seed: {np.round(delta_lag, 4).tolist()}")
    print(f"  GAP+FC   (B-A) per seed: {np.round(delta_gap, 4).tolist()}")
    print(f"  mean interaction effect = {interact_diff.mean():.4f}")
    print(f"  t = {t_int:.3f}, p = {p_int:.4f}, Cohen's d_z = {d_int:.3f}")
    print(f"  {'interaction significant (p<0.05): head type modulates the calibration cost of sharing' if p_int < 0.05 else 'interaction not significant: treat as directional evidence only'}")

    print("\n" + "=" * 60)
    print("E3: frozen-backbone control against the fully shared configuration")
    print("=" * 60)
    dece_frozen = load_dece_by_seed("F_frozen")
    r_frozen = paired_compare(dece_b, dece_frozen, "B_shared", "F_frozen")
    print_comparison("B_shared vs F_frozen (diagnostic control)", r_frozen)

    print("\n" + "=" * 60)
    print("E5: additional datasets")
    print("=" * 60)
    exp5 = load_results("exp5_calibration_results.json")
    if exp5 is None:
        print("\n  Skipped: this comparison needs models retrained on CVC-ClinicDB and")
        print("  ETIS-LaribPolypDB. Neither their cached records nor the summary file")
        print("  exp5_calibration_results.json are bundled with this repository; both")
        print("  are produced by running")
        print("      python compute_calibration.py --group exp5")
        print("  after training those configurations. The results above are unaffected.")
    else:
        for dataset_name in ["CVC-ClinicDB", "ETIS-LaribPolypDB"]:
            ds = exp5[dataset_name]
            a_key = [k for k in ds.keys() if k.startswith("A_")][0]
            b_key = [k for k in ds.keys() if k.startswith("B_")][0]
            dece_ds_a = get_dece_by_seed(ds[a_key])
            dece_ds_b = get_dece_by_seed(ds[b_key])
            r_ds = paired_compare(dece_ds_a, dece_ds_b, "A", "B")
            print_comparison(f"{dataset_name}: A vs B (n=3, exploratory)", r_ds)

    print("\n" + "=" * 60)
    print("All tests complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
