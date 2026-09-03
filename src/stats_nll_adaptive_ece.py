"""
Supplementary calibration metrics: NLL and Adaptive ECE.
=======================================================
Adds two metrics beyond D-ECE and the Brier score, addressing two distinct
concerns: whether the conclusions depend on the particular scoring rule, and
whether they depend on the binning scheme.
  NLL (Negative Log-Likelihood)：
      -mean(y*log(p) + (1-y)*log(1-p))
      A strictly proper scoring rule penalising both miscalibration and poor
      discrimination. Unlike the Brier score, it penalises confidently wrong
      predictions very heavily.

  Adaptive ECE (equal-mass bins):
      The same weighted mean of |mean confidence - accuracy| as D-ECE, but with
      bin boundaries at quantiles of the confidence distribution so that every
      bin holds the same number of predictions. Equal-width bins are sparsely
      populated when confidences concentrate in a narrow range, which inflates
      the variance of the per-bin estimates.

Reads the cached records; no inference, no GPU.

Usage:
    python stats_nll_adaptive_ece.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from paths import CALIB, RECORDS_DIR

OUTPUT_DIR = CALIB

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
SEEDS_6 = [42, 43, 44, 45, 46, 47]  # configurations not extended to 8 seeds

N_BINS = 10
EPS = 1e-12

# (configuration name, seeds actually available for it)
CONFIGS = [
    ("A_independent", SEEDS),
    ("C_shallow", SEEDS_6),
    ("C_deep", SEEDS_6),
    ("B_shared", SEEDS),
    ("gapfc_A_independent", SEEDS),
    ("gapfc_B_shared", SEEDS),
    ("F_frozen", SEEDS_6),
]


def load_records(run_name, dataset="kvasir"):
    path = RECORDS_DIR / f"{dataset}_{run_name}_records.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    return data["confidence"].astype(np.float64), data["is_tp"].astype(np.int64)


def compute_nll(conf, labels):
    """
    Negative log-likelihood; lower is better. Penalises confidently wrong
    predictions much more heavily than the Brier score.
    """
    if len(conf) == 0:
        return None
    p = np.clip(conf, EPS, 1 - EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def compute_adaptive_ece(conf, labels, n_bins=N_BINS):
    """
    Adaptive ECE with equal-mass bins, for comparison against the equal-width
    D-ECE, to test whether conclusions depend on the binning scheme.
    """
    n = len(conf)
    if n == 0:
        return None
    order = np.argsort(conf)
    conf_sorted = conf[order]
    labels_sorted = labels[order]

    # split into n_bins groups of equal size (the last may take 1-2 extra)
    bin_indices = np.array_split(np.arange(n), n_bins)

    ece = 0.0
    for idx in bin_indices:
        if len(idx) == 0:
            continue
        bin_conf = conf_sorted[idx].mean()
        bin_acc = labels_sorted[idx].mean()
        ece += (len(idx) / n) * abs(bin_conf - bin_acc)
    return float(ece)


def compute_equal_width_ece(conf, labels, n_bins=N_BINS):
    """Equal-width D-ECE, computed here for a direct side-by-side comparison."""
    if len(conf) == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        cnt = mask.sum()
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(conf[mask].mean() - labels[mask].mean())
    return float(ece)


def analyze_config(config_name, seed_list):
    rows = []
    for seed in seed_list:
        run_name = f"{config_name}_seed{seed}"
        conf, labels = load_records(run_name)
        if conf is None:
            print(f"    [warn] missing cache {run_name}")
            continue
        rows.append({
            "run": run_name,
            "n_records": int(len(conf)),
            "nll": compute_nll(conf, labels),
            "adaptive_ece": compute_adaptive_ece(conf, labels),
            "equal_width_ece": compute_equal_width_ece(conf, labels),
        })
    return rows


def summarize(config_name, rows):
    if not rows:
        return None

    def m(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    nll_m, nll_sd = m("nll")
    aece_m, aece_sd = m("adaptive_ece")
    ewece_m, ewece_sd = m("equal_width_ece")

    print(f"  {config_name} (n={len(rows)} seeds):")
    print(f"    NLL              = {nll_m:.4f} ± {nll_sd:.4f}")
    print(f"    Adaptive ECE     = {aece_m:.4f} ± {aece_sd:.4f}  (equal-mass bins)")
    print(f"    Equal-width ECE  = {ewece_m:.4f} ± {ewece_sd:.4f}  (equal-width bins, for comparison)")

    return {
        "config": config_name, "n_seeds": len(rows),
        "nll_mean": nll_m, "nll_sd": nll_sd,
        "adaptive_ece_mean": aece_m, "adaptive_ece_sd": aece_sd,
        "equal_width_ece_mean": ewece_m, "equal_width_ece_sd": ewece_sd,
        "per_seed": rows,
    }


def paired_test(label, name_a, vals_a, name_b, vals_b):
    pairs = [(a, b) for a, b in zip(vals_a, vals_b) if a is not None and b is not None]
    if len(pairs) < 2:
        print(f"    {label}: too few valid pairs")
        return None
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    diffs = b - a
    t, p = scipy_stats.ttest_rel(b, a)
    d = float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else float("nan")
    print(f"    {label}: {a.mean():.4f} → {b.mean():.4f}, "
          f"t={t:.3f}, p={p:.4f}, d={d:.3f} (n={len(pairs)}) "
          f"{'[significant]' if p < 0.05 else '[n.s.]'}")
    return {"t": float(t), "p": float(p), "d": d, "n": len(pairs)}


def interaction_test(label, lag_a, lag_b, gapfc_a, gapfc_b):
    """Interaction test: the sharing-induced shift under the LAG-LGFF head
    against the corresponding shift under the GAP+FC head."""
    lag_d, gap_d = [], []
    for i in range(len(SEEDS)):
        if all(i < len(x) and x[i] is not None for x in [lag_a, lag_b, gapfc_a, gapfc_b]):
            lag_d.append(lag_b[i] - lag_a[i])
            gap_d.append(gapfc_b[i] - gapfc_a[i])
    if len(lag_d) < 2:
        print(f"    {label}: too few valid pairs")
        return None
    lag_d = np.array(lag_d)
    gap_d = np.array(gap_d)
    t, p = scipy_stats.ttest_rel(lag_d, gap_d)
    inter = lag_d - gap_d
    d = float(np.mean(inter) / np.std(inter, ddof=1)) if np.std(inter, ddof=1) > 0 else float("nan")
    narrow = ((lag_d.mean() - gap_d.mean()) / lag_d.mean() * 100) if lag_d.mean() != 0 else float("nan")
    print(f"    {label}:")
    print(f"      LAG-LGFF sharing cost (B-A) = {lag_d.mean():.4f} ± {lag_d.std():.4f}")
    print(f"      GAP+FC  sharing cost (B-A) = {gap_d.mean():.4f} ± {gap_d.std():.4f}")
    print(f"      narrowing under the low-capacity head: {narrow:.1f}%")
    print(f"      t={t:.3f}, p={p:.4f}, d={d:.3f} (n={len(lag_d)}) "
          f"{'[significant]' if p < 0.05 else '[n.s.]'}")
    return {"t": float(t), "p": float(p), "d": d, "n": len(lag_d)}


def get_metric_by_seed(rows_by_config, config_name, metric, seed_list):
    """Collect one metric across seeds for one configuration, None where absent."""
    rows = rows_by_config.get(config_name, [])
    lookup = {r["run"]: r[metric] for r in rows}
    return [lookup.get(f"{config_name}_seed{s}") for s in seed_list]


if __name__ == "__main__":
    print("=" * 72)
    print("Supplementary calibration metrics: NLL and Adaptive ECE")
    print("Source: cached prediction records; no inference")
    print("=" * 72)

    print("\n### Per-configuration summary\n")
    rows_by_config = {}
    summary = {}
    for cfg, seed_list in CONFIGS:
        rows = analyze_config(cfg, seed_list)
        rows_by_config[cfg] = rows
        s = summarize(cfg, rows)
        if s:
            summary[cfg] = s
        print()

    out_path = OUTPUT_DIR / "nll_adaptive_ece_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"results written to {out_path}\n")

    # ---------- NLL ----------
    print("=" * 72)
    print("NLL statistical tests (matching the D-ECE and Brier analyses)")
    print("=" * 72)

    A_nll = get_metric_by_seed(rows_by_config, "A_independent", "nll", SEEDS)
    B_nll = get_metric_by_seed(rows_by_config, "B_shared", "nll", SEEDS)
    gA_nll = get_metric_by_seed(rows_by_config, "gapfc_A_independent", "nll", SEEDS)
    gB_nll = get_metric_by_seed(rows_by_config, "gapfc_B_shared", "nll", SEEDS)
    F_nll = get_metric_by_seed(rows_by_config, "F_frozen", "nll", SEEDS_6)
    B_nll_6 = get_metric_by_seed(rows_by_config, "B_shared", "nll", SEEDS_6)

    print("\n  Test 1: independent vs fully shared")
    paired_test("NLL", "A", A_nll, "B", B_nll)

    print("\n  Test 2: gapfc_A vs gapfc_B")
    paired_test("NLL", "gapfc_A", gA_nll, "gapfc_B", gB_nll)

    print("\n  Test 3 (primary): interaction")
    interaction_test("NLL interaction", A_nll, B_nll, gA_nll, gB_nll)

    print("\n  Test 4: frozen-backbone control vs fully shared (6 seeds)")
    paired_test("NLL", "F_frozen", F_nll, "B_shared", B_nll_6)

    # ---------- Adaptive ECE ----------
    print("\n" + "=" * 72)
    print("Adaptive ECE tests (equal-mass bins: does the conclusion depend on binning?)")
    print("=" * 72)

    A_ae = get_metric_by_seed(rows_by_config, "A_independent", "adaptive_ece", SEEDS)
    B_ae = get_metric_by_seed(rows_by_config, "B_shared", "adaptive_ece", SEEDS)
    gA_ae = get_metric_by_seed(rows_by_config, "gapfc_A_independent", "adaptive_ece", SEEDS)
    gB_ae = get_metric_by_seed(rows_by_config, "gapfc_B_shared", "adaptive_ece", SEEDS)
    F_ae = get_metric_by_seed(rows_by_config, "F_frozen", "adaptive_ece", SEEDS_6)
    B_ae_6 = get_metric_by_seed(rows_by_config, "B_shared", "adaptive_ece", SEEDS_6)

    print("\n  Test 1: independent vs fully shared")
    paired_test("Adaptive ECE", "A", A_ae, "B", B_ae)

    print("\n  Test 2: gapfc_A vs gapfc_B")
    paired_test("Adaptive ECE", "gapfc_A", gA_ae, "gapfc_B", gB_ae)

    print("\n  Test 3 (primary): interaction")
    interaction_test("Adaptive ECE interaction", A_ae, B_ae, gA_ae, gB_ae)

    print("\n  Test 4: frozen-backbone control vs fully shared (6 seeds)")
    paired_test("Adaptive ECE", "F_frozen", F_ae, "B_shared", B_ae_6)

    print("\n" + "=" * 72)
    print("Reading the output:")
    print("  If NLL and Adaptive ECE agree with D-ECE and Brier on every verdict, the")
    print("  conclusions depend neither on the metric nor on the binning scheme.")
    print("  Any disagreement between the two ECE variants should be reported.")
    print("  discussed as a limitation of binned estimation.")
    print("=" * 72)
