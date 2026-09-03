"""
Temperature scaling as a diagnostic.

Asks whether the calibration gap introduced by backbone sharing can be removed
by standard post-hoc correction. Two outcomes are distinguishable:

  H1  the gap is a confidence offset -- ranking is intact, only the scale is
      wrong, and temperature scaling should largely remove it;
  H2  the gap reflects degraded discrimination -- the model separates true from
      false detections less well, which no monotone rescaling can repair.

Because temperature scaling is a strictly monotone transformation it cannot
change the ordering of predictions, and therefore cannot change AUC. AUC thus
provides a ranking-based measure that is invariant under the correction, and a
persistent AUC difference between configurations distinguishes H2 from H1.

Implementation notes:
  - The detector emits sigmoid confidences rather than raw logits, so the logit
    is recovered as log(p / (1 - p)) with numerical clipping. This differs from
    fitting on raw logits and is stated as such in the paper.
  - Five-fold CROSS-FITTING: the temperature for each fold is fitted on the
    other four. Fitting and evaluating on the same predictions would bias the
    result optimistically.
  - The temperature is parameterised as log T to enforce T > 0.

Reads the cached records; no inference, no GPU.

Usage:
    python analyse_temperature_scaling.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import optimize as scipy_optimize
from scipy import stats as scipy_stats
from paths import CALIB, RECORDS_DIR

OUTPUT_DIR = CALIB

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
N_BINS = 10
N_FOLDS = 5
EPS = 1e-6  # clipping, to keep the logit finite

CONFIGS = [
    "A_independent",
    "B_shared",
    "gapfc_A_independent",
    "gapfc_B_shared",
    "F_frozen",
]


def load_records(run_name, dataset="kvasir"):
    path = RECORDS_DIR / f"{dataset}_{run_name}_records.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    return data["confidence"].astype(np.float64), data["is_tp"].astype(np.int64)


def conf_to_logit(conf):
    """Recover the logit from a sigmoid confidence, with clipping."""
    conf = np.clip(conf, EPS, 1 - EPS)
    return np.log(conf / (1 - conf))


def logit_to_conf(logit):
    return 1.0 / (1.0 + np.exp(-logit))


def compute_dece(conf, labels, n_bins=N_BINS):
    """D-ECE with equal-width bins."""
    if len(conf) == 0:
        return None
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        cnt = mask.sum()
        if cnt == 0:
            continue
        ece += (cnt / total) * abs(conf[mask].mean() - labels[mask].mean())
    return float(ece)


def compute_brier(conf, labels):
    if len(conf) == 0:
        return None
    return float(np.mean((conf - labels) ** 2))


def compute_auc(conf, labels):
    """
    Area under the ROC curve: how well the model ranks true detections above
    false ones.

    Temperature scaling is a strictly monotone transformation and cannot change
    the ordering, so AUC is identical before and after. It therefore measures
    discrimination independently of calibration, which is what distinguishes a
    correctable confidence offset from a genuine ranking deficit.
    """
    pos = conf[labels == 1]
    neg = conf[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # AUC via the Mann-Whitney U statistic
    all_vals = np.concatenate([pos, neg])
    ranks = scipy_stats.rankdata(all_vals)
    n_pos, n_neg = len(pos), len(neg)
    auc = (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def nll_loss_for_temperature(log_T, logits, labels):
    """
    Fit the temperature by minimising NLL, parameterised as log T to enforce
    T > 0.
    """
    T = np.exp(log_T)
    scaled = logits / T
    # numerically stable binary cross-entropy
    p = logit_to_conf(np.clip(scaled, -30, 30))
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))


def fit_temperature(logits, labels):
    """Fit the temperature on the given data."""
    res = scipy_optimize.minimize_scalar(
        lambda lt: nll_loss_for_temperature(lt, logits, labels),
        bounds=(-3.0, 3.0), method="bounded",
    )
    return float(np.exp(res.x))


def cross_fitted_temperature_scaling(conf, labels, n_folds=N_FOLDS, seed=0):
    """
    K-fold cross-fitting: the temperature for each fold is fitted on the other
    folds and applied to the held-out one, so no prediction contributes to
    fitting the temperature under which it is evaluated.

    Returns (rescaled confidences, per-fold temperatures).
    """
    n = len(conf)
    if n < n_folds * 2:
        return None, []

    logits = conf_to_logit(conf)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)

    calibrated = np.zeros(n, dtype=np.float64)
    temperatures = []

    for f in range(n_folds):
        test_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        if len(np.unique(labels[train_idx])) < 2:
            # only one class present in this fold, cannot fit; fall back to T=1
            calibrated[test_idx] = conf[test_idx]
            temperatures.append(1.0)
            continue
        T = fit_temperature(logits[train_idx], labels[train_idx])
        temperatures.append(T)
        calibrated[test_idx] = logit_to_conf(np.clip(logits[test_idx] / T, -30, 30))

    return calibrated, temperatures


def analyze_config(config_name):
    """Run the temperature-scaling analysis over all seeds of one configuration."""
    rows = []
    for seed in SEEDS:
        run_name = f"{config_name}_seed{seed}"
        conf, labels = load_records(run_name)
        if conf is None:
            print(f"  [warn] missing cache {run_name}")
            continue

        dece_before = compute_dece(conf, labels)
        brier_before = compute_brier(conf, labels)
        auc = compute_auc(conf, labels)

        calibrated, temps = cross_fitted_temperature_scaling(conf, labels, seed=seed)
        if calibrated is None:
            print(f"  [warn] {run_name} too few samples, skipping TS")
            continue

        dece_after = compute_dece(calibrated, labels)
        brier_after = compute_brier(calibrated, labels)

        rows.append({
            "run": run_name,
            "n_records": int(len(conf)),
            "mean_temperature": float(np.mean(temps)),
            "dece_before": dece_before,
            "dece_after": dece_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "auc": auc,
        })
        print(f"  {run_name}: T={np.mean(temps):.3f}  "
              f"D-ECE {dece_before:.4f}→{dece_after:.4f}  "
              f"Brier {brier_before:.4f}→{brier_after:.4f}  AUC={auc:.4f}")

    return rows


def summarize(config_name, rows):
    if not rows:
        return None
    def m(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    dece_b, dece_b_sd = m("dece_before")
    dece_a, dece_a_sd = m("dece_after")
    brier_b, brier_b_sd = m("brier_before")
    brier_a, brier_a_sd = m("brier_after")
    auc_m, auc_sd = m("auc")
    T_m, T_sd = m("mean_temperature")

    dece_fix = (dece_b - dece_a) / dece_b * 100 if dece_b else float("nan")
    brier_fix = (brier_b - brier_a) / brier_b * 100 if brier_b else float("nan")

    print(f"\n  === {config_name} summary (n={len(rows)} seeds) ===")
    print(f"    fitted T      = {T_m:.3f} ± {T_sd:.3f}")
    print(f"    D-ECE before/after = {dece_b:.4f}±{dece_b_sd:.4f} / {dece_a:.4f}±{dece_a_sd:.4f}  (reduced by {dece_fix:+.1f}%)")
    print(f"    Brier before/after = {brier_b:.4f}±{brier_b_sd:.4f} / {brier_a:.4f}±{brier_a_sd:.4f}  (reduced by {brier_fix:+.1f}%)")
    print(f"    AUC          = {auc_m:.4f} ± {auc_sd:.4f}   (unchanged by TS)")

    return {
        "config": config_name, "n_seeds": len(rows),
        "temperature_mean": T_m, "temperature_sd": T_sd,
        "dece_before_mean": dece_b, "dece_after_mean": dece_a, "dece_fix_pct": dece_fix,
        "brier_before_mean": brier_b, "brier_after_mean": brier_a, "brier_fix_pct": brier_fix,
        "auc_mean": auc_m, "auc_sd": auc_sd,
        "per_seed": rows,
    }


def paired_test(name_a, vals_a, name_b, vals_b, label):
    pairs = [(a, b) for a, b in zip(vals_a, vals_b) if a is not None and b is not None]
    if len(pairs) < 2:
        print(f"    {label}: too few valid pairs")
        return
    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])
    t, p = scipy_stats.ttest_rel(b_arr, a_arr)
    print(f"    {label}: {name_a}={a_arr.mean():.4f} vs {name_b}={b_arr.mean():.4f}, "
          f"t={t:.3f}, p={p:.4f} {'[significant]' if p < 0.05 else '[n.s.]'}")


if __name__ == "__main__":
    print("=" * 72)
    print("Temperature scaling: can post-hoc correction remove the sharing-induced gap?")
    print(f"Method: {N_FOLDS}-fold cross-fitting, fitting and evaluation strictly separated")
    print("Source: cached prediction records; no inference")
    print("=" * 72)

    all_summary = {}
    per_config_rows = {}

    for cfg in CONFIGS:
        print(f"\n### {cfg}")
        rows = analyze_config(cfg)
        per_config_rows[cfg] = rows
        s = summarize(cfg, rows)
        if s:
            all_summary[cfg] = s

    out_path = OUTPUT_DIR / "exp6_temperature_scaling_results.json"
    with open(out_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nresults written to {out_path}")

    # ---- does the gap survive temperature scaling? ----
    print("\n" + "=" * 72)
    print("Analysis 1: does the sharing-induced calibration cost survive temperature scaling?")
    print("=" * 72)

    for a_name, b_name, tag in [
        ("A_independent", "B_shared", "LAG-LGFF head"),
        ("gapfc_A_independent", "gapfc_B_shared", "GAP+FC head"),
    ]:
        if a_name not in per_config_rows or b_name not in per_config_rows:
            continue
        print(f"\n  【{tag}】")
        a_rows, b_rows = per_config_rows[a_name], per_config_rows[b_name]
        paired_test(a_name, [r["dece_before"] for r in a_rows],
                    b_name, [r["dece_before"] for r in b_rows], "D-ECE (before TS)")
        paired_test(a_name, [r["dece_after"] for r in a_rows],
                    b_name, [r["dece_after"] for r in b_rows], "D-ECE (after TS)")
        paired_test(a_name, [r["brier_before"] for r in a_rows],
                    b_name, [r["brier_before"] for r in b_rows], "Brier (before TS)")
        paired_test(a_name, [r["brier_after"] for r in a_rows],
                    b_name, [r["brier_after"] for r in b_rows], "Brier (after TS)")

    # ---- is ranking quality (AUC) affected? ----
    print("\n" + "=" * 72)
    print("Analysis 2: AUC, distinguishing a confidence offset from a ranking deficit")
    print("=" * 72)
    print("  TS cannot change AUC. A significant drop under sharing therefore indicates")
    print("  degraded ranking, which no monotone calibrator can repair.")

    for a_name, b_name, tag in [
        ("A_independent", "B_shared", "LAG-LGFF head"),
        ("gapfc_A_independent", "gapfc_B_shared", "GAP+FC head"),
    ]:
        if a_name not in per_config_rows or b_name not in per_config_rows:
            continue
        print(f"\n  【{tag}】")
        a_rows, b_rows = per_config_rows[a_name], per_config_rows[b_name]
        paired_test(a_name, [r["auc"] for r in a_rows],
                    b_name, [r["auc"] for r in b_rows], "AUC")

    print("\n" + "=" * 72)
    print("Reading the output:")
    print("  - gap largely removed by TS      -> a correctable confidence offset")
    print("  - gap persists and AUC drops     -> a ranking deficit TS cannot repair")
    print("  - neither is clear               -> underpowered; report as inconclusive")
    print("=" * 72)
