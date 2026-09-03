"""
Reliability diagrams and Brier scores, and generation of the prediction cache.

Adds two things beyond D-ECE:

  Reliability diagram -- predictions binned by confidence, plotting mean
      confidence against empirical accuracy. A perfectly calibrated model lies
      on the diagonal.

  Brier score -- mean((confidence - is_true_positive)^2). A strictly proper
      scoring rule requiring no binning, so it serves as a bin-free check on
      whether the ECE-based conclusions depend on the binning scheme.

This script also writes the cached (confidence, is_true_positive, image_index)
records that every later analysis reads. Running it once removes the need for a
GPU in all subsequent steps.

Usage:
    python compute_brier_reliability.py --group exp1
    python compute_brier_reliability.py --group exp2
    python compute_brier_reliability.py --group exp3
    python compute_brier_reliability.py --group all
    python compute_brier_reliability.py --group exp1 --no-cache   # force re-inference
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import RECORDS_DIR
from compute_calibration import (
    RUNS_ROOT,
    VAL_IMAGES_DIR,
    VAL_LABELS_DIR,
    OUTPUT_DIR,
    IOU_MATCH_THRESHOLD,
    N_BINS,
    CONF_THRESHOLD_FOR_EVAL,
    IMGSZ,
    load_joint_model,
    transplant_detector,
    load_yolo_labels,
    iou_matrix,
    CONFIG_GROUPS,
)

FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cache of raw prediction records, one file per checkpoint, holding
# (confidence, is_true_positive, image_index). Once written, every later
# analysis -- NLL, Adaptive ECE, temperature scaling -- runs without inference.
RECORDS_DIR.mkdir(parents=True, exist_ok=True)


def save_records(run_folder_name, records, dataset="kvasir"):
    """Save the records as npz, prefixed by dataset name to avoid collisions."""
    if len(records) == 0:
        return None
    confs = np.array([r[0] for r in records], dtype=np.float64)
    labels = np.array([r[1] for r in records], dtype=np.int64)
    img_idx = np.array([r[2] for r in records], dtype=np.int64)
    out_path = RECORDS_DIR / f"{dataset}_{run_folder_name}_records.npz"
    np.savez_compressed(out_path, confidence=confs, is_tp=labels, image_idx=img_idx)
    return out_path


def load_records(run_folder_name, dataset="kvasir"):
    """Read back cached records, or None if the file does not exist."""
    path = RECORDS_DIR / f"{dataset}_{run_folder_name}_records.npz"
    if not path.exists():
        return None
    data = np.load(path)
    if "image_idx" in data:
        return list(zip(data["confidence"].tolist(), data["is_tp"].tolist(),
                        data["image_idx"].tolist()))
    return list(zip(data["confidence"].tolist(), data["is_tp"].tolist()))


def collect_confidence_correctness(model, images_dir, labels_dir):
    """Collect (confidence, is_true_positive, image_index) records, using the
    same matching logic as compute_calibration.py."""
    records = []
    image_files = sorted(Path(images_dir).glob("*.jpg")) + sorted(Path(images_dir).glob("*.png"))
    # Exclude images duplicated in the classification training set
    _excl_path = OUTPUT_DIR / "excluded_val_images.json"
    if _excl_path.exists():
        _excl = set(json.load(open(_excl_path)))
        _before = len(image_files)
        image_files = [f for f in image_files if f.name not in _excl]
        if _before != len(image_files):
            print(f"    [excluded] {_before} -> {len(image_files)} images (cross-task overlap)")

    for _img_i, img_path in enumerate(image_files):
        result = model.predict(
            str(img_path), conf=CONF_THRESHOLD_FOR_EVAL, imgsz=IMGSZ, verbose=False
        )[0]
        img_h, img_w = result.orig_shape

        pred_boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))
        pred_confs = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.zeros((0,))

        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        gt_boxes = load_yolo_labels(label_path, img_w, img_h)

        if len(pred_boxes) == 0:
            continue

        order = np.argsort(-pred_confs)
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)

        if len(gt_boxes) > 0:
            ious = iou_matrix(pred_boxes, gt_boxes)
        else:
            ious = np.zeros((len(pred_boxes), 0))

        for idx in order:
            conf = pred_confs[idx]
            if len(gt_boxes) == 0:
                records.append((conf, 0, _img_i))
                continue
            row = ious[idx].copy()
            row[matched_gt] = -1
            best_j = np.argmax(row)
            if row[best_j] >= IOU_MATCH_THRESHOLD:
                matched_gt[best_j] = True
                records.append((conf, 1, _img_i))
            else:
                records.append((conf, 0, _img_i))

    return records


def compute_brier_score(records):
    """Brier score: mean((confidence - is_tp)^2). Lower is better."""
    if len(records) == 0:
        return None
    confs = np.array([r[0] for r in records])
    labels = np.array([r[1] for r in records])
    return float(np.mean((confs - labels) ** 2))


def compute_reliability_bins(records, n_bins=N_BINS):
    """Bin by confidence, returning per-bin mean confidence, accuracy and count,
    for the reliability diagram."""
    if len(records) == 0:
        return []
    confs = np.array([r[0] for r in records])
    labels = np.array([r[1] for r in records])

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin_low": float(lo), "bin_high": float(hi),
                         "mean_confidence": None, "empirical_accuracy": None, "count": 0})
            continue
        bins.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "mean_confidence": float(confs[mask].mean()),
            "empirical_accuracy": float(labels[mask].mean()),
            "count": count,
        })
    return bins


def analyze_single_checkpoint(run_folder_name, model_variant="lag_lgff", use_cache=True):
    """
    Read from the cache when available; otherwise run inference and write the
    cache. The first pass needs a GPU; every later analysis does not.
    """
    share_until = None

    if use_cache:
        cached = load_records(run_folder_name)
        if cached is not None:
            print(f"    [cached] reusing stored records, skipping inference")
            brier = compute_brier_score(cached)
            bins = compute_reliability_bins(cached)
            return {"run": run_folder_name, "share_until": share_until,
                    "brier_score": brier, "bins": bins, "from_cache": True}

    ckpt_path = RUNS_ROOT / run_folder_name / "final.pt"
    if not ckpt_path.exists():
        print(f"  [warn] not found: {ckpt_path}，skipping")
        return None

    joint, base_dm, share_until = load_joint_model(ckpt_path, model_variant=model_variant)
    model = transplant_detector(joint, base_dm)

    records = collect_confidence_correctness(model, VAL_IMAGES_DIR, VAL_LABELS_DIR)
    saved_path = save_records(run_folder_name, records)
    if saved_path is not None:
        print(f"    [cached] wrote {saved_path.name} ({len(records)} records)")

    brier = compute_brier_score(records)
    bins = compute_reliability_bins(records)

    return {"run": run_folder_name, "share_until": share_until,
            "brier_score": brier, "bins": bins, "from_cache": False}


def aggregate_bins_across_seeds(per_seed_bins_list, n_bins=N_BINS):
    """Average per-bin statistics across seeds for a single summary curve."""
    agg = []
    for i in range(n_bins):
        confs, accs = [], []
        for bins in per_seed_bins_list:
            if i < len(bins) and bins[i]["count"] > 0:
                confs.append(bins[i]["mean_confidence"])
                accs.append(bins[i]["empirical_accuracy"])
        if confs:
            agg.append({"mean_confidence": float(np.mean(confs)), "empirical_accuracy": float(np.mean(accs))})
        else:
            agg.append({"mean_confidence": None, "empirical_accuracy": None})
    return agg


def plot_reliability_diagram(group_name, config_to_agg_bins, output_path):
    """Plot reliability curves for several configurations, with the diagonal."""
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    for config_name, agg_bins in config_to_agg_bins.items():
        xs = [b["mean_confidence"] for b in agg_bins if b["mean_confidence"] is not None]
        ys = [b["empirical_accuracy"] for b in agg_bins if b["empirical_accuracy"] is not None]
        if xs:
            plt.plot(xs, ys, marker="o", label=config_name)

    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy (precision)")
    plt.title(f"Reliability Diagram — {group_name}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  reliability diagram written to {output_path}")


def run_group(group_name, use_cache=True):
    group = CONFIG_GROUPS[group_name]
    all_results = {}
    config_to_agg_bins = {}

    for config_name, run_folders in group.items():
        print(f"\n=== config: {config_name} ===")
        model_variant = "gapfc" if "gapfc" in config_name else "lag_lgff"
        config_results = []
        for run_folder in run_folders:
            print(f"  analysing {run_folder} ...")
            res = analyze_single_checkpoint(run_folder, model_variant=model_variant,
                                            use_cache=use_cache)
            if res is not None:
                config_results.append(res)
                print(f"    Brier score = {res['brier_score']:.4f}")
        all_results[config_name] = config_results

        per_seed_bins = [r["bins"] for r in config_results if r is not None]
        config_to_agg_bins[config_name] = aggregate_bins_across_seeds(per_seed_bins)

    output_path = OUTPUT_DIR / f"{group_name}_reliability_brier_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nresults written to {output_path}")

    print(f"\n=== {group_name} Brier score cross-seed summary ===")
    brier_by_config = {}
    for config_name, results in all_results.items():
        briers = [r["brier_score"] for r in results if r is not None and r["brier_score"] is not None]
        brier_by_config[config_name] = briers
        if briers:
            print(f"  {config_name}: Brier = {np.mean(briers):.4f} ± {np.std(briers):.4f} (n={len(briers)} seeds)")
        else:
            print(f"  {config_name}: no valid data")

    config_names = list(brier_by_config.keys())
    if len(config_names) >= 2:
        print(f"\n=== paired t-tests between adjacent configurations (indicative only) ===")
        for i in range(len(config_names) - 1):
            a_name, b_name = config_names[i], config_names[i + 1]
            a_vals, b_vals = brier_by_config[a_name], brier_by_config[b_name]
            if len(a_vals) == len(b_vals) and len(a_vals) > 1:
                t_stat, p_val = scipy_stats.ttest_rel(a_vals, b_vals)
                print(f"  {a_name} vs {b_name}: t={t_stat:.3f}, p={p_val:.4f}")

    fig_path = FIGURES_DIR / f"reliability_diagram_{group_name}.png"
    plot_reliability_diagram(group_name, config_to_agg_bins, fig_path)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["exp1", "exp2", "exp3", "all"], default="all")
    parser.add_argument("--no-cache", action="store_true",
                        help="force re-inference, ignoring any cached records")
    args = parser.parse_args()

    groups_to_run = ["exp1", "exp2", "exp3"] if args.group == "all" else [args.group]
    for g in groups_to_run:
        print(f"\n{'='*60}\nprocessing {g}\n{'='*60}")
        run_group(g, use_cache=not args.no_cache)
