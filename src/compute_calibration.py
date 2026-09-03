"""
Detection calibration analysis.

Loads trained joint-model checkpoints, transplants the detection pathway into a
standard DetectionModel, runs inference over the detection validation set, and
computes the detection expected calibration error (D-ECE) with bootstrap
confidence intervals. Covers three experiment groups:

    exp1  sharing-depth sweep     A / C_shallow / C_deep / B
    exp2  head ablation           gapfc_A / gapfc_B
    exp3  frozen-backbone control F_frozen
    exp5  additional datasets     CVC-ClinicDB, ETIS-LaribPolypDB

All groups share the same analysis logic and differ only in which checkpoint
folders they read; see CONFIG_GROUPS below.

Note: the joint model cannot be loaded with ultralytics.YOLO() directly. It is
rebuilt via load_joint_model() and its detection pathway transplanted into a
standard DetectionModel by transplant_detector(), a round-trip validated against
the original evaluation code.

Usage:
    python compute_calibration.py --group exp1
    python compute_calibration.py --group exp2
    python compute_calibration.py --group exp3
    python compute_calibration.py --group all
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics is required: pip install ultralytics==8.4.104")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import make_base, JointGIModel  # noqa: E402
from model_gapfc import JointGIModelGAPFC  # noqa: E402
from paths import ROOT, DATA, RUNS, CALIB, DET_DIR, IMGSZ, CLASSES  # noqa: E402

N_CLS = len(CLASSES)  # 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
RUNS_ROOT = RUNS
DATA_YAML = DET_DIR / "dataset.yaml"
VAL_IMAGES_DIR = DET_DIR / "images" / "val"
VAL_LABELS_DIR = DET_DIR / "labels" / "val"
OUTPUT_DIR = CALIB
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Additional-dataset evaluation. Models are retrained from scratch on each of
# these datasets rather than transferred, so each has its own checkpoint root.
# Adjust these if your CVC/ETIS directories do not follow the images/val,
# labels/val layout used for Kvasir-SEG.
CVC_RUNS_ROOT = ROOT / "runs_cvc"
CVC_VAL_IMAGES_DIR = DATA / "cvc_clinicdb" / "images" / "val"
CVC_VAL_LABELS_DIR = DATA / "cvc_clinicdb" / "labels" / "val"

ETIS_RUNS_ROOT = ROOT / "runs_etis"
ETIS_VAL_IMAGES_DIR = DATA / "etis_larib" / "images" / "val"
ETIS_VAL_LABELS_DIR = DATA / "etis_larib" / "labels" / "val"


# ------------------------------------------------------------
# Evaluation constants
# ------------------------------------------------------------
IOU_MATCH_THRESHOLD = 0.5   # IoU at which a prediction counts as a true positive
N_BINS = 10                 # equal-width bins for D-ECE
N_BOOTSTRAP = 1000          # bootstrap resamples for confidence intervals

# Deliberately permissive: calibration assessment needs the full confidence
# distribution, unlike ordinary inference where a high threshold suppresses
# low-confidence detections.
CONF_THRESHOLD_FOR_EVAL = 0.001


# ============================================================
# Experiment groups: checkpoint folder names. This is the only thing that
# differs between the three groups.
# ============================================================
CONFIG_GROUPS = {
    "exp1": {
        "A_independent": [f"A_independent_seed{s}" for s in range(42, 50)],
        "C_shallow":      [f"C_shallow_seed{s}" for s in range(42, 48)],
        "C_deep":         [f"C_deep_seed{s}" for s in range(42, 48)],
        "B_shared":       [f"B_shared_seed{s}" for s in range(42, 50)],
    },
    "exp2": {
        "gapfc_A_independent": [f"gapfc_A_independent_seed{s}" for s in range(42, 50)],
        "gapfc_B_shared":      [f"gapfc_B_shared_seed{s}" for s in range(42, 50)],
    },
    "exp3": {
        "F_frozen": [f"F_frozen_seed{s}" for s in range(42, 48)],
    },
    # Optional: PCGrad / KD interventions reuse the same pipeline if enabled
    # "exp_intervention": {
    #     "PCGrad": [f"PCGrad_seed{s}" for s in range(42, 48)],
    #     "KD":     [f"KD_seed{s}" for s in range(42, 48)],
    # },
}

# Additional datasets: three seeds (42-44) of the two extreme configurations,
# retrained on CVC-ClinicDB and ETIS. Kept separate because both the checkpoint
# root and the validation paths differ; handled by run_exp5().
EXP5_CVC_GROUP = {
    "A_independent_cvc": [f"A_independent_cvc_seed{s}" for s in range(42, 45)],
    "B_shared_cvc":       [f"B_shared_cvc_seed{s}" for s in range(42, 45)],
}
EXP5_ETIS_GROUP = {
    "A_independent_etis": [f"A_independent_etis_seed{s}" for s in range(42, 45)],
    "B_shared_etis":       [f"B_shared_etis_seed{s}" for s in range(42, 45)],
}


# ============================================================
# Loading the joint model and transplanting its detection pathway.
# Reuses the logic validated in evaluate.eval_det_official(): official weights
# passed through this path reproduce mAP@0.5 = 0.881 exactly.
# ============================================================

_base_dm_cache = {}  # cached by nc, so the pretrained weights load only once


def get_base_dm(nc=1, device=DEVICE):
    if nc not in _base_dm_cache:
        _base_dm_cache[nc] = make_base(nc=nc, device=device)
    return _base_dm_cache[nc]


def load_joint_model(ckpt_path, model_variant="lag_lgff", device=DEVICE):
    """
    Load a JointGIModel from a checkpoint.

    final.pt holds {'model': EMA fp16 state_dict, 'share_until': int, ...}.
    model_variant selects the classification head: 'lag_lgff' for the default
    model, 'gapfc' for the low-capacity ablation.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if "share_until" in ckpt:
        share_until = ckpt["share_until"]
    else:
        # Some ablation checkpoints (F_frozen) do not store share_until, so it
        # is mapped from the configuration name. F_frozen is fully shared
        # (share_until=11) with the trunk frozen.
        config_label = ckpt.get("config", "")
        KNOWN_SHARE_UNTIL = {
            "F_frozen": 11,
        }
        if config_label in KNOWN_SHARE_UNTIL:
            share_until = KNOWN_SHARE_UNTIL[config_label]
        else:
            raise KeyError(
                f"checkpoint has no share_until and config '{config_label}' is not in "
                f"KNOWN_SHARE_UNTIL; add its sharing depth to that mapping"
            )
    base_dm = get_base_dm(nc=1, device=device)

    if model_variant == "lag_lgff":
        joint = JointGIModel(share_until, base_dm, n_cls=N_CLS).to(device)
    elif model_variant == "gapfc":
        joint = JointGIModelGAPFC(share_until, base_dm, n_cls=N_CLS).to(device)
    else:
        raise ValueError(f"unknown model_variant: {model_variant}")

    # final.pt stores EMA weights in fp16; cast back to float32 before loading
    sd = {k: v.float() for k, v in ckpt["model"].items()}
    joint.load_state_dict(sd)
    joint.eval()
    return joint, base_dm, share_until


def transplant_detector(joint, base_dm, device=DEVICE):
    """
    Transplant the joint model's detection pathway (shared + det_trunk +
    det_neck) into a standard DetectionModel, wrapped as a YOLO object so that
    .predict() works directly.

    Identical to evaluate.eval_det_official() except that .val() is not called;
    the model is kept for per-image inference so that per-box confidences are
    available for the calibration analysis.
    """
    dm = copy.deepcopy(base_dm)
    seq = dm.model
    for i in range(joint.share_until):
        seq[i].load_state_dict(joint.shared[i].state_dict())
    for j, m in enumerate(joint.det_trunk):
        seq[joint.share_until + j].load_state_dict(m.state_dict())
    for i, m in enumerate(joint.det_neck):
        seq[11 + i].load_state_dict(m.state_dict())

    tmp = YOLO("yolov10s.pt")
    tmp.model = dm.to(device).eval()
    return tmp


# ============================================================
# Core functions
# ============================================================

def load_yolo_labels(label_path, img_w, img_h):
    """Read YOLO-format labels (normalised cx, cy, w, h) as absolute xyxy."""
    boxes = []
    if not os.path.exists(label_path):
        return np.zeros((0, 4))
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes) if boxes else np.zeros((0, 4))


def iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU matrix between two sets of boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter_area
    return np.where(union > 0, inter_area / union, 0)


def collect_confidence_correctness(model, images_dir, labels_dir):
    """
    Run inference over the validation set and return one
    (confidence, is_true_positive) pair per predicted box.

    Inference uses IMGSZ (224), matching the evaluation protocol used elsewhere
    so that resolution does not become a confounding variable.
    """
    records = []  # list of (confidence, is_tp)
    image_files = sorted(Path(images_dir).glob("*.jpg")) + sorted(Path(images_dir).glob("*.png"))
    # Exclude images duplicated in the classification training set
    # (cross-task overlap audit; see excluded_val_images.json)
    _excl_path = OUTPUT_DIR / "excluded_val_images.json"
    if _excl_path.exists():
        _excl = set(json.load(open(_excl_path)))
        _before = len(image_files)
        image_files = [f for f in image_files if f.name not in _excl]
        if _before != len(image_files):
            print(f"    [excluded] {_before} -> {len(image_files)} images (cross-task overlap)")

    for img_path in image_files:
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

        # greedy matching in descending confidence, as in standard mAP
        order = np.argsort(-pred_confs)
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)

        if len(gt_boxes) > 0:
            ious = iou_matrix(pred_boxes, gt_boxes)
        else:
            ious = np.zeros((len(pred_boxes), 0))

        for idx in order:
            conf = pred_confs[idx]
            if len(gt_boxes) == 0:
                records.append((conf, 0))  # no ground truth: every box is a FP
                continue
            row = ious[idx].copy()
            row[matched_gt] = -1  # a matched GT cannot be reused
            best_j = np.argmax(row)
            if row[best_j] >= IOU_MATCH_THRESHOLD:
                matched_gt[best_j] = True
                records.append((conf, 1))
            else:
                records.append((conf, 0))

    return records


def compute_dece(records, n_bins=N_BINS):
    """
    D-ECE (Detection Expected Calibration Error)
    Weighted sum over bins of |mean confidence - fraction of true positives|.
    """
    if len(records) == 0:
        return np.nan, []

    confs = np.array([r[0] for r in records])
    correct = np.array([r[1] for r in records])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(confs, bin_edges[1:-1])

    total_n = len(confs)
    dece = 0.0
    bin_stats = []

    for b in range(n_bins):
        mask = bin_ids == b
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_stats.append({"bin": b, "n": 0, "mean_conf": None, "accuracy": None})
            continue
        mean_conf = confs[mask].mean()
        accuracy = correct[mask].mean()
        dece += (n_in_bin / total_n) * abs(mean_conf - accuracy)
        bin_stats.append({
            "bin": b, "n": int(n_in_bin),
            "mean_conf": float(mean_conf), "accuracy": float(accuracy)
        })

    return dece, bin_stats


def bootstrap_dece_ci(records, n_bootstrap=N_BOOTSTRAP, ci=95):
    """Bootstrap confidence interval for D-ECE."""
    # This resamples predictions. analyse_image_level.py additionally provides
    # an image-level version, which respects the clustering of predictions
    # within images; the two give intervals of comparable width here.
    n = len(records)
    if n == 0:
        return np.nan, np.nan, np.nan

    records_arr = np.array(records)
    boot_deces = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = records_arr[idx]
        sample_records = [(sample[i, 0], sample[i, 1]) for i in range(n)]
        d, _ = compute_dece(sample_records)
        if not np.isnan(d):
            boot_deces.append(d)

    boot_deces = np.array(boot_deces)
    lower = np.percentile(boot_deces, (100 - ci) / 2)
    upper = np.percentile(boot_deces, 100 - (100 - ci) / 2)
    return float(np.mean(boot_deces)), float(lower), float(upper)


def analyze_single_checkpoint(run_folder_name, runs_root=None, images_dir=None,
                                labels_dir=None, model_variant="lag_lgff"):
    """Full calibration analysis for one checkpoint. Defaults to the
    Kvasir-SEG paths used by exp1/exp2/exp3."""
    runs_root = runs_root or RUNS_ROOT
    images_dir = images_dir or VAL_IMAGES_DIR
    labels_dir = labels_dir or VAL_LABELS_DIR

    ckpt_path = runs_root / run_folder_name / "final.pt"
    if not ckpt_path.exists():
        print(f"  [warn] not found: {ckpt_path}，skipping")
        return None

    joint, base_dm, share_until = load_joint_model(ckpt_path, model_variant=model_variant)
    model = transplant_detector(joint, base_dm)

    records = collect_confidence_correctness(model, images_dir, labels_dir)
    dece_point, bin_stats = compute_dece(records)
    dece_mean, ci_low, ci_high = bootstrap_dece_ci(records)

    return {
        "run": run_folder_name,
        "share_until": share_until,
        "n_predictions": len(records),
        "dece_point_estimate": dece_point,
        "dece_bootstrap_mean": dece_mean,
        "dece_ci_95_low": ci_low,
        "dece_ci_95_high": ci_high,
        "bin_stats": bin_stats,
    }


def run_group(group_name):
    """Run one whole group of configurations."""
    group = CONFIG_GROUPS[group_name]
    all_results = {}

    for config_name, run_folders in group.items():
        print(f"\n=== config: {config_name} ===")
        model_variant = "gapfc" if config_name.startswith("gapfc") else "lag_lgff"
        config_results = []
        for run_folder in run_folders:
            print(f"  analysing {run_folder} ...")
            res = analyze_single_checkpoint(run_folder, model_variant=model_variant)
            if res is not None:
                config_results.append(res)
                print(f"    D-ECE = {res['dece_point_estimate']:.4f} "
                      f"[{res['dece_ci_95_low']:.4f}, {res['dece_ci_95_high']:.4f}]")
        all_results[config_name] = config_results

    output_path = OUTPUT_DIR / f"{group_name}_calibration_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nresults written to {output_path}")

    # per-configuration summary across seeds (mean +/- SD)
    print(f"\n=== {group_name} cross-seed summary ===")
    for config_name, results in all_results.items():
        deces = [r["dece_point_estimate"] for r in results if r is not None]
        if deces:
            print(f"  {config_name}: D-ECE = {np.mean(deces):.4f} ± {np.std(deces):.4f} (n={len(deces)} seeds)")

    return all_results


def run_exp5():
    """Additional-dataset evaluation on CVC-ClinicDB and ETIS, three seeds each."""
    all_results = {}

    for dataset_name, group, runs_root, images_dir, labels_dir in [
        ("CVC-ClinicDB", EXP5_CVC_GROUP, CVC_RUNS_ROOT, CVC_VAL_IMAGES_DIR, CVC_VAL_LABELS_DIR),
        ("ETIS-LaribPolypDB", EXP5_ETIS_GROUP, ETIS_RUNS_ROOT, ETIS_VAL_IMAGES_DIR, ETIS_VAL_LABELS_DIR),
    ]:
        print(f"\n########## dataset: {dataset_name} ##########")
        dataset_results = {}
        for config_name, run_folders in group.items():
            print(f"\n=== config: {config_name} ===")
            config_results = []
            for run_folder in run_folders:
                print(f"  analysing {run_folder} ...")
                res = analyze_single_checkpoint(
                    run_folder, runs_root=runs_root,
                    images_dir=images_dir, labels_dir=labels_dir
                )
                if res is not None:
                    config_results.append(res)
                    print(f"    D-ECE = {res['dece_point_estimate']:.4f} "
                          f"[{res['dece_ci_95_low']:.4f}, {res['dece_ci_95_high']:.4f}]")
            dataset_results[config_name] = config_results
        all_results[dataset_name] = dataset_results

    output_path = OUTPUT_DIR / "exp5_calibration_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nresults written to {output_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=list(CONFIG_GROUPS.keys()) + ["exp5", "all"],
        default="exp1"
    )
    args = parser.parse_args()

    if args.group == "all":
        for g in CONFIG_GROUPS:
            run_group(g)
        run_exp5()
    elif args.group == "exp5":
        run_exp5()
    else:
        run_group(args.group)
