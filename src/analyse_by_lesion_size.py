"""
Size-stratified calibration analysis.

Explores whether the calibration cost of backbone sharing varies with lesion
size. The analysis needs a dataset containing a substantial number of small
lesions. Median relative lesion area in the training sets is 18.47% for
Kvasir-SEG, 9.81% for CVC-ClinicDB and 2.85% for ETIS-LaribPolypDB; only the
last covers a range of substantially smaller lesions, so ETIS is used here.

Method:
  1. Tertile thresholds on relative box area (box_area / image_area) are derived
     from the TRAINING set only, never the validation set, to avoid leakage.
  2. Each prediction is assigned to a stratum by the area of the PREDICTED box,
     not of whatever ground-truth box it matches. This matches deployment, where
     the model does not know the true lesion size.
  3. D-ECE is computed separately within each stratum.

Reuses the model loading and inference code in compute_calibration.py.

Usage:
    python analyse_by_lesion_size.py --dataset etis     # recommended
    python analyse_by_lesion_size.py --dataset cvc
    python analyse_by_lesion_size.py --dataset kvasir --group exp1
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Reuse the validated loading, inference and metric code
from paths import ROOT, DATA, DET_DIR
from compute_calibration import (
    RUNS_ROOT,
    VAL_IMAGES_DIR,
    VAL_LABELS_DIR,
    OUTPUT_DIR,
    IOU_MATCH_THRESHOLD,
    N_BINS,
    N_BOOTSTRAP,
    CONF_THRESHOLD_FOR_EVAL,
    IMGSZ,
    load_joint_model,
    transplant_detector,
    load_yolo_labels,
    iou_matrix,
    compute_dece,
    bootstrap_dece_ci,
    CONFIG_GROUPS,
    CVC_RUNS_ROOT,
    CVC_VAL_IMAGES_DIR,
    CVC_VAL_LABELS_DIR,
    ETIS_RUNS_ROOT,
    ETIS_VAL_IMAGES_DIR,
    ETIS_VAL_LABELS_DIR,
    EXP5_CVC_GROUP,
    EXP5_ETIS_GROUP,
)


TRAIN_DIRS = {
    "kvasir": {
        "images": DET_DIR / "images" / "train",
        "labels": DET_DIR / "labels" / "train",
    },
    "cvc": {
        "images": DATA / "cvc_clinicdb" / "images" / "train",
        "labels": DATA / "cvc_clinicdb" / "labels" / "train",
    },
    "etis": {
        "images": DATA / "etis_larib" / "images" / "train",
        "labels": DATA / "etis_larib" / "labels" / "train",
    },
}

DATASET_CONFIG = {
    "kvasir": {
        "runs_root": RUNS_ROOT,
        "val_images": VAL_IMAGES_DIR,
        "val_labels": VAL_LABELS_DIR,
        "groups": CONFIG_GROUPS,
    },
    "cvc": {
        "runs_root": CVC_RUNS_ROOT,
        "val_images": CVC_VAL_IMAGES_DIR,
        "val_labels": CVC_VAL_LABELS_DIR,
        "groups": {"exp5_cvc": EXP5_CVC_GROUP},
    },
    "etis": {
        "runs_root": ETIS_RUNS_ROOT,
        "val_images": ETIS_VAL_IMAGES_DIR,
        "val_labels": ETIS_VAL_LABELS_DIR,
        "groups": {"exp5_etis": EXP5_ETIS_GROUP},
    },
}

CLINICAL_FIELD_DIAMETER_MM = 35.0
CLINICAL_DIMINUTIVE_MM = 5.0
CLINICAL_REFERENCE_RELATIVE_AREA = (CLINICAL_DIMINUTIVE_MM / CLINICAL_FIELD_DIAMETER_MM) ** 2


def compute_size_thresholds(dataset="kvasir"):
    train_images_dir = TRAIN_DIRS[dataset]["images"]
    train_labels_dir = TRAIN_DIRS[dataset]["labels"]

    relative_areas = []
    image_files = (
        sorted(Path(train_images_dir).glob("*.jpg"))
        + sorted(Path(train_images_dir).glob("*.png"))
    )

    for img_path in image_files:
        label_path = Path(train_labels_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                relative_areas.append(w * h)

    relative_areas = np.array(relative_areas)
    threshold_low = np.percentile(relative_areas, 100 / 3)
    threshold_high = np.percentile(relative_areas, 200 / 3)

    print(f"[{dataset}] tertile thresholds on relative GT box area (training set):")
    print(f"  small  < {threshold_low:.4f}")
    print(f"  medium: {threshold_low:.4f} ~ {threshold_high:.4f}")
    print(f"  large  > {threshold_high:.4f}")
    print(f"  reference value for a 5 mm lesion  = {CLINICAL_REFERENCE_RELATIVE_AREA:.4f}")
    print(f"  gap between data-driven and reference threshold: {abs(threshold_low - CLINICAL_REFERENCE_RELATIVE_AREA):.4f}")
    print(f"  (computed from n={len(relative_areas)} training GT boxes)")

    return threshold_low, threshold_high


def collect_confidence_correctness_with_size(model, images_dir, labels_dir):
    records = []
    image_files = sorted(Path(images_dir).glob("*.jpg")) + sorted(Path(images_dir).glob("*.png"))

    for img_path in image_files:
        result = model.predict(
            str(img_path), conf=CONF_THRESHOLD_FOR_EVAL, imgsz=IMGSZ, verbose=False
        )[0]
        img_h, img_w = result.orig_shape
        img_area = img_h * img_w

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
            box = pred_boxes[idx]
            box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            relative_area = box_area / img_area if img_area > 0 else 0

            if len(gt_boxes) == 0:
                records.append((conf, 0, relative_area))
                continue
            row = ious[idx].copy()
            row[matched_gt] = -1
            best_j = np.argmax(row)
            if row[best_j] >= IOU_MATCH_THRESHOLD:
                matched_gt[best_j] = True
                records.append((conf, 1, relative_area))
            else:
                records.append((conf, 0, relative_area))

    return records


def stratify_and_compute_dece(records, threshold_low, threshold_high):
    strata = {"small": [], "medium": [], "large": []}
    for conf, is_tp, rel_area in records:
        if rel_area < threshold_low:
            strata["small"].append((conf, is_tp))
        elif rel_area < threshold_high:
            strata["medium"].append((conf, is_tp))
        else:
            strata["large"].append((conf, is_tp))

    results = {}
    for stratum_name, stratum_records in strata.items():
        if len(stratum_records) == 0:
            results[stratum_name] = {
                "n": 0, "dece_point_estimate": None,
                "dece_bootstrap_mean": None, "ci_95_low": None, "ci_95_high": None,
            }
            continue
        dece_point, _ = compute_dece(stratum_records)
        dece_mean, ci_low, ci_high = bootstrap_dece_ci(stratum_records)
        results[stratum_name] = {
            "n": len(stratum_records),
            "dece_point_estimate": dece_point,
            "dece_bootstrap_mean": dece_mean,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
        }
    return results


def analyze_single_checkpoint_stratified(run_folder_name, threshold_low, threshold_high,
                                           runs_root, val_images, val_labels,
                                           model_variant="lag_lgff"):
    ckpt_path = runs_root / run_folder_name / "final.pt"
    if not ckpt_path.exists():
        print(f"  [warn] not found: {ckpt_path}，skipping")
        return None

    joint, base_dm, share_until = load_joint_model(ckpt_path, model_variant=model_variant)
    model = transplant_detector(joint, base_dm)

    records = collect_confidence_correctness_with_size(model, val_images, val_labels)
    stratified = stratify_and_compute_dece(records, threshold_low, threshold_high)

    return {"run": run_folder_name, "share_until": share_until, "stratified": stratified}


def run_group_stratified(dataset, group_name, threshold_low, threshold_high):
    cfg = DATASET_CONFIG[dataset]
    group = cfg["groups"][group_name]
    runs_root = cfg["runs_root"]
    val_images = cfg["val_images"]
    val_labels = cfg["val_labels"]

    all_results = {}

    for config_name, run_folders in group.items():
        print(f"\n=== config: {config_name} ===")
        model_variant = "gapfc" if "gapfc" in config_name else "lag_lgff"
        config_results = []
        for run_folder in run_folders:
            print(f"  analysing {run_folder} ...")
            res = analyze_single_checkpoint_stratified(
                run_folder, threshold_low, threshold_high,
                runs_root, val_images, val_labels,
                model_variant=model_variant,
            )
            if res is not None:
                config_results.append(res)
                s = res["stratified"]
                for size_name in ["small", "medium", "large"]:
                    d = s[size_name]
                    if d["dece_point_estimate"] is not None:
                        print(f"    {size_name}: D-ECE={d['dece_point_estimate']:.4f} (n={d['n']})")
                    else:
                        print(f"    {size_name}: no data")
        all_results[config_name] = config_results

    output_path = OUTPUT_DIR / f"{dataset}_{group_name}_size_stratified_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nresults written to {output_path}")

    print(f"\n=== [{dataset}] {group_name} cross-seed summary (by size stratum)===")
    for config_name, results in all_results.items():
        print(f"\n  {config_name}:")
        for size_name in ["small", "medium", "large"]:
            deces = [
                r["stratified"][size_name]["dece_point_estimate"]
                for r in results
                if r is not None and r["stratified"][size_name]["dece_point_estimate"] is not None
            ]
            if deces:
                print(f"    {size_name}: D-ECE = {np.mean(deces):.4f} ± {np.std(deces):.4f} (n={len(deces)} seeds)")
            else:
                print(f"    {size_name}: no valid data")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kvasir", "cvc", "etis"], default="etis",
                        help="etis recommended: the only dataset covering substantially smaller lesions")
    parser.add_argument("--group", default=None,
                        help="for kvasir choose exp1/exp2/exp3; cvc and etis have one group each")
    args = parser.parse_args()

    dataset = args.dataset
    cfg = DATASET_CONFIG[dataset]

    if args.group is None:
        group_name = list(cfg["groups"].keys())[0] if dataset != "kvasir" else "exp1"
    else:
        group_name = args.group

    print(f"Step 1: computing size thresholds from the [{dataset}] training set")
    threshold_low, threshold_high = compute_size_thresholds(dataset=dataset)

    print(f"\nStep 2: stratified calibration analysis for [{dataset}] / {group_name}")
    run_group_stratified(dataset, group_name, threshold_low, threshold_high)
