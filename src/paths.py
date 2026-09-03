"""
Centralised path configuration.

Most paths derive from a single project root, read from the BACKBONE_CALIB_ROOT
environment variable and falling back to ./workspace inside the repository:

    export BACKBONE_CALIB_ROOT=/path/to/your/workspace

Expected layout under that root:

    <root>/
        data/
            kvasir_split/            classification data, one folder per class
                train/ val/ test/
            kvasir_seg_yolo/         detection data, YOLO format
                images/train  images/val
                labels/train  labels/val
                dataset.yaml
        runs/                        checkpoints          (auto-created)
        results/                     training summaries   (auto-created)
        calibration_results/         analysis outputs     (auto-created)
        cache/                       GPU data cache       (auto-created)

RECORDS_DIR is the exception. The cached prediction records shipped with this
repository live under data/raw_records/, so the statistical scripts run
immediately after cloning, with no workspace and no GPU. If that directory is
absent -- for instance when regenerating records from freshly trained models --
the workspace location is used instead.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ROOT = Path(os.environ.get("BACKBONE_CALIB_ROOT", REPO / "workspace"))

DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
CALIB = ROOT / "calibration_results"
CACHE = ROOT / "cache"

CLS_DIR = DATA / "kvasir_split"
DET_DIR = DATA / "kvasir_seg_yolo"
DET_YAML = DET_DIR / "dataset.yaml"

# Cached (confidence, is_true_positive, image_index) records, one file per
# trained model. Every calibration metric in the paper is computed from these,
# so once they exist the entire statistical analysis runs without a GPU.
# Prefer the copy distributed with the repository; fall back to the workspace.
_BUNDLED_RECORDS = REPO / "data" / "raw_records"
RECORDS_DIR = _BUNDLED_RECORDS if _BUNDLED_RECORDS.is_dir() else CALIB / "raw_records"

for _d in (RUNS, RESULTS, CALIB, CACHE):
    _d.mkdir(parents=True, exist_ok=True)
if RECORDS_DIR is not _BUNDLED_RECORDS:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

# Kvasir v2 classification categories, in the order used throughout
CLASSES = [
    "dyed-lifted-polyps", "dyed-resection-margins", "esophagitis",
    "normal-cecum", "normal-pylorus", "normal-z-line", "polyps",
    "ulcerative-colitis",
]
POLYP_IDX = 6

IMGSZ = 224
MASK_THRESHOLD = 127
SPLIT_SEED = 42

# String aliases, for modules that build paths with f-strings
ROOT_STR = str(ROOT)
DATA_STR = str(DATA)
RUNS_STR = str(RUNS)
RESULTS_STR = str(RESULTS)
