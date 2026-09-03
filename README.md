# Task Competition and Calibration Under Backbone Sharing

Code accompanying *Task Competition Contributes to Calibration Degradation Under
Backbone Sharing in Polyp Detection*.

This repository reproduces every experiment and statistical analysis in the
paper: training joint polyp-detection / gastrointestinal-classification models at
four backbone-sharing depths, computing four calibration metrics on the resulting
detectors, and running the statistical comparisons reported in the manuscript.

---

## What is here

The study asks what sharing a detection backbone with an auxiliary
classification task does to the calibration of the detector's confidence scores,
and how much of any degradation is attributable to competition between the tasks
rather than to the capacity that sharing removes.

Seven configurations are trained, each across multiple random seeds:

| Configuration | Sharing depth `k` | Classification head | Seeds |
|---|---|---|---|
| `A_independent` | 0 (separate trunks) | LAG-LGFF | 42–49 |
| `C_shallow` | 5 | LAG-LGFF | 42–47 |
| `C_deep` | 9 | LAG-LGFF | 42–47 |
| `B_shared` | 11 (fully shared) | LAG-LGFF | 42–49 |
| `gapfc_A_independent` | 0 | GAP+FC | 42–49 |
| `gapfc_B_shared` | 11 | GAP+FC | 42–49 |
| `F_frozen` | 11, trunk frozen | LAG-LGFF | 42–47 |

Calibration is assessed with D-ECE (equal-width bins), Adaptive ECE (equal-mass
bins), the Brier score and negative log-likelihood, on the detection validation
set.

---

## Setup

### Environment

```bash
pip install -r requirements.txt
```

The detection framework version is **pinned deliberately**. Mixing versions of
`ultralytics` produced spurious differences in detection accuracy in earlier work
from this line, so `requirements.txt` fixes `ultralytics==8.4.104`. Results are
not guaranteed to reproduce under other versions.

### Workspace root

All paths derive from a single environment variable:

```bash
export BACKBONE_CALIB_ROOT=/path/to/your/workspace
```

If unset, the code falls back to `./workspace` inside the repository. The
expected layout is:

```
$BACKBONE_CALIB_ROOT/
    data/
        kvasir_split/              classification data, one folder per class
            train/ val/ test/
        kvasir_seg_yolo/           detection data, YOLO format
            images/train  images/val
            labels/train  labels/val
            dataset.yaml
    runs/                          checkpoints          (created automatically)
    results/                       training summaries   (created automatically)
    calibration_results/           analysis outputs     (created automatically)
    cache/                         GPU data cache       (created automatically)
```

### Data

Both datasets are publicly available and must be downloaded separately:

- **Kvasir v2** (8-class classification): Pogorelov et al., MMSys'17.
  <https://datasets.simula.no/kvasir/>
- **Kvasir-SEG** (polyp segmentation, converted to detection boxes):
  Jha et al., MMM 2020. <https://datasets.simula.no/kvasir-seg/>

Two further datasets are used for the additional-dataset evaluation:

- **CVC-ClinicDB**: Bernal et al., Comput Med Imaging Graph 2015.
- **ETIS-LaribPolypDB**: Silva et al., IJCARS 2014.

`data/split_manifest.json` records the exact train/validation/test assignment
used in the paper, generated once with seed 42. Detection boxes are the tightest
axis-aligned box around each mask region at binarisation threshold 127.

---

## Reproducing the paper

### 1. Training

```bash
cd src
python run_depth_sweep.py       # A, C-shallow, C-deep, B  (seeds 42-47)
python run_head_ablation.py     # gapfc-A, gapfc-B         (seeds 42-47)
python run_frozen_control.py    # F-frozen                 (seeds 42-47)
python run_seed_extension.py    # seeds 48-49 for the four primary configs
```

Each run trains for 15,000 steps (1,875 optimiser updates at 8-step gradient
accumulation) and takes roughly 45 minutes on an RTX 4090D. All four scripts skip
configurations whose checkpoint already exists, so they are safe to re-run.

### 2. Calibration records

```bash
python compute_calibration.py --group exp1   # depth sweep
python compute_calibration.py --group exp2   # head ablation
python compute_calibration.py --group exp3   # frozen control

python compute_brier_reliability.py --group exp1
python compute_brier_reliability.py --group exp2
python compute_brier_reliability.py --group exp3
```

`compute_brier_reliability.py` caches the `(confidence, is_true_positive,
image_index)` records for every model. **Once these exist, every remaining
analysis runs without a GPU.**

Skip this step if you only want to reproduce the statistical analysis: the
records are shipped with the repository (see below).

### 3. Statistical analysis

```bash
python stats_dece.py              # D-ECE: paired tests, TOST, interaction
python stats_brier.py             # same tests on the Brier score
python stats_nll_adaptive_ece.py  # NLL and Adaptive ECE
python stats_trend.py             # linear trend across depths, unified 6-seed basis
python stats_wilcoxon.py          # non-parametric sensitivity analysis
```

### 4. Post-hoc calibration and further analyses

```bash
python analyse_temperature_scaling.py   # 5-fold cross-fitted TS, folds over images
python analyse_image_level.py           # image-level vs prediction-level bootstrap
python analyse_by_lesion_size.py --dataset etis
```

### 5. Supplementary tables

```bash
python make_supplementary_tables.py   # S1: per-seed estimates with intervals
python make_test_inventory.py         # S2: inventory of all 26 tests
```

Both write LaTeX fragments into `calibration_results/`, ready to be `\input` from
the manuscript.

---

## Reproducing the analysis without a GPU

The cached prediction records for all 50 trained models are committed under
`data/raw_records/` (208 KB in total), and `src/paths.py` picks them up
automatically. After cloning, every statistical script runs immediately, with no
workspace, no datasets and no GPU:

```bash
cd src
python stats_dece.py
python stats_brier.py
python stats_nll_adaptive_ece.py
python stats_trend.py
python stats_wilcoxon.py
python analyse_temperature_scaling.py
python analyse_image_level.py
python make_test_inventory.py
python make_supplementary_tables.py   # after analyse_image_level.py has run once
```

`stats_dece.py` computes D-ECE directly from the cached `(confidence,
is_true_positive, image_index)` records (see `compute_dece()`), so E1-E3 need
nothing beyond this repository. Its E5 comparison (CVC/ETIS) is the one
exception: those two datasets were never cached as `.npz` records, so that part
is skipped with an explanation rather than run, and does not affect E1-E3.

`make_supplementary_tables.py` needs `image_level_bootstrap.json`, which
`analyse_image_level.py` writes on its first run — run that one first.

One script is genuinely not reproducible from this repository alone:
`analyse_by_lesion_size.py` re-runs model inference on the training images to
derive lesion-size tertiles, so it needs the actual datasets (and a GPU
environment with `torch`/`ultralytics` installed) rather than just the cached
records.

Each `.npz` holds the `(confidence, is_true_positive, image_index)` triples for
one trained model on the 198-image detection validation set. Every calibration
number in the paper is computed from these.

**What this does and does not reproduce.** It reproduces the analysis
*conditional on these predictions*: the metrics, the paired tests, the
interaction test, temperature scaling, and the bootstrap intervals. It does not
reproduce the predictions themselves, which requires training the models from
the datasets as described above. The two are separate claims, and only the first
is verifiable from this repository alone.

---

## Which script produces which result

| Manuscript element | Script |
|---|---|
| Table 1 (parameter counts) | derived from checkpoints; see `model.py` |
| Table 2 (calibration by sharing depth) | `compute_calibration.py`, `compute_brier_reliability.py`, `stats_nll_adaptive_ece.py` |
| Table 3 (interaction) | `stats_dece.py`, `stats_brier.py`, `stats_nll_adaptive_ece.py` |
| Table 4 (temperature scaling) | `analyse_temperature_scaling.py` |
| Table 5 (additional datasets) | `compute_calibration.py --group exp5` |
| Table 6 (lesion size) | `analyse_by_lesion_size.py --dataset etis` |
| Linear trend tests | `stats_trend.py` |
| Overlap audit | see *Data overlap audit* below |
| Supplementary S1 | `make_supplementary_tables.py`, `analyse_image_level.py` |
| Supplementary S2 | `make_test_inventory.py` |
| Supplementary S4 | `stats_wilcoxon.py` |

---

## Data overlap audit

Kvasir-SEG is derived from the polyp class of Kvasir v2, so the two tasks draw on
overlapping source material. Because the two collections use different file
naming conventions, filename comparison is uninformative; the audit compares
image content using both MD5 and perceptual hashes:

```python
from PIL import Image
import imagehash, glob, os

def phash(p):
    return str(imagehash.phash(Image.open(p).convert("RGB")))

cls_train = {phash(f) for f in glob.glob(".../kvasir_split/train/polyps/*")}
det_val   = {phash(f) for f in glob.glob(".../kvasir_seg_yolo/images/val/*")}
print(len(cls_train & det_val))
```

Both hashing methods returned the same result: two images appear in both the
classification training set and the detection validation set. These were excluded
from all reported evaluations, reducing the detection validation set to 198
images. Their identifiers are listed in
`calibration_results/excluded_val_images.json` once the audit is run. The
cached records shipped here already reflect that exclusion: each covers 198
validation images.

---

## Notes on the code

**Model loading.** The joint model cannot be loaded with `ultralytics.YOLO()`
directly. `compute_calibration.py` rebuilds it via `load_joint_model()` and then
transplants the detection pathway into a standard `DetectionModel` with
`transplant_detector()`, a round-trip that was validated against the original
evaluation code.

**Seed counts differ by configuration.** The four configurations entering the
primary analysis use eight seeds; the intermediate depths and the frozen control
use six; the additional-dataset runs use three. Changing the seed count requires
editing four places: `CONFIG_GROUPS` in `compute_calibration.py`; the `SEEDS`
constant in `stats_brier.py` and `analyse_temperature_scaling.py`; and
`CONFIG_SEEDS` in `stats_dece.py`.

**Loss weighting.** The joint objective is
`L = L_cls + lambda * L_det` with `lambda = 0.0304`, fixed in advance by matching
the gradient norms the two loss terms contribute to the shared parameters, and
held constant across all sharing depths. A depth-dependent weighting would
confound the effect under study.

---

## Citation

```bibtex
@article{chen2026calibration,
  title   = {Task Competition Contributes to Calibration Degradation Under
             Backbone Sharing in Polyp Detection},
  author  = {Chen, Yude},
  journal = {TBD},
  year    = {2026}
}
```

Code: https://github.com/<your-username>/backbone-sharing-calibration

---

## License

MIT. See `LICENSE`.

The datasets used are distributed under their own terms; please consult the
original sources.
