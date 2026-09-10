# Verification scripts

Recomputation of the inferential statistics reported in the manuscript, from the
cached confidence/correctness records. No model inference is required; every
script runs on CPU in seconds to a couple of minutes.

## Scripts

| script | what it produces |
|---|---|
| `wilcoxon_v28.py` | Supplementary S4: Wilcoxon signed-rank tests for the sixteen paired comparisons, under the per-image top-1 protocol used in the main text. Emits the LaTeX table body. |
| `ts_top1_v28.py` | Table 6 (temperature scaling) and the derived quantities in Section 5.4 — fitted temperatures, post-scaling gaps, AUC. Writes `ts_top1_v28_results.json`. |
| `verify_numbers_v29.py` | Regression check of ~180 inline statistics against the cached records: depth trend, variance homogeneity, TOST margins, the confidence/precision decomposition, accuracy contrasts, the detection-only baseline, prediction-count dependence, the threshold scan. |
| `verify_round2_v30.py` | Second pass: re-verifies the corrections made after round one and extends coverage to the permissive-protocol AUC, the bootstrap interval comparison and the per-seed interval spot check. |
| `verify_round3_v30.py` | Third pass: additional datasets under the matched-image protocol, the ETIS size-stratified analysis, lesion-area quantiles, and the perceptual-hash near-duplicate audit of the external validation splits. |

Each script prints a self-check first, reproducing a value that is already
cached, before producing new output. The `*_out.txt` files are the runs
corresponding to the submitted manuscript.

## Inputs

All scripts read from the analysis cache rather than from model checkpoints:

- `calibration_results/raw_records_198_img/` — 58 `.npz` files, one per trained
  model, each carrying `confidence`, `is_tp` and `image_idx` for every retained
  prediction on the 198-image Kvasir-SEG validation split
- `calibration_results/raw_records_external/` — the same for the CVC-ClinicDB and
  ETIS-LaribPolypDB runs
- `calibration_results/accuracy_metrics.json`, `single_task_accuracy.json` —
  per-seed mAP, precision, recall and classification top-1
- `calibration_results/ts_image_folds.json`,
  `exp6_temperature_scaling_results.json` — the permissive-protocol temperature
  scaling results, retained for the protocol comparison

The near-duplicate audit in `verify_round3_v30.py` additionally reads the
external image directories and requires `imagehash` and `Pillow`.

## Protocol note

The main text scores the highest-confidence detection on each image, and paired
contrasts are computed on the images where both configurations produce a
detection. The permissive protocol (every prediction above 0.001) is reported
alongside for comparability with the detection calibration literature. Both are
implemented in these scripts; where a script offers a choice, the default is the
protocol used in the main text.

## Reproducing a specific number

```bash
python3 verify_numbers_v29.py > out.txt 2>&1
grep FAIL out.txt        # any line here is a disagreement with the manuscript
```

A `PASS` line shows the manuscript value and the recomputed value side by side.
