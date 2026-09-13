# Step 3 — complete the repository so the verification scripts run from a clone

Acceptance test for the whole step: in a **fresh clone**, on a machine with no
workspace and no GPU, `python3 verification/verify_numbers_v29.py` reports
**0 FAIL**. Do not tag or archive before that passes.

---

## Part A — files to copy off the AutoDL instance

Everything lives under `/root/autodl-tmp/calibration_results/`. First, take an
inventory so we know the real filenames (the external records in particular):

```bash
cd /root/autodl-tmp/calibration_results
ls raw_records_198_img/ | grep single_task
ls raw_records_external/
ls *.json
du -sh raw_records_198_img raw_records_external
```

Paste that output back — the external naming is the one thing I can't predict
(`verify_round3_v30.py` expects `<ds>_<cfg>_<ds>_seed<NN>.npz`, which is an odd
doubled form, and `verify_round2_v30.py` guesses two other patterns).

### A1. Detection-only baseline records → `data/raw_records/`

8 files, same directory as the 50 already committed, same naming:

```
raw_records_198_img/kvasir_single_task_seed42_records.npz   →  data/raw_records/
...
raw_records_198_img/kvasir_single_task_seed49_records.npz   →  data/raw_records/
```

These are what `verify_numbers_v29.py` needs for the §5.7 baseline checks and
Table 4 Panel A/B. Roughly 33 KB total.

### A2. External dataset records → `data/raw_records_external/` (new directory)

12 files: 2 configurations (`A_independent`, `B_shared`) × 3 seeds (42–44) ×
2 datasets (`cvc`, `etis`). Copy them with whatever names they already have —
we adapt the scripts to the real names rather than renaming the data.

### A3. JSON inputs → `data/metrics/` (new directory)

Seven files are read by the verification scripts. I said "four" earlier; that
was the list in `verification/README.md`, and the scripts actually read more:

| File | Read by |
|---|---|
| `accuracy_metrics.json` | `verify_numbers_v29.py`, `verify_round2_v30.py` |
| `single_task_accuracy.json` | `verify_numbers_v29.py`, `verify_round2_v30.py` |
| `ts_image_folds.json` | `ts_top1_v28.py` |
| `image_level_bootstrap.json` | `verify_round2_v30.py`, `verify_round3_v30.py` |
| `supplementary_ci_results.json` | `verify_round3_v30.py` |
| `etis_exp5_etis_size_stratified_results.json` | `verify_round2_v30.py`, `verify_round3_v30.py` |
| `ts_top1_v28_results.json` | `verify_round2_v30.py` (already committed under `verification/`; copy it here too so one lookup rule covers every input) |

Check each one for absolute paths or machine identifiers before committing:

```bash
grep -l "autodl\|/root/" *.json
```

---

## Part B — DONE (files supplied)

`src/paths.py` gained `METRICS_DIR`, and all ten JSON read sites across four
scripts now use it; the single write site in `ts_top1_v28.py` still targets
`CALIB`. `verify_round2_v30.py` also learned the real external filename pattern
`<ds>_<cfg>_<ds>_seed<NN>.npz`. Just copy the supplied files over.

---

## Part C — DONE (files supplied): four expected values that v39 changed

These are hardcoded manuscript values inside the verification scripts. Left as
they are, the archive would contradict the published paper — the worst possible
failure mode for a reproducibility artifact.

| File / line | Now | Change to | Why |
|---|---|---|---|
| `verify_round3_v30.py:81` | CVC D-ECE `p = 0.214` | `0.206` | **Will FAIL once the external records land.** Tolerance is 0.003; the script computes 0.2056. This is the error we corrected in v39. |
| `verify_round2_v30.py:239` | CVC D-ECE `p = 0.214` | `0.206` | Same value, second site. |
| `verify_numbers_v29.py:174` | Levene `p = 0.006` | `0.005` | Passes either way (tolerance 0.001, computed 0.0054), but should match the manuscript. |
| `verify_numbers_v29.py:374` | MDE `0.0214`, tol `0.003` | `0.024`, tol `0.001` | The script's own formula gives 0.0241 and only passed by 0.0003. v39 now reports 0.024. |

For the last one, also align the method with what v39 states (non-central *t*):

```python
from scipy.optimize import brentq
sd = np.std(dm, ddof=1); se = sd / np.sqrt(8); tc = stats.t.ppf(0.975, 7)
power = lambda d: stats.nct.sf(tc, 7, d / se) + stats.nct.cdf(-tc, 7, d / se)
mde = brentq(lambda d: power(d) - 0.80, 1e-4, 0.10)
CHECK("80%功效可检出差 (non-central t)", float(mde), 0.024, 0.001)
```

---

## Part D — one missing dependency

`ts_top1_v28.py` imports `sklearn.model_selection.KFold`, but `requirements.txt`
has no scikit-learn. Add:

```
scikit-learn>=1.0
```

---

## Part E — acceptance test

From a clean clone, with no `BACKBONE_CALIB_ROOT` set:

```bash
cd verification
python3 wilcoxon_v28.py      > /tmp/w.txt 2>&1   # already passes today
python3 verify_numbers_v29.py > /tmp/v29.txt 2>&1
python3 verify_round2_v30.py  > /tmp/v30b.txt 2>&1
python3 verify_round3_v30.py  > /tmp/v30c.txt 2>&1
python3 ts_top1_v28.py        > /tmp/ts.txt 2>&1
grep -c FAIL /tmp/v29.txt /tmp/v30b.txt /tmp/v30c.txt
```

Expected: 0 FAIL everywhere. SKIPs are acceptable only where they are inherent —
the perceptual-hash audit in round 3 needs the actual dataset images and will
always SKIP without them; `analyse_by_lesion_size.py` likewise. Any other SKIP
means a file is still missing.

Then commit the refreshed `*_out.txt` files alongside the scripts, so the
archived outputs match the archived code.

---

## Then, in order

1. `.gitignore` += `*.tif`
2. Final README revision — add the `verification/` section, the
   `| Figures 2-4 | make_figures.py |` row, the S3/S5/S6 rows, and update the
   "50 models / 208 KB" and "CVC/ETIS were never cached" sentences to match what
   the repository now actually contains
3. Tag `v1.0.0`, publish a Release
4. figshare: upload, Reserve DOI, write the DOI into the manuscript, then Publish
