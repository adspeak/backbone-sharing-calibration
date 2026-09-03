"""
Image-level bootstrap and image-wise cross-fitting for temperature scaling.

Predictions from the same image are not statistically independent, which bears
on two parts of the analysis:

  1. Bootstrap resampling. Resampling individual predictions ignores the
     clustering by image. This script computes intervals both ways -- resampling
     predictions, and resampling whole images with all of their predictions
     attached -- and reports the ratio of interval widths.

  2. Temperature-scaling folds. If folds are formed over predictions, the same
     image can contribute to both the fitting and the evaluation of a
     temperature. Here folds are formed over images.

Reads the cached records; no inference, no GPU.

Usage:
    python analyse_image_level.py
"""
import json
import numpy as np
from pathlib import Path
from scipy import optimize, stats
from paths import CALIB, RECORDS_DIR

OUT = CALIB
REC = RECORDS_DIR

SEEDS_8 = [42, 43, 44, 45, 46, 47, 48, 49]
SEEDS_6 = [42, 43, 44, 45, 46, 47]
N_BINS, N_BOOT, N_FOLDS = 10, 1000, 5
EPS_M, EPS_T = 1e-12, 1e-6
RNG_SEED = 20260830

CONFIGS = [
    ("A_independent", SEEDS_8, "A (independent)"),
    ("C_shallow", SEEDS_6, "C-shallow"),
    ("C_deep", SEEDS_6, "C-deep"),
    ("B_shared", SEEDS_8, "B (fully shared)"),
    ("gapfc_A_independent", SEEDS_8, "gapfc-A"),
    ("gapfc_B_shared", SEEDS_8, "gapfc-B"),
    ("F_frozen", SEEDS_6, "F-frozen"),
]


def load(run):
    p = REC / f"kvasir_{run}_records.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return (d["confidence"].astype(float), d["is_tp"].astype(int),
            d["image_idx"].astype(int))


# ---------- metrics ----------
def dece(c, l, nb=N_BINS):
    n = len(c)
    if n == 0: return None
    ed = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        lo, hi = ed[i], ed[i + 1]
        m = (c >= lo) & (c <= hi) if i == nb - 1 else (c >= lo) & (c < hi)
        if m.sum() == 0: continue
        e += (m.sum() / n) * abs(c[m].mean() - l[m].mean())
    return float(e)


def aece(c, l, nb=N_BINS):
    n = len(c)
    if n == 0: return None
    o = np.argsort(c); cs, ls = c[o], l[o]; e = 0.0
    for idx in np.array_split(np.arange(n), nb):
        if len(idx) == 0: continue
        e += (len(idx) / n) * abs(cs[idx].mean() - ls[idx].mean())
    return float(e)


def brier(c, l):
    return float(np.mean((c - l) ** 2)) if len(c) else None


def nll(c, l):
    if len(c) == 0: return None
    p = np.clip(c, EPS_M, 1 - EPS_M)
    return float(-np.mean(l * np.log(p) + (1 - l) * np.log(1 - p)))


METRICS = [("D-ECE", dece), ("Adaptive ECE", aece), ("Brier", brier), ("NLL", nll)]


# ---------- the two bootstrap schemes ----------
def boot_prediction(c, l, fn, rng, nb=N_BOOT):
    """Resample individual predictions."""
    n = len(c); vals = []
    for _ in range(nb):
        i = rng.randint(0, n, n)
        v = fn(c[i], l[i])
        if v is not None: vals.append(v)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_image(c, l, img, fn, rng, nb=N_BOOT):
    """Resample whole images, carrying all of an image's predictions together."""
    uniq = np.unique(img)
    by_img = {u: np.where(img == u)[0] for u in uniq}
    k = len(uniq); vals = []
    for _ in range(nb):
        pick = rng.randint(0, k, k)
        idx = np.concatenate([by_img[uniq[j]] for j in pick])
        v = fn(c[idx], l[idx])
        if v is not None: vals.append(v)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ---------- temperature scaling with image-wise folds ----------
def logit(c):
    c = np.clip(c, EPS_T, 1 - EPS_T)
    return np.log(c / (1 - c))


def sigm(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_T(z, y):
    def obj(lt):
        p = np.clip(sigm(z / np.exp(lt)), EPS_T, 1 - EPS_T)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    r = optimize.minimize_scalar(obj, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(r.x))


def ts_by_image(c, l, img, seed, nf=N_FOLDS):
    """Image-wise folds: all predictions from one image go to the same fold."""
    uniq = np.unique(img)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    folds = np.array_split(perm, nf)
    z = logit(c)
    cal = np.zeros(len(c)); temps = []
    for f in range(nf):
        te_imgs = set(uniq[folds[f]].tolist())
        te = np.array([i for i in range(len(c)) if img[i] in te_imgs])
        tr = np.array([i for i in range(len(c)) if img[i] not in te_imgs])
        if len(te) == 0: continue
        if len(tr) == 0 or len(np.unique(l[tr])) < 2:
            cal[te] = c[te]; temps.append(1.0); continue
        T = fit_T(z[tr], l[tr]); temps.append(T)
        cal[te] = sigm(z[te] / T)
    return cal, temps


def auc(c, l):
    pos, neg = c[l == 1], c[l == 0]
    if len(pos) == 0 or len(neg) == 0: return None
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def paired(a, b, label):
    pr = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pr) < 2:
        print(f"    {label}: too few pairs"); return
    A = np.array([p[0] for p in pr]); B = np.array([p[1] for p in pr])
    t, p = stats.ttest_rel(B, A)
    d = np.mean(B - A) / np.std(B - A, ddof=1)
    print(f"    {label:<26} {A.mean():.4f} → {B.mean():.4f}  "
          f"t={t:.3f} p={p:.4f} d={d:.3f}  {'sig.' if p<0.05 else 'n.s.'} (n={len(pr)})")


if __name__ == "__main__":
    rng = np.random.RandomState(RNG_SEED)

    print("=" * 96)
    print("Image-level vs prediction-level bootstrap")
    print(f"B={N_BOOT}, fixed random seed {RNG_SEED}")
    print("=" * 96)

    width_ratio = []
    supp = {}
    for cfg, seeds, disp in CONFIGS:
        print(f"\n### {disp}")
        rows = []
        for s in seeds:
            r = load(f"{cfg}_seed{s}")
            if r is None:
                print(f"  [warn] missing {cfg}_seed{s}"); continue
            c, l, img = r
            row = {"seed": s, "n_pred": int(len(c)), "n_img": int(len(np.unique(img)))}
            for mn, mf in METRICS:
                pt = mf(c, l)
                plo, phi = boot_prediction(c, l, mf, rng)
                ilo, ihi = boot_image(c, l, img, mf, rng)
                row[mn] = {"point": pt, "pred_ci": [plo, phi], "img_ci": [ilo, ihi]}
                if mn == "D-ECE":
                    width_ratio.append((ihi - ilo) / (phi - plo))
            rows.append(row)
            r0 = row["D-ECE"]
            print(f"  seed{s}: D-ECE={r0['point']:.4f}  "
                  f"pred-CI [{r0['pred_ci'][0]:.4f}, {r0['pred_ci'][1]:.4f}]  "
                  f"img-CI [{r0['img_ci'][0]:.4f}, {r0['img_ci'][1]:.4f}]")
        supp[cfg] = {"display": disp, "rows": rows}

    print("\n" + "=" * 96)
    print(f"Mean ratio of image-level to prediction-level D-ECE interval width: "
          f"{np.mean(width_ratio):.2f} "
          f"(range {min(width_ratio):.2f}–{max(width_ratio):.2f})")
    print("=" * 96)

    with open(OUT / "image_level_bootstrap.json", "w") as f:
        json.dump(supp, f, indent=2)
    print(f"written to {OUT}/image_level_bootstrap.json")

    # ---------- temperature scaling with image-wise folds ----------
    print("\n" + "=" * 96)
    print("Temperature scaling with image-wise folds")
    print("=" * 96)

    ts = {}
    for cfg, seeds, disp in CONFIGS:
        Ts, before_d, after_d, before_b, after_b, aucs = [], [], [], [], [], []
        for s in seeds:
            r = load(f"{cfg}_seed{s}")
            if r is None: continue
            c, l, img = r
            cal, temps = ts_by_image(c, l, img, seed=s)
            Ts.append(np.mean(temps))
            before_d.append(dece(c, l)); after_d.append(dece(cal, l))
            before_b.append(brier(c, l)); after_b.append(brier(cal, l))
            aucs.append(auc(c, l))
        ts[cfg] = {"T": Ts, "bd": before_d, "ad": after_d,
                   "bb": before_b, "ab": after_b, "auc": aucs}
        print(f"\n  {disp} (n={len(Ts)}):")
        print(f"    T = {np.mean(Ts):.3f} ± {np.std(Ts):.3f}")
        print(f"    D-ECE {np.mean(before_d):.4f} → {np.mean(after_d):.4f} "
              f"(reduced by {(1-np.mean(after_d)/np.mean(before_d))*100:.1f}%)")
        print(f"    Brier {np.mean(before_b):.4f} → {np.mean(after_b):.4f} "
              f"(reduced by {(1-np.mean(after_b)/np.mean(before_b))*100:.1f}%)")
        print(f"    AUC   = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    print("\n" + "=" * 96)
    print("Key comparisons after temperature scaling")
    print("=" * 96)
    for a, b, tag in [("A_independent", "B_shared", "LAG-LGFF head"),
                      ("gapfc_A_independent", "gapfc_B_shared", "GAP+FC head")]:
        print(f"\n  【{tag}】")
        paired(ts[a]["bd"], ts[b]["bd"], "D-ECE (before TS)")
        paired(ts[a]["ad"], ts[b]["ad"], "D-ECE (after TS)")
        paired(ts[a]["bb"], ts[b]["bb"], "Brier (before TS)")
        paired(ts[a]["ab"], ts[b]["ab"], "Brier (after TS)")
        paired(ts[a]["auc"], ts[b]["auc"], "AUC")

    with open(OUT / "ts_image_folds.json", "w") as f:
        json.dump({k: {kk: list(map(float, vv)) for kk, vv in v.items()}
                   for k, v in ts.items()}, f, indent=2)
    print(f"\nwritten to {OUT}/ts_image_folds.json")