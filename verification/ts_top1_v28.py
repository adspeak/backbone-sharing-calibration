# -*- coding: utf-8 -*-
"""
温度缩放重算 —— 每图 top-1 协议（v28）

背景: tab:ts 的 T 列与 TS 后两列在服务器上找不到任何来源文件，
      exp6_temperature_scaling_results.json 与 ts_image_folds.json 都是宽松口径。
      本脚本重算 top-1 口径，并同时产出 tab:ts 与正文所需的全部派生量。

方法沿用 exp6_temperature_scaling.py:
  - logit = log(c/(1-c))，EPS=1e-6 裁剪
  - 5 折交叉拟合，折按【图像】切分
  - 每折在其余四折上以 NLL 最小化拟合 T，在留出折上评估
  - 每种子的 T 取五折均值；TS 后指标在汇总的留出预测上计算

零推理，无卡模式即可。
用法:
    cd verification
    python3 ts_top1_v28.py > ts_top1_v28_out.txt 2>&1
"""
import json
import sys
import numpy as np
from pathlib import Path
from scipy import optimize, stats
from sklearn.model_selection import KFold

# Paths resolve through src/paths.py so the script runs from a fresh clone with
# no configuration: the cached records bundled under data/raw_records/ are used
# when present, otherwise the workspace named by BACKBONE_CALIB_ROOT.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import (CALIB, METRICS_DIR, RECORDS_DIR,  # noqa: E402
                   EXTERNAL_RECORDS_DIR)

RECORDS = RECORDS_DIR
S8 = [42, 43, 44, 45, 46, 47, 48, 49]
S6 = [42, 43, 44, 45, 46, 47]
NB, EPS, N_FOLDS, FOLD_SEED = 10, 1e-6, 5, 42

CONFIGS = [                      # (缓存名, 表里的显示名, 种子表)
    ("single_task",         "Detection-only baseline",    S8),
    ("A_independent",       "A (independent, k=0)",       S8),
    ("C_shallow",           "C-shallow (k=5)",            S6),
    ("C_deep",              "C-deep (k=9)",               S6),
    ("B_shared",            "B (fully shared, k=11)",     S8),
    ("gapfc_A_independent", "gapfc-A",                    S8),
    ("gapfc_B_shared",      "gapfc-B",                    S8),
    ("F_frozen",            "F-frozen",                   S6),
]

# ------------------------------------------------------------------ 指标
def dece(c, l, nb=NB):
    n = len(c); e = 0.0; ed = np.linspace(0, 1, nb + 1)
    for i in range(nb):
        lo, hi = ed[i], ed[i + 1]
        m = (c >= lo) & (c <= hi) if i == nb - 1 else (c >= lo) & (c < hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(c[m].mean() - l[m].mean())
    return float(e)

def brier(c, l):
    return float(np.mean((c - l) ** 2))

def auc(c, l):
    if l.sum() == 0 or l.sum() == len(l):
        return float("nan")
    return float(stats.mannwhitneyu(c[l == 1], c[l == 0],
                                    alternative="greater").statistic
                 / (l.sum() * (len(l) - l.sum())))

# ------------------------------------------------------------------ 读取
def load_raw(cfg, seed):
    p = RECORDS / f"kvasir_{cfg}_seed{seed}_records.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return (d["confidence"].astype(float),
            d["is_tp"].astype(int),
            d["image_idx"].astype(int))

def get_set(cfg, seed, protocol):
    """protocol='top1' 每图最高置信度一条; 'permissive' 全部预测。"""
    r = load_raw(cfg, seed)
    if r is None:
        return None
    c, l, idx = r
    if protocol == "permissive":
        return c, l, idx
    best = {}
    for ci, li, ii in zip(c, l, idx):
        if ii not in best or ci > best[ii][0]:
            best[ii] = (ci, li)
    keys = np.array(sorted(best))
    return (np.array([best[k][0] for k in keys]),
            np.array([best[k][1] for k in keys]),
            keys)

# ------------------------------------------------------------------ TS
def to_logit(c):
    c = np.clip(c, EPS, 1 - EPS)
    return np.log(c / (1 - c))

def fit_T(logit, label):
    def nll(logT):
        T = np.exp(logT[0])
        p = np.clip(1 / (1 + np.exp(-logit / T)), EPS, 1 - EPS)
        return -np.mean(label * np.log(p) + (1 - label) * np.log(1 - p))
    r = optimize.minimize(nll, x0=[0.0], method="Nelder-Mead")
    return float(np.exp(r.x[0]))

def ts_one_model(c, l, idx):
    """按图 5 折交叉拟合。返回 (T_mean, 校准后置信度数组 对齐输入顺序)."""
    logit = to_logit(c)
    imgs = np.unique(idx)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    cal = np.empty_like(c)
    Ts = []
    for tr_i, te_i in kf.split(imgs):
        tr_imgs, te_imgs = set(imgs[tr_i]), set(imgs[te_i])
        m_tr = np.array([i in tr_imgs for i in idx])
        m_te = np.array([i in te_imgs for i in idx])
        if m_tr.sum() == 0 or m_te.sum() == 0:
            continue
        T = fit_T(logit[m_tr], l[m_tr])
        Ts.append(T)
        cal[m_te] = 1 / (1 + np.exp(-logit[m_te] / T))
    return float(np.mean(Ts)), cal

def run_protocol(protocol):
    """返回 res[cfg][seed] = dict(T, before/after 指标, 逐图校准后置信度)."""
    res = {}
    for cfg, _, seeds in CONFIGS:
        res[cfg] = {}
        for s in seeds:
            g = get_set(cfg, s, protocol)
            if g is None:
                continue
            c, l, idx = g
            T, cal = ts_one_model(c, l, idx)
            res[cfg][s] = dict(
                T=T, n=len(c),
                bd=dece(c, l), ad=dece(cal, l),
                bb=brier(c, l), ab=brier(cal, l),
                auc=auc(c, l),
                per_img_before={i: (ci, li) for i, ci, li in zip(idx, c, l)},
                per_img_after={i: (ci, li) for i, ci, li in zip(idx, cal, l)},
            )
    return res

# ------------------------------------------------------------------ 配对
def paired(res, ca, cb, seeds, key, fn):
    """在两配置的图像交集上算 fn 差值 (B - A)。key: per_img_before/after"""
    d = []
    for s in seeds:
        if s not in res[ca] or s not in res[cb]:
            return None
        A, B = res[ca][s][key], res[cb][s][key]
        ks = sorted(set(A) & set(B))
        a = np.array([A[k][0] for k in ks]); al = np.array([A[k][1] for k in ks])
        b = np.array([B[k][0] for k in ks]); bl = np.array([B[k][1] for k in ks])
        d.append(fn(b, bl) - fn(a, al))
    return np.array(d)

def tt(d):
    t, p = stats.ttest_1samp(d, 0.0)
    sd = np.std(d, ddof=1)
    return float(np.mean(d)), float(t), float(p), float(np.mean(d) / sd) if sd > 0 else float("nan")

def paired_T(res, ca, cb, seeds):
    a = np.array([res[ca][s]["T"] for s in seeds])
    b = np.array([res[cb][s]["T"] for s in seeds])
    return a, b, tt(b - a)

# ------------------------------------------------------------------ 主
def main():
    if not RECORDS.exists():
        sys.exit(f"缓存不存在: {RECORDS}")

    # ---------- 自检: 用全部预测复现宽松口径的已知值 ----------
    print("=" * 78)
    print("[0] 自检 —— 用 raw_records_198_img 的全部预测重算宽松口径,")
    print("    与 ts_image_folds.json 对比。T 允许有差(折的随机种子不同),")
    print("    但 bd(TS 前 D-ECE) 必须几乎完全一致。")
    print("=" * 78)
    ref = json.load(open(METRICS_DIR / "ts_image_folds.json"))
    perm = run_protocol("permissive")
    print(f"{'config':22s} {'bd_new':>8s} {'bd_ref':>8s} {'差':>8s} "
          f"{'T_new':>7s} {'T_ref':>7s}")
    ok = True
    for cfg, _, seeds in CONFIGS:
        if cfg not in ref or not perm[cfg]:
            continue
        bd_new = np.mean([perm[cfg][s]["bd"] for s in perm[cfg]])
        T_new  = np.mean([perm[cfg][s]["T"]  for s in perm[cfg]])
        bd_ref = np.mean(ref[cfg]["bd"]); T_ref = np.mean(ref[cfg]["T"])
        diff = bd_new - bd_ref
        if abs(diff) > 0.002:
            ok = False
        print(f"{cfg:22s} {bd_new:8.4f} {bd_ref:8.4f} {diff:+8.4f} "
              f"{T_new:7.3f} {T_ref:7.3f}")
    print()
    print(">>> bd 全部对上" if ok else
          ">>> *** bd 对不上，两个缓存内容不同，先别用下面的结果 ***")
    print()

    # ---------- top-1 口径 ----------
    res = run_protocol("top1")

    print("=" * 78)
    print("[1] tab:ts 替换值（top-1 协议，各配置在自己的预测集上）")
    print("=" * 78)
    print(f"{'Configuration':28s} {'n':>2s} {'T':>6s} {'D-ECE前':>8s} {'D-ECE后':>8s} "
          f"{'Brier前':>8s} {'Brier后':>8s} {'AUC':>7s} {'改善%':>7s}")
    fix = []
    for cfg, disp, seeds in CONFIGS:
        r = res[cfg]
        if not r:
            continue
        m = lambda k: np.mean([r[s][k] for s in r])
        pct = 100 * np.mean([(r[s]["bd"] - r[s]["ad"]) / r[s]["bd"] for s in r])
        fix.append(pct)
        print(f"{disp:28s} {len(r):2d} {m('T'):6.3f} {m('bd'):8.4f} {m('ad'):8.4f} "
              f"{m('bb'):8.4f} {m('ab'):8.4f} {m('auc'):7.4f} {pct:7.1f}")
    print()
    print(f">>> 正文『reduces calibration error by X--Y%』应写: "
          f"{min(fix):.0f}--{max(fix):.0f}%")
    print()

    print("=" * 78)
    print("[2] 温度对比（正文 §5.4 / §5.9 / §6.1 所需）")
    print("=" * 78)
    for lab, ca, cb in [("LAG-LGFF  A->B", "A_independent", "B_shared"),
                        ("GAP+FC    A->B", "gapfc_A_independent", "gapfc_B_shared")]:
        a, b, (mn, t, p, dz) = paired_T(res, ca, cb, S8)
        print(f"{lab}: {a.mean():.3f} -> {b.mean():.3f}  "
              f"= {100*(b.mean()/a.mean()-1):+.1f}%   "
              f"(t={t:.3f}, p={p:.4f}, dz={dz:.2f}, n=8)")
    print()

    print("=" * 78)
    print("[3] TS 后的配对差值（匹配图像交集，正文 §5.4 所需）")
    print("=" * 78)
    for mlab, fn in [("D-ECE", dece), ("Brier", brier)]:
        dA = paired(res, "A_independent", "B_shared", S8, "per_img_after", fn)
        dG = paired(res, "gapfc_A_independent", "gapfc_B_shared", S8, "per_img_after", fn)
        bA = paired(res, "A_independent", "B_shared", S8, "per_img_before", fn)
        bG = paired(res, "gapfc_A_independent", "gapfc_B_shared", S8, "per_img_before", fn)
        for lab, d in [("LAG  TS前", bA), ("LAG  TS后", dA),
                       ("GAP  TS前", bG), ("GAP  TS后", dG),
                       ("交互 TS后", dA - dG)]:
            mn, t, p, dz = tt(d)
            print(f"{mlab:6s} {lab:10s} {mn:+8.4f}  t={t:7.3f}  p={p:.4f}  dz={dz:5.2f}")
        print()

    print("=" * 78)
    print("[4] AUC（TS 不改变排序，此为 TS 前后共同值；匹配交集配对检验）")
    print("=" * 78)
    for lab, ca, cb in [("LAG-LGFF", "A_independent", "B_shared"),
                        ("GAP+FC", "gapfc_A_independent", "gapfc_B_shared")]:
        d = paired(res, ca, cb, S8, "per_img_before", auc)
        a = np.mean([res[ca][s]["auc"] for s in S8])
        b = np.mean([res[cb][s]["auc"] for s in S8])
        mn, t, p, dz = tt(d)
        print(f"{lab:10s} {a:.4f} -> {b:.4f}   差 {mn:+.4f}  t={t:6.3f}  p={p:.4f}")
    print()

    print("=" * 78)
    print("[5] 逐种子温度（备查）")
    print("=" * 78)
    for cfg, disp, seeds in CONFIGS:
        r = res[cfg]
        if r:
            print(f"{disp:28s} " + " ".join(f"{r[s]['T']:6.3f}" for s in sorted(r)))

    json.dump({c: {str(s): {k: v for k, v in r.items()
                            if not k.startswith("per_img")}
                   for s, r in res[c].items()} for c in res},
              open(CALIB / "ts_top1_v28_results.json", "w"), indent=2)
    print(f"\n已写出: {CALIB/'ts_top1_v28_results.json'}")

if __name__ == "__main__":
    main()
