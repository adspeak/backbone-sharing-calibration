# -*- coding: utf-8 -*-
"""
S4 Wilcoxon 表重算 —— 每图 top-1 + 配对图像交集协议（v28）

不修改 wilcoxon_sensitivity.py。零推理，无卡模式即可。
读 raw_records_198_img/（唯一带 image_idx 的缓存）。

用法:
    cd /root/autodl-tmp/code
    python3 wilcoxon_v28.py > wilcoxon_v28_out.txt 2>&1

输出三段:
  [1] 自检 —— 配对 t 检验 p 值，应与正文/检验清单一致
  [2] 16 项 Wilcoxon 结果
  [3] 可直接粘贴的 LaTeX 表体
"""
import sys
import numpy as np
from pathlib import Path
from scipy import stats

RECORDS = Path("/root/autodl-tmp/calibration_results/raw_records_198_img")
S8 = [42, 43, 44, 45, 46, 47, 48, 49]
S6 = [42, 43, 44, 45, 46, 47]
NB, EPS = 10, 1e-12

# ---------------------------------------------------------------- 配置名发现
# 文件名形如 kvasir_<config>_seed<NN>_records.npz
def discover():
    cfgs = {}
    for p in sorted(RECORDS.glob("kvasir_*_records.npz")):
        stem = p.name[len("kvasir_"):-len("_records.npz")]
        cfg, _, seed = stem.rpartition("_seed")
        cfgs.setdefault(cfg, []).append(int(seed))
    return {k: sorted(v) for k, v in cfgs.items()}

def pick(cfgs, *cands):
    """按候选名依次匹配，返回实际存在的配置名。"""
    for c in cands:
        if c in cfgs:
            return c
    low = {k.lower(): k for k in cfgs}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

# ---------------------------------------------------------------- 读取 / 协议
def load_raw(cfg, seed):
    p = RECORDS / f"kvasir_{cfg}_seed{seed}_records.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return (d["confidence"].astype(float),
            d["is_tp"].astype(int),
            d["image_idx"].astype(int))

def top1(cfg, seed):
    """每图保留最高置信度的一条预测 -> dict{image_idx: (conf, is_tp)}"""
    r = load_raw(cfg, seed)
    if r is None:
        return None
    c, l, idx = r
    best = {}
    for ci, li, ii in zip(c, l, idx):
        if ii not in best or ci > best[ii][0]:
            best[ii] = (ci, li)
    return best

def matched(cfg_a, cfg_b, seed):
    """两配置同种子的图像交集上的 top-1 向量。"""
    A, B = top1(cfg_a, seed), top1(cfg_b, seed)
    if A is None or B is None:
        return None
    keys = sorted(set(A) & set(B))
    if not keys:
        return None
    ca = np.array([A[k][0] for k in keys]); la = np.array([A[k][1] for k in keys])
    cb = np.array([B[k][0] for k in keys]); lb = np.array([B[k][1] for k in keys])
    return (ca, la), (cb, lb), len(keys)

# ---------------------------------------------------------------- 指标
def dece(c, l, nb=NB):
    n = len(c); e = 0.0; ed = np.linspace(0, 1, nb + 1)
    for i in range(nb):
        lo, hi = ed[i], ed[i + 1]
        m = (c >= lo) & (c <= hi) if i == nb - 1 else (c >= lo) & (c < hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(c[m].mean() - l[m].mean())
    return e

def aece(c, l, nb=NB):
    n = len(c); o = np.argsort(c); cs, ls = c[o], l[o]; e = 0.0
    for idx in np.array_split(np.arange(n), nb):
        if len(idx) == 0:
            continue
        e += (len(idx) / n) * abs(cs[idx].mean() - ls[idx].mean())
    return e

def brier(c, l):
    return float(np.mean((c - l) ** 2))

def nll(c, l):
    p = np.clip(c, EPS, 1 - EPS)
    return float(-np.mean(l * np.log(p) + (1 - l) * np.log(1 - p)))

METRICS = [("D-ECE", dece), ("Adaptive ECE", aece), ("Brier", brier), ("NLL", nll)]

# ---------------------------------------------------------------- 配对差值
def deltas(cfg_a, cfg_b, seeds, fn):
    """每个种子: metric(B) - metric(A)，在两者图像交集上算。"""
    out, ns = [], []
    for s in seeds:
        m = matched(cfg_a, cfg_b, s)
        if m is None:
            return None, None
        (ca, la), (cb, lb), n = m
        out.append(fn(cb, lb) - fn(ca, la))
        ns.append(n)
    return np.array(out), ns

def tests(d):
    """对差值向量做单样本 t 与 Wilcoxon（对零）。"""
    t, pt = stats.ttest_1samp(d, 0.0)
    try:
        w, pw = stats.wilcoxon(d)
    except ValueError:          # 全零差值
        w, pw = float("nan"), 1.0
    dz = float(np.mean(d) / np.std(d, ddof=1)) if np.std(d, ddof=1) > 0 else float("nan")
    return float(t), float(pt), float(pw), dz

# ---------------------------------------------------------------- 主流程
def main():
    if not RECORDS.exists():
        sys.exit(f"缓存目录不存在: {RECORDS}")

    cfgs = discover()
    print("=" * 72)
    print("发现的配置（配置名 -> 种子）")
    print("=" * 72)
    for k, v in sorted(cfgs.items()):
        print(f"  {k:32s} {v}")
    print()

    A  = pick(cfgs, "A_independent")
    B  = pick(cfgs, "B_shared")
    GA = pick(cfgs, "gapfc_A_independent", "gapfc_A", "gapfcA_independent")
    GB = pick(cfgs, "gapfc_B_shared", "gapfc_B", "gapfcB_shared")
    F  = pick(cfgs, "F_frozen", "F_frozen_shared", "frozen")

    missing = [n for n, v in [("A", A), ("B", B), ("gapfc-A", GA),
                              ("gapfc-B", GB), ("F-frozen", F)] if v is None]
    if missing:
        sys.exit(f"下列配置在缓存里找不到，请把上面的配置名列表贴回来: {missing}")

    print(f"映射: A={A}  B={B}  gapfc-A={GA}  gapfc-B={GB}  F-frozen={F}\n")

    rows = []
    for mname, fn in METRICS:
        # 1) Independent vs shared (E1), n=8
        d1, n1 = deltas(A, B, S8, fn)
        # 2) Residual under GAP+FC (E2), n=8
        d2, n2 = deltas(GA, GB, S8, fn)
        # 3) Frozen backbone vs shared (E3), n=6  —— F 相对 B
        d3, n3 = deltas(B, F, S6, fn)
        # 4) Interaction: 每种子 D_s = Delta_LAG - Delta_GAP, n=8
        d4 = d1 - d2
        for cname, d, ns in [("Independent vs shared", d1, n1),
                             ("Residual under GAP+FC", d2, n2),
                             ("Frozen backbone vs shared", d3, n3),
                             ("Interaction", d4, None)]:
            t, pt, pw, dz = tests(d)
            rows.append((mname, cname, float(np.mean(d)), t, pt, pw, dz,
                         len(d), ns))

    # --------- [1] 自检
    print("=" * 72)
    print("[1] 自检 —— 下列 t 检验 p 值应与正文 / 检验清单一致")
    print("    参考值: D-ECE Ind-vs-shared p=0.0050 | Interaction p=0.0138")
    print("            D-ECE Residual GAP+FC p=0.2425 | Frozen vs shared p=0.5532")
    print("=" * 72)
    print(f"{'metric':13s} {'comparison':26s} {'mean':>9s} {'t':>7s} "
          f"{'p(t)':>8s} {'dz':>6s} {'n':>3s} {'N images':>18s}")
    for m, c, mean, t, pt, pw, dz, n, ns in rows:
        nstr = f"{min(ns)}-{max(ns)}" if ns else "(derived)"
        print(f"{m:13s} {c:26s} {mean:+9.4f} {t:7.3f} {pt:8.4f} {dz:6.2f} "
              f"{n:3d} {nstr:>18s}")
    print()

    # --------- [2] Wilcoxon
    print("=" * 72)
    print("[2] Wilcoxon signed-rank（新协议）—— 与 t 检验判定是否一致")
    print("=" * 72)
    print(f"{'metric':13s} {'comparison':26s} {'p(t)':>8s} {'p(W)':>8s} "
          f"{'n':>3s}  verdict")
    n_disagree = 0
    for m, c, mean, t, pt, pw, dz, n, ns in rows:
        agree = (pt < 0.05) == (pw < 0.05)
        if not agree:
            n_disagree += 1
        print(f"{m:13s} {c:26s} {pt:8.4f} {pw:8.4f} {n:3d}  "
              f"{'agree' if agree else '*** DISAGREE ***'}")
    print()
    print(f"判定不一致的项数: {n_disagree} / 16")
    n0078 = sum(1 for r in rows if abs(r[5] - 0.0078125) < 1e-6)
    print(f"Wilcoxon p 恰为 0.0078（n=8 的下限）的项数: {n0078}")
    print()
    if n_disagree == 0:
        print(">>> 正文 §5 开头『all sixteen cases』与 §6.5『agree throughout』可保留原样")
    else:
        print(">>> 正文两处『全部一致』的断言必须改为『除下列 %d 项外一致』" % n_disagree)
    print()

    # --------- [3] LaTeX
    print("=" * 72)
    print("[3] 粘贴到 article 的 tab:supp-wilcoxon（整块替换 tabular 内容）")
    print("=" * 72)
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Metric & Comparison & $t$-test $p$ & Wilcoxon $p$ & $n$ \\")
    order = ["Independent vs shared", "Residual under GAP+FC",
             "Frozen backbone vs shared", "Interaction"]
    for mi, (mname, _) in enumerate(METRICS):
        print(r"\midrule")
        first = True
        for cname in order:
            r = next(x for x in rows if x[0] == mname and x[1] == cname)
            lab = mname if first else ""
            first = False
            print(f"{lab:12s} & {cname:25s} & {r[4]:.4f} & {r[5]:.4f} & {r[7]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    main()
