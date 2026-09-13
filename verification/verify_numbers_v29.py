# -*- coding: utf-8 -*-
"""
全文数字回归核对 (v29)
======================
把正文里的每个数字硬编码进来，与缓存重算结果逐条比对，打 PASS / FAIL。

数据来源（全部已缓存，零推理，无卡模式即可）：
  raw_records_198_img/        置信度 / is_tp / image_idx
  accuracy_metrics.json       各配置逐种子 mAP / precision / recall / cls_acc
  single_task_accuracy.json   检测-only 基线逐种子精度

用法:
    cd verification
    python3 verify_numbers_v29.py > verify_v29_out.txt 2>&1
    grep -c FAIL verify_v29_out.txt
"""
import json, sys, itertools
import numpy as np
from pathlib import Path
from scipy import stats, optimize

# Paths resolve through src/paths.py so the script runs from a fresh clone with
# no configuration: the cached records bundled under data/raw_records/ are used
# when present, otherwise the workspace named by BACKBONE_CALIB_ROOT.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import (CALIB, METRICS_DIR, RECORDS_DIR,  # noqa: E402
                   EXTERNAL_RECORDS_DIR)

REC = RECORDS_DIR
S8 = [42, 43, 44, 45, 46, 47, 48, 49]
S6 = [42, 43, 44, 45, 46, 47]
NB, EPS = 10, 1e-12

A, B   = "A_independent", "B_shared"
GA, GB = "gapfc_A_independent", "gapfc_B_shared"
CS, CD, FR, ST = "C_shallow", "C_deep", "F_frozen", "single_task"

# ------------------------------------------------------------------ 比对框架
_n_pass = _n_fail = 0
def CHECK(label, computed, paper, tol):
    global _n_pass, _n_fail
    if computed is None or (isinstance(computed, float) and np.isnan(computed)):
        print(f"  SKIP  {label:52s} 无法计算")
        return
    ok = abs(computed - paper) <= tol
    _n_pass, _n_fail = _n_pass + ok, _n_fail + (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:52s} 论文={paper:<10.4f} 重算={computed:<10.4f} "
          f"差={computed-paper:+.4f}")

def SECTION(t):
    print("\n" + "=" * 92); print(t); print("=" * 92)

# ------------------------------------------------------------------ 指标
def dece(c, l, nb=NB):
    n = len(c); e = 0.0; ed = np.linspace(0, 1, nb + 1)
    for i in range(nb):
        lo, hi = ed[i], ed[i + 1]
        m = (c >= lo) & (c <= hi) if i == nb - 1 else (c >= lo) & (c < hi)
        if m.sum(): e += (m.sum() / n) * abs(c[m].mean() - l[m].mean())
    return float(e)

def aece(c, l, nb=NB):
    n = len(c); o = np.argsort(c); cs, ls = c[o], l[o]; e = 0.0
    for idx in np.array_split(np.arange(n), nb):
        if len(idx): e += (len(idx) / n) * abs(cs[idx].mean() - ls[idx].mean())
    return float(e)

def brier(c, l): return float(np.mean((c - l) ** 2))
def nll(c, l):
    p = np.clip(c, EPS, 1 - EPS)
    return float(-np.mean(l * np.log(p) + (1 - l) * np.log(1 - p)))
def auc(c, l):
    if l.sum() in (0, len(l)): return float("nan")
    return float(stats.mannwhitneyu(c[l == 1], c[l == 0], alternative="greater").statistic
                 / (l.sum() * (len(l) - l.sum())))

METRICS = dict(dece=dece, aece=aece, brier=brier, nll=nll)

# ------------------------------------------------------------------ 读取
def raw(cfg, seed):
    p = REC / f"kvasir_{cfg}_seed{seed}_records.npz"
    if not p.exists(): return None
    d = np.load(p)
    return d["confidence"].astype(float), d["is_tp"].astype(int), d["image_idx"].astype(int)

def top1(cfg, seed):
    r = raw(cfg, seed)
    if r is None: return None
    c, l, idx = r; best = {}
    for ci, li, ii in zip(c, l, idx):
        if ii not in best or ci > best[ii][0]: best[ii] = (ci, li)
    return best

def own(cfg, seed):
    b = top1(cfg, seed)
    if b is None: return None
    k = sorted(b)
    return np.array([b[i][0] for i in k]), np.array([b[i][1] for i in k])

def inter(ca, cb, seed):
    Aa, Bb = top1(ca, seed), top1(cb, seed)
    if Aa is None or Bb is None: return None
    k = sorted(set(Aa) & set(Bb))
    return (np.array([Aa[i][0] for i in k]), np.array([Aa[i][1] for i in k]),
            np.array([Bb[i][0] for i in k]), np.array([Bb[i][1] for i in k]))

def perm(cfg, seed, thr=0.001):
    r = raw(cfg, seed)
    if r is None: return None
    c, l, _ = r; m = c >= thr
    return c[m], l[m]

def diffs(ca, cb, seeds, fn):
    out = []
    for s in seeds:
        g = inter(ca, cb, s)
        if g is None: return None
        ca_, la_, cb_, lb_ = g
        out.append(fn(cb_, lb_) - fn(ca_, la_))
    return np.array(out)

def tt(d):
    t, p = stats.ttest_1samp(d, 0.0); sd = np.std(d, ddof=1)
    return float(np.mean(d)), float(t), float(p), float(np.mean(d)/sd) if sd > 0 else np.nan

def tost(d, margin):
    """两个单侧检验，返回 (较大的 p, 90% CI)"""
    n = len(d); m = np.mean(d); se = np.std(d, ddof=1)/np.sqrt(n)
    p_lo = stats.t.sf((m - (-margin))/se, n-1)     # H0: d <= -margin
    p_hi = stats.t.cdf((m - margin)/se, n-1)       # H0: d >= +margin
    tcrit = stats.t.ppf(0.95, n-1)
    return float(max(p_lo, p_hi)), (float(m - tcrit*se), float(m + tcrit*se))

def ols(y, x):
    """y = a + b x，返回 (a, SE_a, p_a, CI_a, b, p_b)"""
    n = len(y); X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta; dof = n - 2
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tv = beta / se; pv = 2 * stats.t.sf(np.abs(tv), dof)
    tc = stats.t.ppf(0.975, dof)
    return (beta[0], se[0], pv[0], (beta[0]-tc*se[0], beta[0]+tc*se[0]), beta[1], pv[1])

# ================================================================== 开始
if not REC.exists(): sys.exit(f"缓存不存在: {REC}")
acc = json.load(open(METRICS_DIR/"accuracy_metrics.json"))
sta = json.load(open(METRICS_DIR/"single_task_accuracy.json"))
def A_(cfg, seed, key):
    if cfg == ST: return sta[str(seed)][key]
    return acc[f"{cfg}_seed{seed}"][key]

# ------------------------------------------------------------------ §5.1
SECTION("§5.1  深度趋势 / 方差异质性")
depths = [(A, 0), (CS, 5), (CD, 9), (B, 11)]
for mname, fn, sl, st_, sp in [("D-ECE", dece, 0.0042, 4.96, 0.004),
                               ("Adaptive ECE", aece, 0.0044, 3.79, 0.013),
                               ("Brier", brier, 0.0043, 5.97, 0.002),
                               ("NLL", nll, 0.0245, 6.02, 0.002)]:
    slopes = []
    for s in S6:
        ys = [fn(*own(c, s)) for c, _ in depths]
        ks = [k for _, k in depths]
        slopes.append(np.polyfit(ks, ys, 1)[0])
    slopes = np.array(slopes); t, p = stats.ttest_1samp(slopes, 0)
    CHECK(f"趋势斜率 {mname}", float(slopes.mean()), sl, 0.0001)
    CHECK(f"趋势 t   {mname}", float(t), st_, 0.006)
    CHECK(f"趋势 p   {mname}", float(p), sp, 0.001)

vals = [[dece(*own(c, s)) for s in S6] for c, _ in depths]
W, pW = stats.levene(*vals, center="mean")
chi, pB = stats.bartlett(*vals)
CHECK("Levene W", float(W), 5.70, 0.02)
CHECK("Levene p", float(pW), 0.005, 0.001)
CHECK("Bartlett p", float(pB), 0.027, 0.001)
sds = [float(np.std(v, ddof=1)) for v in vals]
CHECK("六种子 SD @k=0 (最小)", sds[0], 0.010, 0.001)
CHECK("六种子 SD @k=5 (最大)", sds[1], 0.042, 0.001)
print(f"        四个深度的 SD(ddof=1): " + " ".join(f"{x:.4f}" for x in sds))

g = inter(A, B, 42)
print(f"        A/B 交集图像数（各种子）: " +
      " ".join(str(len(inter(A, B, s)[0])) for s in S8))
CHECK("A/B 交集图像数 均值", float(np.mean([len(inter(A,B,s)[0]) for s in S8])), 177, 1.0)

# ------------------------------------------------------------------ §5.2
SECTION("§5.2  GAP+FC 残余 与 TOST")
for mname, fn, pm, pp in [("D-ECE", dece, 0.0065, 0.243), ("Adaptive ECE", aece, 0.0022, 0.731),
                          ("Brier", brier, 0.0093, 0.156), ("NLL", nll, 0.0754, 0.011)]:
    d = diffs(GA, GB, S8, fn); m, t, p, dz = tt(d)
    CHECK(f"残余均值 {mname}", m, pm, 0.0002)
    CHECK(f"残余 p    {mname}", p, pp, 0.001)

for mname, fn, ci_lo, ci_hi, margins in [
        ("D-ECE", dece, -0.0032, 0.0162, [(0.02, 0.017)]),
        ("Adaptive ECE", aece, -0.0093, 0.0136, [(0.02, 0.011)]),
        ("Brier", brier, -0.0018, 0.0204, [(0.03, 0.005), (0.02, 0.055)])]:
    d = diffs(GA, GB, S8, fn)
    for mg, pp in margins:
        p, ci = tost(d, mg)
        CHECK(f"TOST p {mname} ±{mg}", p, pp, 0.002)
    p, ci = tost(d, margins[0][0])
    CHECK(f"90%CI 下限 {mname}", ci[0], ci_lo, 0.0004)
    CHECK(f"90%CI 上限 {mname}", ci[1], ci_hi, 0.0004)

# ------------------------------------------------------------------ §5.4
SECTION("§5.4  置信度/精确率分解")
for lab, ca, cb, pc_a, pc_b, pdc, pdp, ppc, ppp in [
        ("LAG", A, B, 0.784, 0.676, -0.107, -0.011, 0.0015, 0.372),
        ("GAP", GA, GB, None, None, -0.033, -0.004, None, 0.553)]:
    dc, dp, ca_m, cb_m = [], [], [], []
    for s in S8:
        a_c, a_l, b_c, b_l = inter(ca, cb, s)
        dc.append(b_c.mean() - a_c.mean()); dp.append(b_l.mean() - a_l.mean())
        ca_m.append(a_c.mean()); cb_m.append(b_c.mean())
    mc, tc, pc, _ = tt(np.array(dc)); mp, tp2, pp2, _ = tt(np.array(dp))
    if pc_a is not None:
        CHECK(f"{lab} 置信度 A", float(np.mean(ca_m)), pc_a, 0.0006)
        CHECK(f"{lab} 置信度 B", float(np.mean(cb_m)), pc_b, 0.0006)
    CHECK(f"{lab} Δ置信度", mc, pdc, 0.0006)
    if ppc is not None: CHECK(f"{lab} Δ置信度 p", pc, ppc, 0.0005)
    CHECK(f"{lab} Δ精确率", mp, pdp, 0.0006)
    CHECK(f"{lab} Δ精确率 p", pp2, ppp, 0.001)

SECTION("§5.4  检测精度 / 分类精度")
def accdiff(ca, cb, key, seeds=S8):
    return np.array([A_(cb, s, key) - A_(ca, s, key) for s in seeds])
for lab, ca, cb, key, pm, pt, pp in [
        ("LAG mAP@50",  A, B,  "det_mAP50",   -0.096, -4.06, 0.006),
        ("GAP mAP@50",  GA, GB, "det_mAP50",  -0.025, -4.01, 0.006)]:
    m, t, p, dz = tt(accdiff(ca, cb, key))
    CHECK(f"{lab} 差", m, pm, 0.0006); CHECK(f"{lab} t", t, pt, 0.02); CHECK(f"{lab} p", p, pp, 0.001)
d = accdiff(A, B, "det_mAP50") - accdiff(GA, GB, "det_mAP50")
m, t, p, dz = tt(d)
CHECK("mAP@50 交互", m, -0.071, 0.0006); CHECK("mAP@50 交互 t", t, -2.79, 0.02)
CHECK("mAP@50 交互 p", p, 0.023, 0.001); CHECK("mAP@50 交互 dz", dz, -0.99, 0.02)
d2 = accdiff(A, B, "det_mAP50_95") - accdiff(GA, GB, "det_mAP50_95")
m2, _, p2, _ = tt(d2)
CHECK("mAP@50-95 交互", m2, -0.072, 0.0006); CHECK("mAP@50-95 交互 p", p2, 0.005, 0.001)
for lab, key, pa, pb, pp in [("validator precision", "det_precision", 0.799, 0.730, 0.018),
                             ("validator recall", "det_recall", 0.727, 0.670, 0.043)]:
    va = np.mean([A_(A, s, key) for s in S8]); vb = np.mean([A_(B, s, key) for s in S8])
    m, t, p, _ = tt(accdiff(A, B, key))
    CHECK(f"{lab} A", float(va), pa, 0.0006); CHECK(f"{lab} B", float(vb), pb, 0.0006)
    CHECK(f"{lab} p", p, pp, 0.001)
for lab, cfg, pv in [("cls top-1 A", A, 0.9030), ("cls top-1 B", B, 0.9022),
                     ("cls top-1 gapfc-A", GA, 0.9039), ("cls top-1 gapfc-B", GB, 0.9058),
                     ("cls top-1 F-frozen", FR, 0.8246)]:
    seeds = S6 if cfg == FR else S8
    CHECK(lab, float(np.mean([A_(cfg, s, "cls_acc") for s in seeds])), pv, 0.0006)
dcls = accdiff(A, B, "cls_acc") - accdiff(GA, GB, "cls_acc")
_, _, pcls, _ = tt(dcls)
CHECK("cls top-1 交互 p", pcls, 0.567, 0.002)

SECTION("§5.4 / S5  ΔmAP 与 ΔD-ECE 的相关（M6）")
ddece = diffs(A, B, S8, dece); dmap = accdiff(A, B, "det_mAP50")
r, pr = stats.pearsonr(dmap, ddece); rho, prho = stats.spearmanr(dmap, ddece)
CHECK("Pearson r", float(r), -0.354, 0.002); CHECK("Pearson p", float(pr), 0.390, 0.002)
CHECK("Spearman rho", float(rho), 0.071, 0.002); CHECK("Spearman p", float(prho), 0.867, 0.003)
a, sea, pa, cia, b_, pb = ols(ddece, dmap)
CHECK("协变量调整截距", a, 0.0348, 0.0004); CHECK("截距 SE", sea, 0.0238, 0.0004)
CHECK("截距 p", pa, 0.193, 0.002); CHECK("截距 CI 下限", cia[0], -0.023, 0.001)
CHECK("截距 CI 上限", cia[1], 0.093, 0.001); CHECK("ΔmAP 斜率 p", pb, 0.390, 0.002)
sub = [S8.index(s) for s in S6]
r6, p6 = stats.pearsonr(dmap[sub], ddece[sub]); rho6, prho6 = stats.spearmanr(dmap[sub], ddece[sub])
CHECK("种子42-47 Pearson r", float(r6), 0.711, 0.003); CHECK("种子42-47 Pearson p", float(p6), 0.113, 0.003)
CHECK("种子42-47 Spearman rho", float(rho6), 0.771, 0.003); CHECK("种子42-47 Spearman p", float(prho6), 0.072, 0.003)

# ------------------------------------------------------------------ §5.7
SECTION("§5.7  检测-only 基线")
for mname, fn, pm, pp in [("D-ECE", dece, None, 0.634), ("Adaptive ECE", aece, None, 0.413),
                          ("Brier", brier, None, 0.712), ("NLL", nll, None, 0.782)]:
    d = diffs(ST, A, S8, fn); m, t, p, dz = tt(d)
    CHECK(f"基线 vs A  p {mname}", p, pp, 0.001)
    if mname == "D-ECE": CHECK("基线 vs A  dz", dz, 0.18, 0.02)
vals_st, vals_a = [], []
for s in S8:
    st_c, st_l, a_c, a_l = inter(ST, A, s)
    vals_st.append(dece(st_c, st_l)); vals_a.append(dece(a_c, a_l))
CHECK("基线 D-ECE (匹配集)", float(np.mean(vals_st)), 0.1660, 0.0006)
CHECK("A   D-ECE (匹配集)", float(np.mean(vals_a)), 0.1708, 0.0006)
for mname, fn, pm, pp in [("D-ECE", dece, 0.0552, 0.017), ("Adaptive ECE", aece, 0.0556, 0.032),
                          ("Brier", brier, 0.0469, 0.034), ("NLL", nll, 0.2627, 0.019)]:
    d = diffs(ST, B, S8, fn); m, t, p, dz = tt(d)
    CHECK(f"基线 vs B  差 {mname}", m, pm, 0.0004)
    CHECK(f"基线 vs B  p  {mname}", p, pp, 0.001)
m, t, p, _ = tt(accdiff(ST, A, "det_mAP50"))
CHECK("基线 vs A mAP@50 p", p, 0.871, 0.002)
n = 8; se = np.std(accdiff(ST, A, "det_mAP50"), ddof=1)/np.sqrt(n)
tc = stats.t.ppf(0.975, n-1); mm = np.mean(accdiff(ST, A, "det_mAP50"))
CHECK("mAP 差 95%CI 下限", float(mm-tc*se), -0.028, 0.001)
CHECK("mAP 差 95%CI 上限", float(mm+tc*se), 0.033, 0.001)
for lab, key, pp in [("mAP@50-95", "det_mAP50_95", 0.982), ("recall", "det_recall", 0.799)]:
    _, _, p, _ = tt(accdiff(ST, A, key)); CHECK(f"基线 vs A {lab} p", p, pp, 0.002)
for lab, key, pm, pp in [("mAP@50", "det_mAP50", -0.0983, 0.026),
                         ("mAP@50-95", "det_mAP50_95", -0.1043, 0.009)]:
    m, _, p, _ = tt(accdiff(ST, B, key))
    CHECK(f"基线 vs B {lab} 差", m, pm, 0.0006); CHECK(f"基线 vs B {lab} p", p, pp, 0.001)

# ------------------------------------------------------------------ §5.9
SECTION("§5.9  宽松协议：预测数依赖")
ALL = [(ST,S8),(A,S8),(CS,S6),(CD,S6),(B,S8),(GA,S8),(GB,S8),(FR,S6)]
JOINT = [(A,S8),(CS,S6),(CD,S6),(B,S8),(GA,S8),(GB,S8),(FR,S6)]   # 50 个联合训练模型
ns, ds, labels = [], [], []
for cfg, seeds in JOINT:
    for s in seeds:
        c, l = perm(cfg, s); ns.append(len(c)); ds.append(dece(c, l)); labels.append(cfg)
ns, ds = np.array(ns), np.array(ds)
CHECK("联合训练模型数", float(len(ns)), 50, 0.5)
CHECK("预测数 最小", float(ns.min()), 191, 0.5)
CHECK("预测数 最大", float(ns.max()), 555, 0.5)
CHECK("预测数 均值", float(ns.mean()), 264, 1.0)
r50, p50 = stats.pearsonr(ns, ds); CHECK("r (50 模型)", float(r50), -0.79, 0.006)
m32 = np.isin(labels, [A, B, GA, GB])
r32, _ = stats.pearsonr(ns[m32], ds[m32]); CHECK("r (32 模型)", float(r32), -0.82, 0.006)
for cfg, pv in [(A, -0.80), (GA, -0.95)]:
    mm = np.array(labels) == cfg
    rr, _ = stats.pearsonr(ns[mm], ds[mm]); CHECK(f"r 组内 {cfg}", float(rr), pv, 0.006)
for cfg, pv in [(A,328),(CS,241),(CD,226),(B,224),(GA,303),(GB,244),(FR,271)]:
    seeds = dict(JOINT)[cfg]
    CHECK(f"平均预测数 {cfg}", float(np.mean([len(perm(cfg,s)[0]) for s in seeds])), pv, 0.6)
dN  = np.array([len(perm(B,s)[0]) - len(perm(A,s)[0]) for s in S8])
dD  = np.array([dece(*perm(B,s)) - dece(*perm(A,s)) for s in S8])
rp, pp_ = stats.pearsonr(dN, dD)
CHECK("配对 r(ΔN,ΔD-ECE)", float(rp), -0.84, 0.006); CHECK("配对 r p", float(pp_), 0.009, 0.001)
a, sea, pa, cia, b_, pb = ols(dD, dN)
CHECK("外推到等预测数的余量", a, 0.0153, 0.0006)
CHECK("ΔN 斜率 p", pb, 0.009, 0.002)
CHECK("宽松 A vs B D-ECE", float(dD.mean()), 0.0532, 0.0004)
for mname, fn, pp in [("D-ECE", dece, 0.30), ("Adaptive ECE", aece, 0.65), ("Brier", brier, 0.61)]:
    dm = np.array([fn(*perm(B,s)) - fn(*perm(A,s)) for s in S8])
    _, _, _, _, _, _ = ols(dm, dN)
    aa, _, ppa, _, _, _ = ols(dm, dN)
    CHECK(f"协变量调整后 配置效应 p {mname}", ppa, pp, 0.02)

SECTION("§5.9  最低箱分解（宽松协议）")
contribA = contribB = 0.0; cA=[];cB=[];prA=[];prB=[];tpA=[];fpA=[];tpB=[];fpB=[]
for s in S8:
    for cfg, acc_c, accm, accp, acctp, accfp in [(A,None,cA,prA,tpA,fpA),(B,None,cB,prB,tpB,fpB)]:
        c, l = perm(cfg, s); m = (c >= 0.0) & (c < 0.1)
        accm.append(c[m].mean()); accp.append(l[m].mean())
        acctp.append(int(l[m].sum())); accfp.append(int((1-l[m]).sum()))
def bincontrib(cfg, s):
    c, l = perm(cfg, s); m = (c >= 0.0) & (c < 0.1)
    return (m.sum()/len(c)) * abs(c[m].mean() - l[m].mean())
cont = np.mean([bincontrib(B,s) - bincontrib(A,s) for s in S8])
CHECK("最低箱贡献差", float(cont), 0.0405, 0.0008)
CHECK("最低箱占比 %", float(100*cont/dD.mean()), 76, 1.5)
CHECK("最低箱 平均置信度 A", float(np.mean(cA)), 0.016, 0.0008)
CHECK("最低箱 平均置信度 B", float(np.mean(cB)), 0.020, 0.0008)
CHECK("最低箱 精确率 A", float(np.mean(prA)), 0.195, 0.002)
CHECK("最低箱 精确率 B", float(np.mean(prB)), 0.354, 0.002)
CHECK("最低箱 TP A", float(np.mean(tpA)), 22.5, 0.3)
CHECK("最低箱 FP A", float(np.mean(fpA)), 120.0, 0.5)
CHECK("最低箱 TP B", float(np.mean(tpB)), 24.9, 0.3)
CHECK("最低箱 FP B", float(np.mean(fpB)), 49.0, 0.5)

SECTION("§5.9  阈值扫描 与 非限制形式")
for thr, na, nb, pd_, pa_, pb_, pn_ in [
        (0.001,328,224, 0.0532, 0.0455, 0.0485, 0.2243),
        (0.05, 198,160, 0.0001,-0.0020,-0.0096,-0.0238),
        (0.10, 186,150, 0.0032,-0.0023,-0.0078,-0.0179),
        (0.25, 169,137,-0.0045,-0.0031,-0.0064,-0.0151)]:
    CHECK(f"阈值{thr} N_A", float(np.mean([len(perm(A,s,thr)[0]) for s in S8])), na, 0.6)
    CHECK(f"阈值{thr} N_B", float(np.mean([len(perm(B,s,thr)[0]) for s in S8])), nb, 0.6)
    for mname, fn, pv in [("D-ECE",dece,pd_),("AdaECE",aece,pa_),("Brier",brier,pb_),("NLL",nll,pn_)]:
        dm = np.array([fn(*perm(B,s,thr)) - fn(*perm(A,s,thr)) for s in S8])
        CHECK(f"阈值{thr} {mname}", float(dm.mean()), pv, 0.0004)
dm = np.array([dece(*perm(B,s,0.05)) - dece(*perm(A,s,0.05)) for s in S8])
se = np.std(dm, ddof=1)/np.sqrt(8); tc = stats.t.ppf(0.975,7)
CHECK("阈值0.05 95%CI 下限", float(dm.mean()-tc*se), -0.017, 0.0015)
CHECK("阈值0.05 95%CI 上限", float(dm.mean()+tc*se), 0.018, 0.0015)
_se = np.std(dm, ddof=1)/np.sqrt(8); _tc = stats.t.ppf(0.975, 7)
_power = lambda d: (stats.nct.sf(_tc, 7, d/_se) + stats.nct.cdf(-_tc, 7, d/_se))
mde = optimize.brentq(lambda d: _power(d) - 0.80, 1e-4, 0.10)
CHECK("80%功效可检出差 (non-central t)", float(mde), 0.024, 0.001)
# 非限制的每图 top-1（不取交集）
nA = np.mean([len(own(A,s)[0]) for s in S8]); nB = np.mean([len(own(B,s)[0]) for s in S8])
CHECK("非限制 每图top1 图像数 A", float(nA), 196, 0.6)
CHECK("非限制 每图top1 图像数 B", float(nB), 179, 0.6)
du = np.array([dece(*own(B,s)) - dece(*own(A,s)) for s in S8])
mu, tu, pu, _ = tt(du)
CHECK("非限制 A vs B D-ECE", mu, 0.0435, 0.0004); CHECK("非限制 p", pu, 0.003, 0.001)

# ------------------------------------------------------------------ §6.5
SECTION("§6.5  异常种子 48/49 与迁移检查")
CHECK("B seed48 mAP@50", float(A_(B,48,"det_mAP50")), 0.773, 0.0008)
CHECK("B seed49 mAP@50", float(A_(B,49,"det_mAP50")), 0.751, 0.0008)
r42_47 = [A_(B,s,"det_mAP50") for s in S6]
CHECK("B seeds42-47 mAP 最小", float(min(r42_47)), 0.589, 0.0008)
CHECK("B seeds42-47 mAP 最大", float(max(r42_47)), 0.634, 0.0008)
cov = {s: len(own(B,s)[0]) for s in S8}
CHECK("B seed48 覆盖图像数", float(cov[48]), 198, 0.5)
CHECK("B seed49 覆盖图像数", float(cov[49]), 195, 0.5)
CHECK("B seeds42-47 覆盖 最小", float(min(cov[s] for s in S6)), 164, 0.5)
CHECK("B seeds42-47 覆盖 最大", float(max(cov[s] for s in S6)), 182, 0.5)
a_r = [A_(A,s,"det_mAP50") for s in S6]
CHECK("A seeds42-47 mAP 最小", float(min(a_r)), 0.7133, 0.0008)
CHECK("A seeds42-47 mAP 最大", float(max(a_r)), 0.7550, 0.0008)

# ------------------------------------------------------------------ 汇总
SECTION("汇总")
print(f"  PASS {_n_pass}   FAIL {_n_fail}")
print("\n未覆盖（需要图像文件或外部数据集缓存，本脚本不查）：")
print("  §3.1 病灶面积中位数 18.47/9.81/2.85%、下四分位 1.29%")
print("  §3.1 近重复审计计数、§4.3 重叠审计计数（需感知哈希重跑）")
print("  §5.5 外部数据集全部数值（需 raw_records_external）")
print("  §5.6 分层各层预测数 10--48")
print("  S1 自助区间（需重采样，量大）")
