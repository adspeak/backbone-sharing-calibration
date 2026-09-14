# -*- coding: utf-8 -*-
"""
全文数字回归核对 · 第二轮 (v30)
===============================
第一轮修掉的两个脚本 bug 已改正:
  * mAP 95%CI 方向（论文用 基线 - A）
  * Levene 用 Brown--Forsythe（center='median'）

新增覆盖:
  §4.1 自助区间两种重采样的比较
  §5.4 宽松协议 AUC
  §5.5 外部数据集全部数值
  §5.6 分层分析
  §3.1 病灶面积分位数
  §3.1/§4.3 近重复审计与重叠审计
  S1 自助区间抽查

零推理。无卡模式即可。
用法:
    cd verification
    python3 verify_round2_v30.py > verify_round2_out.txt 2>&1
    grep -E ' FAIL | SKIP ' verify_round2_out.txt
"""
import json, sys, os, glob
import numpy as np
from pathlib import Path
from scipy import stats

# Paths resolve through src/paths.py so the script runs from a fresh clone with
# no configuration: the cached records bundled under data/raw_records/ are used
# when present, otherwise the workspace named by BACKBONE_CALIB_ROOT.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import (CALIB, METRICS_DIR, RECORDS_DIR,  # noqa: E402
                   EXTERNAL_RECORDS_DIR)

from paths import ROOT  # noqa: E402

REC = RECORDS_DIR
EXT = EXTERNAL_RECORDS_DIR
S8 = [42,43,44,45,46,47,48,49]; S6=[42,43,44,45,46,47]; S3=[42,43,44]
NB, EPS = 10, 1e-12
A,B  = "A_independent","B_shared"
GA,GB= "gapfc_A_independent","gapfc_B_shared"
CS,CD,FR,ST = "C_shallow","C_deep","F_frozen","single_task"

_p=_f=_s=0
def CHECK(label, computed, paper, tol):
    global _p,_f
    if computed is None or (isinstance(computed,float) and np.isnan(computed)):
        SKIP(label,"无法计算"); return
    ok = abs(computed-paper) <= tol
    _p += ok; _f += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:54s} 论文={paper:<10.4f} 重算={computed:<10.4f} 差={computed-paper:+.4f}")
def SKIP(label, why):
    global _s; _s += 1
    print(f"  SKIP  {label:54s} {why}")
def SECTION(t): print("\n"+"="*95); print(t); print("="*95)

def dece(c,l,nb=NB):
    n=len(c); e=0.; ed=np.linspace(0,1,nb+1)
    for i in range(nb):
        lo,hi=ed[i],ed[i+1]
        m=(c>=lo)&(c<=hi) if i==nb-1 else (c>=lo)&(c<hi)
        if m.sum(): e+=(m.sum()/n)*abs(c[m].mean()-l[m].mean())
    return float(e)
def aece(c,l,nb=NB):
    n=len(c); o=np.argsort(c); cs,ls=c[o],l[o]; e=0.
    for idx in np.array_split(np.arange(n),nb):
        if len(idx): e+=(len(idx)/n)*abs(cs[idx].mean()-ls[idx].mean())
    return float(e)
def brier(c,l): return float(np.mean((c-l)**2))
def nll(c,l):
    p=np.clip(c,EPS,1-EPS); return float(-np.mean(l*np.log(p)+(1-l)*np.log(1-p)))
def auc(c,l):
    if l.sum() in (0,len(l)): return float("nan")
    return float(stats.mannwhitneyu(c[l==1],c[l==0],alternative="greater").statistic/(l.sum()*(len(l)-l.sum())))

def raw(cfg,seed,d=REC,ds="kvasir"):
    for _pat in (f"{ds}_{cfg}_seed{seed}_records.npz",
                 f"{ds}_{cfg}_{ds}_seed{seed}.npz",
                 f"{ds}_{cfg}_seed{seed}.npz"):
        p=d/_pat
        if p.exists(): break
    else: return None
    z=np.load(p)
    idx = z["image_idx"].astype(int) if "image_idx" in z.files else None
    return z["confidence"].astype(float), z["is_tp"].astype(int), idx
def top1(cfg,seed,d=REC,ds="kvasir"):
    r=raw(cfg,seed,d,ds)
    if r is None or r[2] is None: return None
    c,l,i=r; b={}
    for a_,b_,c_ in zip(c,l,i):
        if c_ not in b or a_>b[c_][0]: b[c_]=(a_,b_)
    return b
def own(cfg,seed,d=REC,ds="kvasir"):
    b=top1(cfg,seed,d,ds)
    if b is None: return None
    k=sorted(b); return np.array([b[j][0] for j in k]), np.array([b[j][1] for j in k])
def inter(ca,cb,seed,d=REC,ds="kvasir"):
    Aa,Bb=top1(ca,seed,d,ds),top1(cb,seed,d,ds)
    if Aa is None or Bb is None: return None
    k=sorted(set(Aa)&set(Bb))
    return (np.array([Aa[i][0] for i in k]),np.array([Aa[i][1] for i in k]),
            np.array([Bb[i][0] for i in k]),np.array([Bb[i][1] for i in k]))
def perm(cfg,seed,thr=0.001,d=REC,ds="kvasir"):
    r=raw(cfg,seed,d,ds)
    if r is None: return None
    c,l,_=r; m=c>=thr; return c[m],l[m]
def diffs(ca,cb,seeds,fn,d=REC,ds="kvasir"):
    out=[]
    for s in seeds:
        g=inter(ca,cb,s,d,ds)
        if g is None: return None
        a_,al,b_,bl=g; out.append(fn(b_,bl)-fn(a_,al))
    return np.array(out)
def tt(d):
    t,p=stats.ttest_1samp(d,0.); sd=np.std(d,ddof=1)
    return float(np.mean(d)),float(t),float(p),float(np.mean(d)/sd) if sd>0 else np.nan

acc=json.load(open(METRICS_DIR/"accuracy_metrics.json"))
sta=json.load(open(METRICS_DIR/"single_task_accuracy.json"))
def A_(cfg,seed,key):
    return sta[str(seed)][key] if cfg==ST else acc[f"{cfg}_seed{seed}"][key]

# ============================================================ P. 探查
SECTION("[P] 环境探查（供后续补检验用，不计入 PASS/FAIL）")
for p in [EXT, ROOT/"GI_P2_data", ROOT/"CVC_raw", ROOT/"datasets"]:
    print(f"  {str(p):50s} {'存在' if p.exists() else '不存在'}")
if EXT.exists():
    fs=sorted(os.path.basename(x) for x in glob.glob(str(EXT/"*.npz")))
    print(f"  raw_records_external: {len(fs)} 个文件")
    for f in fs[:12]: print("     ", f)
    if fs:
        z=np.load(EXT/fs[0]); print("      字段:", {k:z[k].shape for k in z.files})
for j in ["image_level_bootstrap.json","exp1_size_stratified_results.json",
          "etis_exp5_etis_size_stratified_results.json","exp5_calibration_results.json",
          "supplementary_ci_results.json"]:
    p=METRICS_DIR/j
    if not p.exists(): print(f"  {j:48s} 不存在"); continue
    d=json.load(open(p))
    ks=list(d)[:6] if isinstance(d,dict) else f"list len={len(d)}"
    print(f"  {j:48s} 顶层键: {ks}")
    if isinstance(d,dict) and d:
        v=d[list(d)[0]]
        print(f"      首元素类型 {type(v).__name__}: {list(v)[:10] if isinstance(v,dict) else str(v)[:90]}")
try:
    import imagehash; from PIL import Image
    print("  imagehash / PIL: 可用")
    HAS_HASH=True
except Exception as e:
    print(f"  imagehash / PIL: 不可用 ({e})"); HAS_HASH=False

# ============================================================ 1. 复验 v30 的改动
SECTION("[1] v30 本轮改动的复验")
def accdiff(ca,cb,key,seeds=S8): return np.array([A_(cb,s,key)-A_(ca,s,key) for s in seeds])
for lab,ca,cb,key,pm,pt,pp in [("LAG mAP@50",A,B,"det_mAP50",-0.096,-3.92,0.006),
                               ("GAP mAP@50",GA,GB,"det_mAP50",-0.025,-3.86,0.006)]:
    m,t,p,dz=tt(accdiff(ca,cb,key))
    CHECK(f"{lab} 差",m,pm,0.0006); CHECK(f"{lab} t",t,pt,0.006); CHECK(f"{lab} p",p,pp,0.001)
d=accdiff(A,B,"det_mAP50")-accdiff(GA,GB,"det_mAP50"); m,t,p,dz=tt(d)
CHECK("mAP@50 交互 t",t,-2.91,0.006); CHECK("mAP@50 交互 dz",dz,-1.03,0.006)
dcls=accdiff(A,B,"cls_acc")-accdiff(GA,GB,"cls_acc")
CHECK("cls top-1 交互 p",tt(dcls)[2],0.565,0.001)
for mname,fn,pv in [("Adaptive ECE",aece,0.0025),("Brier",brier,-0.0108),("NLL",nll,-0.0254)]:
    dm=np.array([fn(*perm(B,s,0.25))-fn(*perm(A,s,0.25)) for s in S8])
    CHECK(f"阈值0.25 {mname}",float(dm.mean()),pv,0.0004)
JOINT=[(A,S8),(CS,S6),(CD,S6),(B,S8),(GA,S8),(GB,S8),(FR,S6)]
ns,ds=[],[]
for cfg,seeds in JOINT:
    for s in seeds:
        c,l=perm(cfg,s); ns.append(len(c)); ds.append(dece(c,l))
CHECK("联合训练模型数",float(len(ns)),50,0.5)
CHECK("预测总数 (50 模型)",float(np.sum(ns)),13213,0.5)
CHECK("r (50 模型)",float(stats.pearsonr(np.array(ns),np.array(ds))[0]),-0.74,0.006)
D=[[dece(*own(c,s)) for s in S6] for c in [A,CS,CD,B]]
W,pW=stats.levene(*D,center="median")
CHECK("Levene W (Brown--Forsythe)",float(W),5.70,0.01)
CHECK("Levene p (Brown--Forsythe)",float(pW),0.005,0.001)
dm=accdiff(A,ST,"det_mAP50")   # 论文方向：基线 - A
se=np.std(dm,ddof=1)/np.sqrt(8); tc=stats.t.ppf(0.975,7)
CHECK("mAP 差 95%CI 下限",float(dm.mean()-tc*se),-0.028,0.001)
CHECK("mAP 差 95%CI 上限",float(dm.mean()+tc*se),0.033,0.001)

# ============================================================ 2. TS 派生量
SECTION("[2] §5.4 温度缩放派生量（与 ts_top1_v28_results.json 对齐）")
p=METRICS_DIR/"ts_top1_v28_results.json"
if not p.exists():
    SKIP("TS 结果文件","未找到 ts_top1_v28_results.json，请先跑 ts_top1_v28.py")
else:
    ts=json.load(open(p))
    def tm(cfg,key,seeds): return float(np.mean([ts[cfg][str(s)][key] for s in seeds]))
    for cfg,seeds,pT,pad,pab in [(ST,S8,2.551,0.0991,0.1613),(A,S8,2.614,0.1173,0.1660),
                                 (CS,S6,2.697,0.1292,0.1585),(CD,S6,3.487,0.1615,0.1875),
                                 (B,S8,3.698,0.1765,0.1888),(GA,S8,2.832,0.1297,0.1710),
                                 (GB,S8,3.088,0.1484,0.1755),(FR,S6,3.755,0.1812,0.1983)]:
        CHECK(f"tab:ts T   {cfg}",tm(cfg,"T",seeds),pT,0.002)
        CHECK(f"tab:ts D-ECE后 {cfg}",tm(cfg,"ad",seeds),pad,0.0002)
        CHECK(f"tab:ts Brier后 {cfg}",tm(cfg,"ab",seeds),pab,0.0002)
    fix=[100*np.mean([(ts[c][str(s)]["bd"]-ts[c][str(s)]["ad"])/ts[c][str(s)]["bd"] for s in ss])
         for c,ss in [(ST,S8),(A,S8),(CS,S6),(CD,S6),(B,S8),(GA,S8),(GB,S8),(FR,S6)]]
    CHECK("TS 改善下界 %",float(min(fix)),18,0.5); CHECK("TS 改善上界 %",float(max(fix)),40,0.5)
    tA,tB=tm(A,"T",S8),tm(B,"T",S8); gA,gB=tm(GA,"T",S8),tm(GB,"T",S8)
    CHECK("LAG 温度增幅 %",100*(tB/tA-1),42,0.6); CHECK("GAP 温度增幅 %",100*(gB/gA-1),9,0.6)

SECTION("[3] §5.4 宽松协议 AUC（论文用于说明修订了早先的读法）")
aA=np.mean([auc(*perm(A,s)) for s in S8]); aB=np.mean([auc(*perm(B,s)) for s in S8])
dA=np.array([auc(*perm(B,s))-auc(*perm(A,s)) for s in S8])
CHECK("宽松 AUC A",float(aA),0.8663,0.0006); CHECK("宽松 AUC B",float(aB),0.8062,0.0006)
CHECK("宽松 AUC p",tt(dA)[2],0.013,0.001)

# ============================================================ 4. §4.1 自助区间
SECTION("[4] §4.1 两种重采样方案的区间比较")
p=METRICS_DIR/"image_level_bootstrap.json"
if not p.exists(): SKIP("image_level_bootstrap.json","不存在")
else:
    d=json.load(open(p))
    print("      顶层键:", list(d)[:8])
    widths=[]
    def walk(o):
        if isinstance(o,dict):
            lo=hi=lo2=hi2=None
            for k,v in o.items():
                if isinstance(v,(list,tuple)) and len(v)==2 and all(isinstance(x,(int,float)) for x in v):
                    widths.append((k, v[1]-v[0]))
                walk(v)
        elif isinstance(o,list):
            for v in o[:200]: walk(v)
    walk(d)
    if widths:
        byk={}
        for k,w in widths: byk.setdefault(k,[]).append(w)
        for k,v in list(byk.items())[:10]:
            print(f"      区间宽度 {k:32s} n={len(v):4d} 均值={np.mean(v):.4f}")
        print("      >>> 论文称两方案端点差 < 0.006；请人工对照上面两组宽度")
    else:
        print("      未识别出区间字段，请把顶层键结构贴回")

# ============================================================ 5. §5.5 外部数据集
SECTION("[5] §5.5 外部数据集")
if not EXT.exists():
    SKIP("外部数据集全部检验","raw_records_external 不存在")
else:
    fs=[os.path.basename(x) for x in glob.glob(str(EXT/"*.npz"))]
    for ds,pN,rows in [("cvc",115,[("D-ECE",dece,0.0700,0.1021,0.0320,1.80,0.214),
                                   ("Brier",brier,0.0312,0.0722,0.0410,4.19,0.053),
                                   ("NLL",nll,0.1152,0.3030,0.1878,3.29,0.081)]),
                       ("etis",32,[("D-ECE",dece,0.2157,0.2164,0.0007,0.02,0.989)])]:
        cand=[f for f in fs if ds in f.lower()]
        if not cand:
            SKIP(f"{ds} 全部检验", f"未找到含 '{ds}' 的 npz（实际文件: {fs[:4]}）"); continue
        # 实际命名: <ds>_<cfg>_<ds>_seed<NN>.npz
        def g(cfg,s):
            for pat in [f"{ds}_{cfg}_{ds}_seed{s}.npz",
                        f"{ds}_{cfg}_seed{s}_records.npz", f"{ds}_{cfg}_seed{s}.npz"]:
                if (EXT/pat).exists(): return own(cfg,s,EXT,ds)
            return None
        def gp(s):
            """Matched-image protocol: restrict to images on which both
            configurations produce a prediction, as the manuscript does."""
            ta,tb=top1(A,s,EXT,ds),top1(B,s,EXT,ds)
            if ta is None or tb is None: return None
            k=sorted(set(ta)&set(tb))
            return (np.array([ta[j][0] for j in k]),np.array([ta[j][1] for j in k]),
                    np.array([tb[j][0] for j in k]),np.array([tb[j][1] for j in k]))
        if g(A,42) is None:
            SKIP(f"{ds} 全部检验", f"命名不匹配，实际: {cand[:4]}"); continue
        pr=[gp(s) for s in S3]
        CHECK(f"{ds} N̄", float(np.mean([len(x[0]) for x in pr])), pN, 1.0)
        for mname,fn,pa,pb,pd_,pt,pp in rows:
            va=np.mean([fn(x[0],x[1]) for x in pr]); vb=np.mean([fn(x[2],x[3]) for x in pr])
            dd=np.array([fn(x[2],x[3])-fn(x[0],x[1]) for x in pr]); m,t,p,_=tt(dd)
            CHECK(f"{ds} {mname} A",float(va),pa,0.0006)
            CHECK(f"{ds} {mname} B",float(vb),pb,0.0006)
            CHECK(f"{ds} {mname} 差",m,pd_,0.0006)
            CHECK(f"{ds} {mname} t",t,pt,0.02); CHECK(f"{ds} {mname} p",p,pp,0.002)
        if ds=="etis":
            npm=[len(perm(A,s,0.001,EXT,ds)[0]) for s in S3]+[len(perm(B,s,0.001,EXT,ds)[0]) for s in S3]
            CHECK("ETIS 宽松 N 最小",float(min(npm)),34,0.5)
            CHECK("ETIS 宽松 N 最大",float(max(npm)),82,0.5)
            dd=np.array([dece(*perm(B,s,0.001,EXT,ds))-dece(*perm(A,s,0.001,EXT,ds)) for s in S3])
            m,t,p,dz=tt(dd)
            CHECK("ETIS 宽松 p",p,0.006,0.002); CHECK("ETIS 宽松 dz",dz,7.49,0.05)

# ============================================================ 6. §5.6 分层
SECTION("[6] §5.6 ETIS 分层分析")
p=METRICS_DIR/"etis_exp5_etis_size_stratified_results.json"
if not p.exists(): SKIP("分层分析","json 不存在")
else:
    d=json.load(open(p)); print("      顶层键:",list(d)[:8])
    print("      >>> tab:size 论文值 Small 0.2113/0.0965  Medium 0.1669/0.2131  Large 0.1214/0.2256")
    print("      >>> 各层预测数论文称 10--48；请对照下面输出")
    def show(o,pre="",lv=0):
        if lv>3: return
        if isinstance(o,dict):
            for k,v in list(o.items())[:8]:
                if isinstance(v,(int,float)): print(f"      {pre}{k} = {v}")
                else: print(f"      {pre}{k} ({type(v).__name__})"); show(v,pre+"  ",lv+1)
        elif isinstance(o,list) and o:
            print(f"      {pre}[len={len(o)}]"); show(o[0],pre+"  ",lv+1)
    show(d)

# ============================================================ 7. §3.1 病灶面积
SECTION("[7] §3.1 病灶相对面积分位数  论文: Kvasir 18.47%  CVC 9.81%  ETIS 2.85%  ETIS Q1 1.29%")
def areas(label_dir):
    out=[]
    for f in glob.glob(os.path.join(label_dir,"*.txt")):
        for line in open(f):
            pr=line.split()
            if len(pr)>=5: out.append(float(pr[3])*float(pr[4]))
    return np.array(out)
found=False
for name,pat,pv,q1 in [("Kvasir-SEG","**/labels/train",18.47,None),
                       ("CVC-ClinicDB","**/cvc*/**/labels/train",9.81,None),
                       ("ETIS","**/etis*/**/labels/train",2.85,1.29)]:
    hits=[d for d in glob.glob(str(ROOT/pat),recursive=True) if os.path.isdir(d)]
    if not hits: SKIP(f"{name} 相对面积中位数", f"未找到标签目录 (模式 {pat})"); continue
    a=areas(hits[0])
    if len(a)==0: SKIP(f"{name} 相对面积中位数","标签目录为空"); continue
    found=True
    print(f"      使用目录 {hits[0]}  框数={len(a)}")
    CHECK(f"{name} 相对面积中位数 %",float(100*np.median(a)),pv,0.05)
    if q1: CHECK(f"{name} 相对面积 Q1 %",float(100*np.percentile(a,25)),q1,0.05)
if not found:
    print("      >>> 未定位到 YOLO 标签目录。请跑: find $BACKBONE_CALIB_ROOT -type d -name labels | head")

# ============================================================ 8. 哈希审计
SECTION("[8] §3.1 近重复审计 / §4.3 重叠审计")
print("      论文: CVC 6/122(5%) 28(23%) 101(83%)；ETIS 3(8%) 12(31%) 24(62%)")
print("      论文: cls-train∩det-val=2  cls-train∩det-train=10  cls-val∩det-train=2  cls-val∩det-val=0")
if not HAS_HASH:
    SKIP("哈希审计","imagehash/PIL 不可用；pip install imagehash pillow 后可补")
else:
    print("      >>> 需要图像目录路径才能跑。请跑:")
    print("          find $BACKBONE_CALIB_ROOT -type d -name images | head -20")
    SKIP("哈希审计","待提供图像目录路径")

# ============================================================ 9. S1 自助区间抽查
SECTION("[9] S1 自助区间抽查（single_task seed42，B=1000，图像级重采样）")
g=own(ST,42)
if g is None: SKIP("S1 抽查","缓存缺失")
else:
    c,l=g; n=len(c); rng=np.random.default_rng(0)
    boot={k:[] for k in ("dece","aece","brier","nll")}
    for _ in range(1000):
        idx=rng.integers(0,n,n)
        boot["dece"].append(dece(c[idx],l[idx])); boot["aece"].append(aece(c[idx],l[idx]))
        boot["brier"].append(brier(c[idx],l[idx])); boot["nll"].append(nll(c[idx],l[idx]))
    for k,pt_,lo,hi in [("dece",0.1269,0.088,0.191),("aece",0.0997,0.079,0.174),
                        ("brier",0.1578,0.117,0.201),("nll",0.5383,0.394,0.690)]:
        v=np.array(boot[k])
        CHECK(f"S1 {k} 点估计",{'dece':dece,'aece':aece,'brier':brier,'nll':nll}[k](c,l),pt_,0.0002)
        CHECK(f"S1 {k} CI 下限",float(np.percentile(v,2.5)),lo,0.012)
        CHECK(f"S1 {k} CI 上限",float(np.percentile(v,97.5)),hi,0.012)
    print("      注: 自助区间依赖随机种子，±0.012 容差内即认为一致")

SECTION("汇总")
print(f"  PASS {_p}   FAIL {_f}   SKIP {_s}")
