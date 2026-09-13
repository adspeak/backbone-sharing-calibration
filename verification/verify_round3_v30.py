# -*- coding: utf-8 -*-
"""
全文数字回归核对 · 第三轮补丁 (v30)
==================================
补齐第二轮 SKIP / 存疑的五块:
  [A] §5.5 外部数据集   —— 命名修正为 <ds>_<cfg>_<ds>_seed<NN>.npz
  [B] §5.6 分层分析     —— 直接读 etis_exp5_..._results.json
  [C] §3.1 病灶面积     —— 列出所有候选标签目录及其中位数，不猜
  [D] §3.1/§4.3 哈希审计 —— 自动定位图像目录后跑感知哈希
  [E] S1 自助区间稳定性  —— 多随机种子，判断 NLL 下限差异是否只是随机性
  [F] §4.1 两种重采样    —— 读 image_level_bootstrap / supplementary_ci

用法:
    cd verification
    python3 verify_round3_v30.py > verify_round3_out.txt 2>&1
    cat verify_round3_out.txt
"""
import json, os, sys, glob, itertools
import numpy as np
from pathlib import Path
from scipy import stats

# Paths resolve through src/paths.py so the script runs from a fresh clone with
# no configuration: the cached records bundled under data/raw_records/ are used
# when present, otherwise the workspace named by BACKBONE_CALIB_ROOT.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import CALIB, RECORDS_DIR, EXTERNAL_RECORDS_DIR  # noqa: E402

from paths import ROOT  # noqa: E402

REC = RECORDS_DIR
EXT = EXTERNAL_RECORDS_DIR
S3=[42,43,44]; S8=list(range(42,50)); NB,EPS=10,1e-12
A,B="A_independent","B_shared"

_p=_f=_s=0
def CHECK(lbl,c,pv,tol):
    global _p,_f
    if c is None or (isinstance(c,float) and np.isnan(c)): SKIP(lbl,"无法计算"); return
    ok=abs(c-pv)<=tol; _p+=ok; _f+=(not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {lbl:52s} 论文={pv:<10.4f} 重算={c:<10.4f} 差={c-pv:+.4f}")
def SKIP(lbl,why):
    global _s; _s+=1; print(f"  SKIP  {lbl:52s} {why}")
def SEC(t): print("\n"+"="*95); print(t); print("="*95)

def dece(c,l,nb=NB):
    n=len(c);e=0.;ed=np.linspace(0,1,nb+1)
    for i in range(nb):
        lo,hi=ed[i],ed[i+1]
        m=(c>=lo)&(c<=hi) if i==nb-1 else (c>=lo)&(c<hi)
        if m.sum(): e+=(m.sum()/n)*abs(c[m].mean()-l[m].mean())
    return float(e)
def aece(c,l,nb=NB):
    n=len(c);o=np.argsort(c);cs,ls=c[o],l[o];e=0.
    for i in np.array_split(np.arange(n),nb):
        if len(i): e+=(len(i)/n)*abs(cs[i].mean()-ls[i].mean())
    return float(e)
def brier(c,l): return float(np.mean((c-l)**2))
def nll(c,l):
    p=np.clip(c,EPS,1-EPS); return float(-np.mean(l*np.log(p)+(1-l)*np.log(1-p)))
def tt(d):
    t,p=stats.ttest_1samp(d,0.); sd=np.std(d,ddof=1)
    return float(np.mean(d)),float(t),float(p),float(np.mean(d)/sd) if sd>0 else np.nan

# ---------------------------------------------------------------- [A] 外部数据集
SEC("[A] §5.5 外部数据集（命名: <ds>_<cfg>_<ds>_seed<NN>.npz）")
def ext_top1(ds,cfg,s):
    p=EXT/f"{ds}_{cfg}_{ds}_seed{s}.npz"
    if not p.exists(): return None
    z=np.load(p); c=z["confidence"].astype(float); l=z["is_tp"].astype(int); i=z["image_idx"].astype(int)
    b={}
    for a_,b_,c_ in zip(c,l,i):
        if c_ not in b or a_>b[c_][0]: b[c_]=(a_,b_)
    k=sorted(b); return np.array([b[j][0] for j in k]), np.array([b[j][1] for j in k])
def ext_perm(ds,cfg,s,thr=0.001):
    p=EXT/f"{ds}_{cfg}_{ds}_seed{s}.npz"
    if not p.exists(): return None
    z=np.load(p); c=z["confidence"].astype(float); l=z["is_tp"].astype(int); m=c>=thr
    return c[m],l[m]

SPEC={"cvc":(115,[("D-ECE",dece,0.0700,0.1021,0.0320,1.85,0.214),
                  ("Brier",brier,0.0312,0.0722,0.0410,4.19,0.053),
                  ("NLL",nll,0.1152,0.3030,0.1878,3.29,0.081)]),
      "etis":(32,[("D-ECE",dece,0.2157,0.2164,0.0007,0.02,0.989)])}
for ds,(pN,rows) in SPEC.items():
    if ext_top1(ds,A,42) is None:
        SKIP(f"{ds} 全部","文件仍不匹配"); continue
    na=[len(ext_top1(ds,A,s)[0]) for s in S3]; nb=[len(ext_top1(ds,B,s)[0]) for s in S3]
    print(f"      {ds} 每图 top-1 图像数  A={na}  B={nb}")
    CHECK(f"{ds} N̄",float(np.mean(na+nb)),pN,1.0)
    for mn,fn,pa,pb,pd_,pt,pp in rows:
        va=np.mean([fn(*ext_top1(ds,A,s)) for s in S3]); vb=np.mean([fn(*ext_top1(ds,B,s)) for s in S3])
        dd=np.array([fn(*ext_top1(ds,B,s))-fn(*ext_top1(ds,A,s)) for s in S3]); m,t,p,_=tt(dd)
        CHECK(f"{ds} {mn} A",float(va),pa,0.0006); CHECK(f"{ds} {mn} B",float(vb),pb,0.0006)
        CHECK(f"{ds} {mn} 差",m,pd_,0.0006); CHECK(f"{ds} {mn} t",t,pt,0.03); CHECK(f"{ds} {mn} p",p,pp,0.003)
    if ds=="etis":
        npm=[len(ext_perm(ds,c,s)[0]) for c in (A,B) for s in S3]
        print(f"      ETIS 宽松协议每模型预测数: {npm}")
        CHECK("ETIS 宽松 N 最小",float(min(npm)),40,0.5); CHECK("ETIS 宽松 N 最大",float(max(npm)),58,0.5)
        dd=np.array([dece(*ext_perm(ds,B,s))-dece(*ext_perm(ds,A,s)) for s in S3]); m,t,p,dz=tt(dd)
        CHECK("ETIS 宽松 p",p,0.006,0.002); CHECK("ETIS 宽松 dz",dz,7.49,0.06)
# mAP 来自 results/*.json
for f,ds,pa,pb,pd_,pt,pp in [("results_cvc.json","CVC",0.9583,0.9495,-0.0088,-1.19,0.357),
                             ("results_etis.json","ETIS",0.8431,0.8823,0.0393,3.75,0.064)]:
    p=ROOT/"results"/f
    if not p.exists(): SKIP(f"{ds} mAP","results json 不存在"); continue
    d=json.load(open(p))
    print(f"      {f} 顶层键: {list(d)[:6] if isinstance(d,dict) else type(d)}")

# ---------------------------------------------------------------- [B] 分层
SEC("[B] §5.6 ETIS 分层分析（tab:size）")
p=CALIB/"etis_exp5_etis_size_stratified_results.json"
d=json.load(open(p))
PAPER={"small":(0.2113,0.0054,0.0965,0.0230),
       "medium":(0.1669,0.0659,0.2131,0.0574),
       "large":(0.1214,0.0557,0.2256,0.0405)}
ns_all=[]
for st,(ma,sa,mb,sb) in PAPER.items():
    va=[r["stratified"][st] for r in d["A_independent_etis"]]
    vb=[r["stratified"][st] for r in d["B_shared_etis"]]
    key=[k for k in va[0] if "dec" in k.lower() or k.lower()=="dece"]
    if not key: print(f"      {st} 可用字段: {list(va[0])}"); continue
    k=key[0]
    a=np.array([x[k] for x in va]); b=np.array([x[k] for x in vb])
    ns_all += [x["n"] for x in va]+[x["n"] for x in vb]
    CHECK(f"tab:size {st} A 均值",float(a.mean()),ma,0.0002)
    CHECK(f"tab:size {st} A SD",float(a.std(ddof=1)),sa,0.0002)
    CHECK(f"tab:size {st} B 均值",float(b.mean()),mb,0.0002)
    CHECK(f"tab:size {st} B SD",float(b.std(ddof=1)),sb,0.0002)
if ns_all:
    print(f"      各层每模型预测数: {sorted(set(ns_all))}")
    CHECK("分层预测数 最小",float(min(ns_all)),10,0.5)
    CHECK("分层预测数 最大",float(max(ns_all)),48,0.5)

# ---------------------------------------------------------------- [C] 病灶面积
SEC("[C] §3.1 病灶相对面积  论文: Kvasir 18.47%  CVC 9.81%  ETIS 2.85%  ETIS Q1 1.29%")
def areas(dirpath):
    out=[]
    for f in glob.glob(os.path.join(dirpath,"*.txt")):
        for line in open(f):
            pr=line.split()
            if len(pr)>=5: out.append(float(pr[3])*float(pr[4]))
    return np.array(out)
cands=sorted({d for d in glob.glob(str(ROOT/"**"/"labels"/"*"),recursive=True) if os.path.isdir(d)})
print(f"      找到 {len(cands)} 个标签目录：")
for d_ in cands:
    a=areas(d_)
    if len(a)==0: print(f"        {d_:70s} (空)"); continue
    print(f"        {d_:70s} n={len(a):5d}  中位数={100*np.median(a):6.2f}%  Q1={100*np.percentile(a,25):6.2f}%")
print("      >>> 请人工把上表与论文的 18.47 / 9.81 / 2.85 / 1.29 对应")

# ---------------------------------------------------------------- [D] 哈希审计
SEC("[D] §3.1 近重复审计 / §4.3 重叠审计")
try:
    import imagehash; from PIL import Image
except Exception as e:
    SKIP("哈希审计",f"依赖不可用 {e}"); imagehash=None
if imagehash:
    imgdirs=sorted({d for d in glob.glob(str(ROOT/"**"/"images"/"*"),recursive=True) if os.path.isdir(d)})
    print(f"      找到 {len(imgdirs)} 个图像目录：")
    for d_ in imgdirs:
        n=len(glob.glob(os.path.join(d_,"*")))
        print(f"        {d_:70s} {n} 张")
    def phashes(d_):
        out={}
        for f in sorted(glob.glob(os.path.join(d_,"*"))):
            try: out[os.path.basename(f)]=imagehash.phash(Image.open(f))
            except Exception: pass
        return out
    def nearest(valdir,traindir):
        V,T=phashes(valdir),phashes(traindir)
        if not V or not T: return None
        tv=list(T.values()); res=[]
        for k,h in V.items(): res.append(min(h-t for t in tv))
        return np.array(res)
    def pick(pat):
        hits=[d for d in imgdirs if pat in d.lower()]
        return hits[0] if hits else None
    for ds,ntot,p0,p5,p10 in [("cvc",122,6,28,101),("etis",39,3,12,24)]:
        v=pick(f"{ds}") ; tr=None
        vs=[d for d in imgdirs if ds in d.lower() and ("val" in d.lower() or "valid" in d.lower())]
        ts=[d for d in imgdirs if ds in d.lower() and "train" in d.lower()]
        if not vs or not ts: SKIP(f"{ds} 近重复审计",f"未定位 train/val 目录"); continue
        dist=nearest(vs[0],ts[0])
        if dist is None: SKIP(f"{ds} 近重复审计","读图失败"); continue
        print(f"      {ds}: val={vs[0]} ({len(dist)} 张) train={ts[0]}")
        CHECK(f"{ds} 验证集张数",float(len(dist)),ntot,0.5)
        CHECK(f"{ds} 距离=0 张数",float((dist==0).sum()),p0,0.5)
        CHECK(f"{ds} 距离<=5 张数",float((dist<=5).sum()),p5,0.5)
        CHECK(f"{ds} 距离<=10 张数",float((dist<=10).sum()),p10,0.5)
    # 重叠审计：Kvasir 分类 vs 检测
    print("      重叠审计（论文: cls-tr∩det-val=2, cls-tr∩det-tr=10, cls-val∩det-tr=2, cls-val∩det-val=0）")
    print("      >>> 需要分类与检测两套目录路径，请从上面的图像目录列表里指认后我补脚本")

# ---------------------------------------------------------------- [E] 自助区间稳定性
SEC("[E] S1 自助区间稳定性（single_task seed42 的 NLL 下限，论文 0.394）")
z=np.load(REC/"kvasir_single_task_seed42_records.npz")
c=z["confidence"].astype(float); l=z["is_tp"].astype(int); i=z["image_idx"].astype(int)
b={}
for a_,b_,c_ in zip(c,l,i):
    if c_ not in b or a_>b[c_][0]: b[c_]=(a_,b_)
k=sorted(b); C=np.array([b[j][0] for j in k]); L=np.array([b[j][1] for j in k]); n=len(C)
los=[]
for seed in range(8):
    rng=np.random.default_rng(seed)
    v=[nll(C[idx],L[idx]) for idx in (rng.integers(0,n,n) for _ in range(1000))]
    los.append(np.percentile(v,2.5))
print(f"      8 个随机种子下的 2.5% 分位: " + " ".join(f"{x:.4f}" for x in los))
print(f"      范围 {min(los):.4f}--{max(los):.4f}，论文 0.394")
CHECK("论文值落在自助随机范围内",float(min(los)<=0.394<=max(los)),1.0,0.5)

# ---------------------------------------------------------------- [F] 两种重采样
SEC("[F] §4.1 两种重采样方案（论文称端点差 < 0.006）")
d1=json.load(open(CALIB/"image_level_bootstrap.json"))
d2=json.load(open(CALIB/"supplementary_ci_results.json"))
def rows_of(d,cfg):
    v=d[cfg]
    return v["rows"] if isinstance(v,dict) and "rows" in v else None
r1,r2=rows_of(d1,A),rows_of(d2,A)
print(f"      image_level_bootstrap  A 的 rows 类型 {type(r1).__name__}，首元素: {str(r1[0])[:200] if r1 else None}")
print(f"      supplementary_ci       A 的 rows 类型 {type(r2).__name__}，首元素: {str(r2[0])[:200] if r2 else None}")
def endpoints(rows):
    out=[]
    for r in rows or []:
        if isinstance(r,dict):
            for kk,vv in r.items():
                if isinstance(vv,(list,tuple)) and len(vv)==2 and all(isinstance(x,(int,float)) for x in vv):
                    out.append((kk,tuple(vv)))
        elif isinstance(r,(list,tuple)):
            out.append(("row",tuple(r)))
    return out
e1,e2=endpoints(r1),endpoints(r2)
if e1 and e2 and len(e1)==len(e2):
    diffs=[max(abs(a[1][0]-b_[1][0]),abs(a[1][1]-b_[1][1])) for a,b_ in zip(e1,e2)]
    print(f"      配对端点数 {len(diffs)}  最大端点差 {max(diffs):.4f}")
    CHECK("两方案最大端点差 < 0.006",float(max(diffs)),0.0,0.006)
else:
    print(f"      端点数 {len(e1)} vs {len(e2)}，结构不匹配，需人工对照上面的首元素")

SEC("汇总"); print(f"  PASS {_p}   FAIL {_f}   SKIP {_s}")
