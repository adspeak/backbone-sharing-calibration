"""
Non-parametric sensitivity analysis.

Repeats every paired comparison as a Wilcoxon signed-rank test alongside the
paired t-test, and flags any comparison where the two disagree on significance.

Note that with eight paired observations the smallest attainable Wilcoxon
p-value is 0.0078, so the non-parametric test has limited resolution at the
small-p end; several entries therefore share that value.

Reads the cached records; no inference, no GPU.

Usage:
    python stats_wilcoxon.py
"""
import numpy as np
from pathlib import Path
from scipy import stats
from paths import RECORDS_DIR

RECORDS = RECORDS_DIR
S8 = [42,43,44,45,46,47,48,49]
S6 = [42,43,44,45,46,47]
NB, EPS = 10, 1e-12

def load(run):
    p = RECORDS / f"kvasir_{run}_records.npz"
    if not p.exists(): return None, None
    d = np.load(p); return d["confidence"].astype(float), d["is_tp"].astype(int)

def dece(c,l,nb=NB):
    n=len(c); e=0.0; ed=np.linspace(0,1,nb+1)
    for i in range(nb):
        lo,hi=ed[i],ed[i+1]
        m=(c>=lo)&(c<=hi) if i==nb-1 else (c>=lo)&(c<hi)
        if m.sum()==0: continue
        e+=(m.sum()/n)*abs(c[m].mean()-l[m].mean())
    return e

def aece(c,l,nb=NB):
    n=len(c); o=np.argsort(c); cs,ls=c[o],l[o]; e=0.0
    for idx in np.array_split(np.arange(n),nb):
        if len(idx)==0: continue
        e+=(len(idx)/n)*abs(cs[idx].mean()-ls[idx].mean())
    return e

def brier(c,l): return np.mean((c-l)**2)
def nll(c,l):
    p=np.clip(c,EPS,1-EPS); return -np.mean(l*np.log(p)+(1-l)*np.log(1-p))

METRICS=[("D-ECE",dece),("Adaptive ECE",aece),("Brier",brier),("NLL",nll)]

def vals(cfg, seeds, fn):
    out=[]
    for s in seeds:
        c,l=load(f"{cfg}_seed{s}")
        out.append(fn(c,l) if c is not None else None)
    return out

def both_tests(a, b, label):
    pairs=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    if len(pairs)<2: print(f"    {label}: too few pairs"); return
    A=np.array([p[0] for p in pairs]); B=np.array([p[1] for p in pairs])
    t,pt=stats.ttest_rel(B,A)
    try:
        w,pw=stats.wilcoxon(B,A)
    except Exception as e:
        print(f"    {label}: Wilcoxon failed ({e})"); return
    agree = (pt<0.05)==(pw<0.05)
    print(f"    {label:<14} t-test p={pt:.4f} {'sig.' if pt<0.05 else 'n.s.':<4} | "
          f"Wilcoxon W={w:.1f} p={pw:.4f} {'sig.' if pw<0.05 else 'n.s.':<4} | "
          f"{'agree' if agree else '*** DISAGREE ***'} (n={len(pairs)})")

print("="*90)
print("Wilcoxon signed-rank sensitivity analysis, against the paired t-test")
print("="*90)

for mname, mfn in METRICS:
    print(f"\n### {mname}")
    A=vals("A_independent",S8,mfn); B=vals("B_shared",S8,mfn)
    gA=vals("gapfc_A_independent",S8,mfn); gB=vals("gapfc_B_shared",S8,mfn)
    F=vals("F_frozen",S6,mfn); B6=vals("B_shared",S6,mfn)

    both_tests(A,B,"A vs B")
    both_tests(gA,gB,"gapfc A vs B")
    both_tests(F,B6,"F_frozen vs B")

    # interaction
    lag=[]; gap=[]
    for i in range(len(S8)):
        if all(x[i] is not None for x in [A,B,gA,gB]):
            lag.append(B[i]-A[i]); gap.append(gB[i]-gA[i])
    lag=np.array(lag); gap=np.array(gap)
    t,pt=stats.ttest_rel(lag,gap)
    w,pw=stats.wilcoxon(lag,gap)
    agree=(pt<0.05)==(pw<0.05)
    print(f"    {'interaction':<14} t-test p={pt:.4f} {'sig.' if pt<0.05 else 'n.s.':<4} | "
          f"Wilcoxon W={w:.1f} p={pw:.4f} {'sig.' if pw<0.05 else 'n.s.':<4} | "
          f"{'agree' if agree else '*** DISAGREE ***'} (n={len(lag)})")

print("\n"+"="*90)
print("With n=8 the smallest attainable Wilcoxon p is 0.0078, so it has lower")
print("resolution than the t-test; what matters is whether the verdicts agree.")
print("="*90)