"""
Depth sweep: A / B / C_shallow / C_deep across seeds 42-47 (24 runs).

Resumable -- configurations whose final checkpoint already exists are skipped.
"""
import sys, json, time, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import GPUData
from model import make_base
from train_core import train_one
from paths import RUNS, RESULTS

dev = 'cuda'
D = GPUData(dev)
base = make_base(nc=1, device=dev)

CONFIGS = [('A_independent',0), ('C_shallow',5), ('C_deep',9), ('B_shared',11)]
SEEDS = [42,43,44,45,46,47]

print(f"GPU: {torch.cuda.get_device_name(0)}")
res = {}
t0 = time.time()
for name, k in CONFIGS:
    for s in SEEDS:
        print("\n" + "="*72); print(f"▶ {name} seed{s}"); print("="*72)
        log = train_one(name, k, s, D, base, N=15000,
                        out_dir=f'{RUNS}/{name}_seed{s}', device=dev)
        res[f'{name}_s{s}'] = log['val'][-1]
        json.dump(res, open(f'{RESULTS}/main_results.json','w'), indent=2)  # written after every run
print(f"\nall done in {(time.time()-t0)/3600:.1f} h")
print(f"results: {RESULTS}/main_results.json")
