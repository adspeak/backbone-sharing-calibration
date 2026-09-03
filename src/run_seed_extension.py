"""
Seed extension: raise the four configurations entering the interaction test from
six seeds to eight.

Only the four configurations required by the interaction test are extended, with
seeds 48 and 49 added to each (8 runs):

    A_independent (k=0)         LAG-LGFF head, separate trunks
    B_shared (k=11)             LAG-LGFF head, fully shared
    gapfc_A_independent (k=0)   GAP+FC head, separate trunks
    gapfc_B_shared (k=11)       GAP+FC head, fully shared

F_frozen is not extended: its p-values are far from significance and two extra
seeds could not plausibly change the conclusion. C_shallow and C_deep are not
extended either, since the trend test already reaches significance on every
metric.

Design constraints:
  - the training protocol is identical to the original runs (N = 15000 steps,
    reusing train_one and train_one_gapfc, with all hyperparameters imported
    unchanged from train_core.py); otherwise old and new seeds could not be
    pooled in a paired test
  - output directories follow the same naming as the original runs
  - the underlying training functions skip any configuration whose final
    checkpoint already exists, so seeds 42-47 cannot be overwritten
  - results are written to a separate summary file

Usage:
    python run_seed_extension.py
    # or in the background:
    #   nohup python run_seed_extension.py > seed_extension.log 2>&1 &
"""

import sys, os, json, time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import GPUData
from model import make_base
from train_core import train_one
from run_head_ablation import train_one_gapfc
from paths import RUNS, RESULTS

NEW_SEEDS = [48, 49]
N_ITERS = 15000  # must match the original runs, or seeds cannot be pooled

# (configuration name, share_until, is GAP+FC variant)
CONFIGS = [
    ('A_independent',       0,  False),
    ('B_shared',            11, False),
    ('gapfc_A_independent', 0,  True),
    ('gapfc_B_shared',      11, True),
]


if __name__ == '__main__':
    dev = 'cuda'
    D = GPUData(dev)
    base = make_base(nc=1, device=dev)

    total = len(CONFIGS) * len(NEW_SEEDS)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Seed extension: 4 configurations x {len(NEW_SEEDS)} new seeds = {total} runs")
    print(f"About 45 min per run; roughly {total * 45 / 60:.1f} hours in total")
    print(f"New seeds: {NEW_SEEDS}, N={N_ITERS} steps, matching the original runs")
    print("=" * 72)

    res = {}
    t0 = time.time()

    for name, k, is_gapfc in CONFIGS:
        for s in NEW_SEEDS:
            print("\n" + "=" * 72)
            print(f"▶ {name} seed{s}  ({'GAP+FC head' if is_gapfc else 'LAG-LGFF head'})")
            print("=" * 72)

            out_dir = f'{RUNS}/{name}_seed{s}'

            if is_gapfc:
                log = train_one_gapfc(name, k, s, D, base, N=N_ITERS,
                                      out_dir=out_dir, device=dev)
            else:
                log = train_one(name, k, s, D, base, N=N_ITERS,
                                out_dir=out_dir, device=dev)

            res[f'{name}_s{s}'] = log['val'][-1]
            json.dump(res, open(f'{RESULTS}/main_results_seedext.json', 'w'), indent=2)

            elapsed = (time.time() - t0) / 3600
            done = len(res)
            print(f"  [progress] {done}/{total} done, {elapsed:.2f} h elapsed")

    print(f"\nall done in {(time.time()-t0)/3600:.2f} h")
    print(f"results: {RESULTS}/main_results_seedext.json")
    print("\nNext: update the SEEDS lists in the analysis scripts to include 48 and 49")
