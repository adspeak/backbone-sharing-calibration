"""
Head ablation: GAP+FC classification head at the two extreme sharing depths,
k=0 (independent) and k=11 (fully shared), across seeds 42-47 (12 runs).

The training protocol is identical to the depth sweep; only the model class
changes (JointGIModel -> JointGIModelGAPFC). Results are written to a separate
file so that the main experiment's outputs are untouched. Resumable.
"""
import os, sys, json, copy, time, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import GPUData
from model import make_base
from model_gapfc import JointGIModelGAPFC          # <<< the only model change
from evaluate import eval_all
from paths import RUNS, RESULTS
# ---- identical training constants to train_core.py ----
from train_core import (LR0, LAM0, ACCUM, CLIP, WARMUP, BS_CLS, BS_DET,
                         EMA_DECAY, EMA, build_optimizer, lr_at)

def train_one_gapfc(config_name, share_until, seed, D, base_dm, N=15000,
                    out_dir=None, device='cuda', eval_every=500):
    os.makedirs(out_dir, exist_ok=True)
    fin_path = f'{out_dir}/final.pt'; ck_path = f'{out_dir}/ckpt.pt'
    if os.path.exists(fin_path) and os.path.getsize(fin_path) > 1e6:
        try:
            torch.load(fin_path, map_location='cpu', weights_only=False)
            log = json.load(open(f'{out_dir}/log.json'))
            print(f"  [done] {config_name} seed{seed} already complete, skipping"); return log
        except Exception: pass
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = JointGIModelGAPFC(share_until, base_dm).to(device)   # <<< the only differing line
    det_loss_fn = base_dm.init_criterion()
    ce = nn.CrossEntropyLoss()
    opt = build_optimizer(model); ema = EMA(model)
    gen = torch.Generator(device=device).manual_seed(seed)
    start_it = 0; log = {'iter': [], 'l_cls': [], 'l_det': [], 'val': []}
    if os.path.exists(ck_path):
        try:
            c = torch.load(ck_path, map_location=device, weights_only=False)
            model.load_state_dict(c['model']); opt.load_state_dict(c['opt'])
            ema.ema.load_state_dict(c['ema']); start_it = c['it']; log = c['log']
            print(f"  [resume] from iter {start_it}")
        except Exception as e:
            print(f"  [warn] corrupt checkpoint, starting over: {e}")
    model.train()
    for it in range(start_it, N):
        for g in opt.param_groups: g['lr'] = lr_at(it, N)
        xc, yc = D.sample_cls(BS_CLS, gen)
        cls_logits, _ = model(xc, task='cls')[0]
        l_cls = ce(cls_logits, yc)
        bd = D.sample_det(BS_DET, gen)
        _, det_out = model(bd['img'], task='det')
        l_det = det_loss_fn(det_out, bd)[0].sum() / BS_DET
        loss = (l_cls + LAM0 * l_det) / ACCUM
        if not torch.isfinite(loss):
            print(f"  [warn] non-finite loss at iter {it}, skipping"); opt.zero_grad(set_to_none=True); continue
        loss.backward()
        if (it + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CLIP)
            opt.step(); opt.zero_grad(set_to_none=True)
        ema.update(model)
        if (it + 1) % 500 == 0:
            log['iter'].append(it+1); log['l_cls'].append(float(l_cls)); log['l_det'].append(float(l_det))
        if (it + 1) % eval_every == 0:
            v = eval_all(ema.ema, base_dm, D, device); log['val'].append(v)
            print(f"    val@{it+1}: acc={v['cls_acc']:.4f} F1={v['cls_macro_f1']:.4f} "
                  f"mAP50={v['det_mAP50']:.4f} R={v['det_recall']:.4f}")
            model.train()
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'ema': ema.ema.state_dict(), 'it': it+1, 'log': log}, ck_path)
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in ema.ema.state_dict().items()}
    torch.save({'model': sd, 'config': config_name, 'share_until': share_until,
                'seed': seed, 'N': N, 'lam': LAM0, 'dtype': 'fp16', 'head': 'gapfc'}, fin_path)
    json.dump(log, open(f'{out_dir}/log.json','w'), indent=2)
    _sz = os.path.getsize(fin_path)/1e6
    assert _sz > 5, f"final.pt write failed ({_sz:.1f}MB)"
    torch.load(fin_path, map_location='cpu', weights_only=False)
    print(f"  [ok] checkpoint verified, {_sz:.1f} MB")
    if os.path.exists(ck_path): os.remove(ck_path)
    return log

if __name__ == '__main__':
    dev = 'cuda'
    D = GPUData(dev)
    base = make_base(nc=1, device=dev)
    CONFIGS = [('gapfc_A_independent', 0), ('gapfc_B_shared', 11)]   # the two extreme depths only
    SEEDS = [42,43,44,45,46,47]
    print(f"GPU: {torch.cuda.get_device_name(0)}  | GAP+FC ablation, A/B x 6 seeds = 12 runs")
    res = {}; t0 = time.time()
    for name, k in CONFIGS:
        for s in SEEDS:
            print("\n" + "="*72); print(f"▶ {name} seed{s}"); print("="*72)
            log = train_one_gapfc(name, k, s, D, base, N=15000,
                                  out_dir=f'{RUNS}/{name}_seed{s}', device=dev)
            res[f'{name}_s{s}'] = log['val'][-1]
            json.dump(res, open(f'{RESULTS}/main_results_gapfc.json','w'), indent=2)
    print(f"\nall done in {(time.time()-t0)/3600:.1f} h")
    print(f"results: {RESULTS}/main_results_gapfc.json")
