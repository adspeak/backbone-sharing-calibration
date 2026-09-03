"""
Frozen-backbone control.

Same fully shared structure as B_shared, but the shared trunk is frozen at its
pretrained initialisation so that only the two heads are updated. This is a
diagnostic control rather than a causal isolation: freezing removes
representational adaptation to both tasks at once, so an effect observed here
could reflect either the absence of task competition or simple underfitting.
"""
import sys, os, json, copy, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import make_base, JointGIModel
from data import GPUData
from evaluate import eval_all
from train_core import (LR0, LAM0, ACCUM, CLIP, WARMUP, BS_CLS, BS_DET,
                         EMA, build_optimizer, lr_at)

def train_frozen(seed, D, base_dm, N=15000, out_dir=None, device='cuda', eval_every=500):
    os.makedirs(out_dir, exist_ok=True)
    fin = f'{out_dir}/final.pt'
    if os.path.exists(fin) and os.path.getsize(fin) > 1e6:
        print(f"  [done] frozen seed{seed} already complete, skipping"); return json.load(open(f'{out_dir}/log.json'))

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = JointGIModel(11, base_dm).to(device)      # fully shared
    # ---- freeze the shared trunk; only the two heads remain trainable ----
    for p in model.shared.parameters():
        p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  frozen trunk: {n_train/1e6:.2f}M trainable / {n_frozen/1e6:.2f}M frozen")

    det_loss_fn = base_dm.init_criterion()
    ce = nn.CrossEntropyLoss()
    opt = build_optimizer(model)                       # filters by requires_grad
    ema = EMA(model)
    gen = torch.Generator(device=device).manual_seed(seed)
    log = {'iter': [], 'l_cls': [], 'l_det': [], 'val': []}

    model.train()
    for it in range(N):
        for g in opt.param_groups: g['lr'] = lr_at(it, N)
        xc, yc = D.sample_cls(BS_CLS, gen)
        cl, _ = model(xc, task='cls')[0]
        l_cls = ce(cl, yc)
        bd = D.sample_det(BS_DET, gen)
        _, do = model(bd['img'], task='det')
        l_det = det_loss_fn(do, bd)[0].sum() / BS_DET
        loss = (l_cls + LAM0 * l_det) / ACCUM
        if not torch.isfinite(loss): opt.zero_grad(set_to_none=True); continue
        loss.backward()
        if (it+1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CLIP)
            opt.step(); opt.zero_grad(set_to_none=True)
        ema.update(model)
        if (it+1) % 500 == 0:
            log['iter'].append(it+1); log['l_cls'].append(float(l_cls)); log['l_det'].append(float(l_det))
        if (it+1) % eval_every == 0:
            v = eval_all(ema.ema, base_dm, D, device); log['val'].append(v); model.train()
            print(f"    val@{it+1}: acc={v['cls_acc']:.4f} mAP50={v['det_mAP50']:.4f}")

    sd = {k:(v.half() if v.is_floating_point() else v) for k,v in ema.ema.state_dict().items()}
    torch.save({'model':sd,'config':'F_frozen','seed':seed,'N':N,'dtype':'fp16'}, fin)
    json.dump(log, open(f'{out_dir}/log.json','w'), indent=2)
    assert os.path.getsize(fin)/1e6 > 5; torch.load(fin, map_location='cpu', weights_only=False)
    print(f"  [ok] wrote {os.path.getsize(fin)/1e6:.1f}MB")
    return log