"""Joint training loop.

Alternating task batches, gradient-norm-matched loss weighting, non-finite loss
guard, EMA of the weights, gradient accumulation, resumable checkpointing, and
verification that the final checkpoint was written correctly.
"""
import os, copy, json, math, torch, torch.nn as nn
# init_criterion() is used instead of v8DetectionLoss: v10Detect needs E2ELoss
from model import JointGIModel, make_base
from evaluate import eval_all

# ---------------- Training recipe ----------------
LR0   = 1e-4       # the transformer classification head needs a small LR
LAM0  = 0.0304     # calibrated so both losses contribute comparable gradient norms
ACCUM = 8          # nbs=64 / bs=8
CLIP  = 10.0
WARMUP= 500
BS_CLS= 16
BS_DET= 8
EMA_DECAY = 0.9999

class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters(): p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
                if e.dtype.is_floating_point:
                    e.mul_(self.decay).add_(m.detach(), alpha=1-self.decay)

def build_optimizer(model):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim <= 1 or 'bn' in n.lower() or 'norm' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': 5e-4},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=LR0, betas=(0.9, 0.999))

def lr_at(it, N):
    if it < WARMUP: return LR0 * it / WARMUP
    prog = (it - WARMUP) / (N - WARMUP)
    return LR0 * (1 - 0.99 * prog)                 # linear decay to 1% of LR0

def train_one(config_name, share_until, seed, D, base_dm, N=10000,
              out_dir=None, device='cuda', eval_every=500):
    os.makedirs(out_dir, exist_ok=True)
    fin_path = f'{out_dir}/final.pt'
    ck_path  = f'{out_dir}/ckpt.pt'

    # ---- resume: skip if already finished ----
    if os.path.exists(fin_path) and os.path.getsize(fin_path) > 1e6:
        try:
            ck = torch.load(fin_path, map_location='cpu', weights_only=False)
            log = json.load(open(f'{out_dir}/log.json'))
            print(f"  [done] {config_name} seed{seed} already complete, skipping")
            return log
        except Exception:
            pass

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = JointGIModel(share_until, base_dm).to(device)
    det_loss_fn = base_dm.init_criterion()   # E2ELoss, required by the v10Detect head
    ce = nn.CrossEntropyLoss()
    opt = build_optimizer(model)
    ema = EMA(model)
    gen = torch.Generator(device=device).manual_seed(seed)

    start_it = 0; log = {'iter': [], 'l_cls': [], 'l_det': [], 'val': []}
    # ---- restore from checkpoint ----
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

        # ---- classification batch ----
        xc, yc = D.sample_cls(BS_CLS, gen)
        cls_logits, _ = model(xc, task='cls')[0]
        l_cls = ce(cls_logits, yc)
        # ---- detection batch ----
        bd = D.sample_det(BS_DET, gen)
        _, det_out = model(bd['img'], task='det')
        l_det = det_loss_fn(det_out, bd)[0].sum() / BS_DET     # batch-average

        loss = (l_cls + LAM0 * l_det) / ACCUM
        if not torch.isfinite(loss):
            print(f"  [warn] non-finite loss at iter {it}, skipping batch"); opt.zero_grad(set_to_none=True); continue
        loss.backward()

        if (it + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], CLIP)
            opt.step(); opt.zero_grad(set_to_none=True)
        ema.update(model)

        if (it + 1) % 500 == 0:
            log['iter'].append(it+1); log['l_cls'].append(float(l_cls)); log['l_det'].append(float(l_det))
        # ---- periodic validation, using the EMA weights ----
        if (it + 1) % eval_every == 0:
            v = eval_all(ema.ema, base_dm, D, device)
            log['val'].append(v)
            print(f"    val@{it+1}: acc={v['cls_acc']:.4f} F1={v['cls_macro_f1']:.4f} "
                  f"mAP50={v['det_mAP50']:.4f} R={v['det_recall']:.4f}")
            model.train()
            # save resumable checkpoint (includes optimiser state)
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'ema': ema.ema.state_dict(), 'it': it+1, 'log': log}, ck_path)

    # ---- write final checkpoint (EMA weights, fp16) and verify ----
    sd = {k: (v.half() if v.is_floating_point() else v)
          for k, v in ema.ema.state_dict().items()}
    torch.save({'model': sd, 'config': config_name, 'share_until': share_until,
                'seed': seed, 'N': N, 'lam': LAM0, 'dtype': 'fp16'}, fin_path)
    json.dump(log, open(f'{out_dir}/log.json','w'), indent=2)

    _sz = os.path.getsize(fin_path)/1e6
    assert _sz > 5, f"final.pt write failed ({_sz:.1f}MB)"
    torch.load(fin_path, map_location='cpu', weights_only=False)  # read back to verify
    print(f"  [ok] checkpoint verified, {_sz:.1f} MB")
    if os.path.exists(ck_path): os.remove(ck_path)                # drop the large resume checkpoint
    return log
