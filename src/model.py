"""
Joint model: shared YOLOv10s backbone, LAG-LGFF classification head, YOLOv10
detection head.

share_until=k means backbone modules with index < k are shared between the two
tasks, while those with index >= k are private to each.
"""
import copy, torch, torch.nn as nn, torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg

def make_base(nc=1, device='cuda'):
    """Build a single-class YOLOv10s DetectionModel with COCO-pretrained weights."""
    src = YOLO('yolov10s.pt')
    yaml = copy.deepcopy(src.model.yaml)
    dm = DetectionModel(yaml, nc=nc, verbose=False)
    dm.load(src.model)                       # transfers 607 of 619 tensors
    dm.args = get_cfg(DEFAULT_CFG)           # supplies box/cls/dfl loss gains
    for p in dm.parameters():
        p.requires_grad_(True)               # undo the freeze applied by strip_optimizer
    return dm.to(device)

# ---------------- LAG-LGFF classification head (fixed component) ----------------
class LAGLGFF(nn.Module):
    """Local branch (stride-16 tap), global self-attention branch (stride-32 tap),
    and cross-attention fusion of the two."""
    def __init__(self, c_local=256, c_global=512, n_cls=8, dim=256):
        super().__init__()
        self.local_proj  = nn.Conv2d(c_local, dim, 1)
        self.global_proj = nn.Conv2d(c_global, dim, 1)
        self.pool = nn.AdaptiveAvgPool2d((7,7))
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(dim, n_cls))
    def forward(self, f_local, f_global):
        L = self.local_proj(self.pool(f_local))          # B,dim,7,7
        G = self.global_proj(f_global)                   # B,dim,7,7
        B, D, H, W = G.shape
        Lt = L.flatten(2).transpose(1,2); Gt = G.flatten(2).transpose(1,2)  # B,49,dim
        Ga, _ = self.attn(Gt, Gt, Gt); Ga = self.norm1(Gt + Ga)
        Ca, _ = self.cross(Ga, Lt, Lt); Ca = self.norm2(Ga + Ca)
        pooled = Ca.mean(1)                               # B,dim
        return self.head(pooled * self.gate(pooled))

class JointGIModel(nn.Module):
    def __init__(self, share_until, base_dm, n_cls=8):
        super().__init__()
        b = base_dm.model                                # nn.Sequential, 0..23
        self.share_until = share_until
        self.shared = nn.ModuleList([b[i] for i in range(share_until)])
        self.cls_trunk = nn.ModuleList([copy.deepcopy(b[i]) for i in range(share_until, 11)])
        self.det_trunk = nn.ModuleList([copy.deepcopy(b[i]) for i in range(share_until, 11)])
        self.det_neck  = nn.ModuleList([b[i] for i in range(11, 24)])
        self.det_save  = base_dm.save                    # skip indices the neck needs
        self.cls_head  = LAGLGFF(256, 512, n_cls)
        self._base = base_dm

    def _run_trunk(self, x, shared, private):
        taps = {}
        for i, m in enumerate(shared):
            x = m(x); taps[i] = x
        off = len(shared)
        for j, m in enumerate(private):
            x = m(x); taps[off + j] = x
        return x, taps                                   # taps indexed 0..10

    def forward(self, x, task='both'):
        cls_logits = det_out = None
        s = [m(x) for m in self.shared] if False else None
        # shared segment
        xt = x; shared_taps = {}
        for i, m in enumerate(self.shared):
            xt = m(xt); shared_taps[i] = xt
        # classification branch
        if task in ('cls','both'):
            xc = xt; taps = dict(shared_taps)
            for j, m in enumerate(self.cls_trunk):
                xc = m(xc); taps[self.share_until + j] = xc
            cls_logits = self.cls_head(taps[6], taps[9])
        # detection branch
        if task in ('det','both'):
            xd = xt; taps = dict(shared_taps)
            for j, m in enumerate(self.det_trunk):
                xd = m(xd); taps[self.share_until + j] = xd
            y = [None]*24
            for i in range(11): y[i] = taps[i]
            x2 = taps[10]
            for i, m in enumerate(self.det_neck):
                mi = 11 + i
                if m.f != -1:
                    x2 = (y[m.f] if isinstance(m.f,int) else [x2 if k==-1 else y[k] for k in m.f])
                x2 = m(x2); y[mi] = x2
            det_out = x2
        return (cls_logits, None), det_out

    def shared_params(self):
        return [p for m in self.shared for p in m.parameters() if p.requires_grad]
