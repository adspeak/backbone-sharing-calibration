"""
GAP+FC ablation of the joint model.

Identical to JointGIModel in every respect except the classification head, which
is replaced by global average pooling followed by a single fully connected layer
operating on taps[9]. The backbone sharing structure, the detection branch and
the training protocol are unchanged, so the substitution varies the capacity of
the competing head while holding the rest of the architecture fixed.
"""
import copy, torch, torch.nn as nn
from model import make_base  # same base constructor, identical backbone transfer

class GAPFCHead(nn.Module):
    """Global average pooling over taps[9] followed by one linear layer.
    No attention, no fusion."""
    def __init__(self, c_in=512, n_cls=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(c_in, n_cls)
    def forward(self, f_global):          # f_global = taps[9], (B,512,7,7)
        return self.fc(self.pool(f_global).flatten(1))

class JointGIModelGAPFC(nn.Module):
    def __init__(self, share_until, base_dm, n_cls=8):
        super().__init__()
        b = base_dm.model
        self.share_until = share_until
        self.shared    = nn.ModuleList([b[i] for i in range(share_until)])
        self.cls_trunk = nn.ModuleList([copy.deepcopy(b[i]) for i in range(share_until, 11)])
        self.det_trunk = nn.ModuleList([copy.deepcopy(b[i]) for i in range(share_until, 11)])
        self.det_neck  = nn.ModuleList([b[i] for i in range(11, 24)])
        self.det_save  = base_dm.save
        self.cls_head  = GAPFCHead(512, n_cls)     # <<< the only change
        self._base = base_dm

    def forward(self, x, task='both'):
        cls_logits = det_out = None
        xt = x; shared_taps = {}
        for i, m in enumerate(self.shared):
            xt = m(xt); shared_taps[i] = xt
        if task in ('cls','both'):
            xc = xt; taps = dict(shared_taps)
            for j, m in enumerate(self.cls_trunk):
                xc = m(xc); taps[self.share_until + j] = xc
            cls_logits = self.cls_head(taps[9])         # <<< reads taps[9] only
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
