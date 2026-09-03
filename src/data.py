"""GPU-resident dataset.

Both task datasets are cached in device memory as 8-bit arrays and augmented on
the GPU, which avoids host-to-device transfer in the training loop.
"""
import glob, os, numpy as np, torch
from PIL import Image
from paths import CLS_DIR, DET_DIR, CLASSES, IMGSZ

def _letterbox(im, s=IMGSZ):
    W, H = im.size; r = min(s/W, s/H)
    nw, nh = int(round(W*r)), int(round(H*r))
    cv = Image.new('RGB', (s, s), (114,114,114))
    px, py = (s-nw)//2, (s-nh)//2
    cv.paste(im.resize((nw,nh), Image.BILINEAR), (px,py))
    return cv, r, px, py

class GPUData:
    def __init__(self, device='cuda', cache=None):
        cache = str(cache) if cache is not None else str(CACHE)
        os.makedirs(cache, exist_ok=True)
        self.device = device
        S = IMGSZ
        # ---------- classification: load or build cache ----------
        cx = f'{cache}/cls_train_x.npy'
        if not os.path.exists(cx):
            self._build_cls(cache, S)
        self.Xc = torch.from_numpy(np.load(f'{cache}/cls_train_x.npy')).to(device)  # uint8 NHWC
        self.Yc = torch.from_numpy(np.load(f'{cache}/cls_train_y.npy')).to(device)
        self.Xcv= torch.from_numpy(np.load(f'{cache}/cls_val_x.npy')).to(device)
        self.Ycv= torch.from_numpy(np.load(f'{cache}/cls_val_y.npy')).to(device)
        # ---------- detection ----------
        dx = f'{cache}/det_train_x.npy'
        if not os.path.exists(dx):
            self._build_det(cache, S)
        self.Xd = torch.from_numpy(np.load(f'{cache}/det_train_x.npy')).to(device)
        bo = np.load(f'{cache}/det_train_b.npy', allow_pickle=True)
        self.maxb = max(1, max(len(b) for b in bo))
        self.Bpad = torch.zeros(len(bo), self.maxb, 4, device=device)
        self.Bcnt = torch.zeros(len(bo), dtype=torch.long, device=device)
        for i, b in enumerate(bo):
            b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
            n = len(b)
            if n:
                self.Bpad[i, :n] = torch.from_numpy(b).to(device); self.Bcnt[i] = n
        self.n_cls = self.Xc.shape[0]; self.n_det = self.Xd.shape[0]
        print(f"GPUData: cls_train={self.n_cls} cls_val={self.Xcv.shape[0]} det_train={self.n_det}")

    def _build_cls(self, cache, S):
        for split in ['train','val']:
            items = [(f,ci) for ci,c in enumerate(CLASSES)
                     for f in sorted(glob.glob(f'{CLS_DIR}/{split}/{c}/*'))]
            X = np.zeros((len(items),S,S,3), np.uint8); Y = np.zeros(len(items), np.int64)
            for i,(f,y) in enumerate(items):
                X[i] = np.asarray(Image.open(f).convert('RGB').resize((S,S), Image.BILINEAR)); Y[i]=y
            np.save(f'{cache}/cls_{split}_x.npy', X); np.save(f'{cache}/cls_{split}_y.npy', Y)

    def _build_det(self, cache, S):
        for split in ['train','val']:
            imgs = sorted(glob.glob(f'{DET_DIR}/images/{split}/*.*'))
            X = np.zeros((len(imgs),S,S,3), np.uint8); B = []
            for i,f in enumerate(imgs):
                im = Image.open(f).convert('RGB'); cv,r,px,py = _letterbox(im,S)
                X[i] = np.asarray(cv); W,H = im.size
                lb = f'{DET_DIR}/labels/{split}/'+os.path.splitext(os.path.basename(f))[0]+'.txt'
                bx = []
                if os.path.exists(lb):
                    for line in open(lb):
                        p = line.split()
                        if len(p)==5:
                            _,cx,cy,w,h = map(float,p)
                            nw,nh = int(round(W*r)),int(round(H*r))
                            bx.append([(cx*nw+px)/S,(cy*nh+py)/S,w*nw/S,h*nh/S])
                B.append(np.array(bx,np.float32) if bx else np.zeros((0,4),np.float32))
            np.save(f'{cache}/det_{split}_x.npy', X)
            np.save(f'{cache}/det_{split}_b.npy', np.array(B,dtype=object), allow_pickle=True)

    def _aug(self, x):  # B,H,W,3 uint8 -> B,3,H,W float in [0,1], with flips
        x = x.permute(0,3,1,2).float().div(255)
        if torch.rand(1, device=self.device) < 0.5: x = x.flip(3)
        return x

    def sample_cls(self, bs, gen):
        idx = torch.randint(0, self.n_cls, (bs,), generator=gen, device=self.device)
        x = self._aug(self.Xc[idx])
        # classification additionally gets vertical flip and brightness jitter
        if torch.rand(1, device=self.device) < 0.5: x = x.flip(2)
        f = 1 + (torch.rand(bs,1,1,1, device=self.device)-0.5)*0.4
        x = (x*f).clamp(0,1)
        return x, self.Yc[idx]

    def sample_det(self, bs, gen):
        idx = torch.randint(0, self.n_det, (bs,), generator=gen, device=self.device)
        imgs = self.Xd[idx].permute(0,3,1,2).float().div(255)
        flip = torch.rand(1, device=self.device) < 0.5
        if flip: imgs = imgs.flip(3)
        bidx, cls, boxes = [], [], []
        for bi, i in enumerate(idx.tolist()):
            n = int(self.Bcnt[i])
            for k in range(n):
                b = self.Bpad[i,k].clone()
                if flip: b[0] = 1 - b[0]
                bidx.append(bi); cls.append(0.0); boxes.append(b)
        if boxes:
            boxes = torch.stack(boxes)
            bidx = torch.tensor(bidx, dtype=torch.float, device=self.device)
            cls  = torch.tensor(cls, device=self.device).unsqueeze(1)
        else:
            boxes = torch.zeros(0,4,device=self.device)
            bidx = torch.zeros(0,device=self.device); cls = torch.zeros(0,1,device=self.device)
        return {'img': imgs, 'batch_idx': bidx, 'cls': cls, 'bboxes': boxes}

    def iter_cls_val(self, bs=64):
        for i in range(0, self.Xcv.shape[0], bs):
            x = self.Xcv[i:i+bs].permute(0,3,1,2).float().div(255)
            yield x, self.Ycv[i:i+bs]
