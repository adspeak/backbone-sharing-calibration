"""
Evaluation helpers.

Detection is evaluated through the official validator so that numbers are
comparable with standard reporting; classification uses top-1 accuracy and
macro-F1 over the whole validation set.
"""
import copy, torch, numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg
from sklearn.metrics import f1_score
from paths import DET_YAML, IMGSZ

@torch.no_grad()
def eval_cls(joint, D, device='cuda'):
    """Classification over the whole validation set: top-1 accuracy and macro-F1."""
    joint.eval()
    P, Y = [], []
    for x, y in D.iter_cls_val(bs=64):
        logits, _ = joint(x, task='cls')[0]
        P.append(logits.argmax(1).cpu()); Y.append(y.cpu())
    P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy()
    return {'cls_acc': float((P==Y).mean()),
            'cls_macro_f1': float(f1_score(Y, P, average='macro'))}

@torch.no_grad()
def eval_det_official(joint, base_dm, device='cuda'):
    """Transplant the joint detection pathway into a standard DetectionModel and
    evaluate with the official validator.

    Round-trip validated: official weights reproduce mAP@0.5 = 0.881 exactly.
    """
    joint.eval()
    dm = copy.deepcopy(base_dm)
    seq = dm.model
    # overwrite the corresponding modules of dm with the joint weights
    for i in range(joint.share_until):
        seq[i].load_state_dict(joint.shared[i].state_dict())
    for j, m in enumerate(joint.det_trunk):
        seq[joint.share_until + j].load_state_dict(m.state_dict())
    for i, m in enumerate(joint.det_neck):
        seq[11 + i].load_state_dict(m.state_dict())

    tmp = YOLO('yolov10s.pt')
    tmp.model = dm.to(device).eval()
    metrics = tmp.val(data=DET_YAML, imgsz=IMGSZ, batch=16, verbose=False,
                      plots=False, save_json=False, device=device)
    b = metrics.box
    return {'det_mAP50': float(b.map50), 'det_mAP50_95': float(b.map),
            'det_precision': float(b.mp), 'det_recall': float(b.mr)}

def eval_all(joint, base_dm, D, device='cuda'):
    r = {}
    r.update(eval_cls(joint, D, device))
    r.update(eval_det_official(joint, base_dm, device))
    return r
