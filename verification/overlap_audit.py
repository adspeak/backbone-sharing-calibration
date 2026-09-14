#!/usr/bin/env python3
"""
Cross-task image overlap audit (manuscript Section 4.3).

Kvasir-SEG is derived from the polyps class of Kvasir v2, so the classification
and detection partitions can draw on the same source images. Filenames differ in
format between the two collections, so filename comparison is uninformative; the
audit compares image CONTENT with both MD5 and perceptual hashing.

Reported in the manuscript:
    classification-train  n detection-val    =  2
    classification-train  n detection-train  = 10
    classification-val    n detection-train  =  2
    classification-val    n detection-val    =  0

The two images in the first cell are the ones excluded from every reported
evaluation, which is why the detection validation set is scored at 198 images
rather than 200. This script reads the images only to hash them; nothing here
feeds a model, and the cached records are untouched.

Requires the datasets, so it does not run from a bare clone. Point
BACKBONE_CALIB_ROOT at the workspace that holds them:

    export BACKBONE_CALIB_ROOT=/path/to/workspace
    cd verification
    python3 overlap_audit.py
"""
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import ROOT, METRICS_DIR  # noqa: E402

try:
    from PIL import Image
    import imagehash
    HAVE_HASH = True
except ImportError:
    HAVE_HASH = False

PAPER = {("cls-train", "det-val"): 2,
         ("cls-train", "det-train"): 10,
         ("cls-val", "det-train"): 2,
         ("cls-val", "det-val"): 0}

_pass = _fail = _skip = 0


def CHECK(label, got, want):
    global _pass, _fail
    ok = got == want
    _pass += ok
    _fail += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:44s} 论文={want:<6d} 重算={got:<6d} 差={got-want:+d}")


def SKIP(label, why):
    global _skip
    _skip += 1
    print(f"  SKIP  {label:44s} {why}")


def find_dir(*patterns):
    for pat in patterns:
        hits = [d for d in glob.glob(str(ROOT / pat), recursive=True) if os.path.isdir(d)]
        if hits:
            return hits[0]
    return None


DIRS = {
    "cls-train": find_dir("**/kvasir_split/train/polyps", "**/*[Ss]plit/train/polyps"),
    "cls-val":   find_dir("**/kvasir_split/val/polyps", "**/*[Ss]plit/val/polyps"),
    "det-train": find_dir("**/kvasir_seg*/images/train", "**/*[Kk]vasir*[Ss][Ee][Gg]*/**/images/train"),
    "det-val":   find_dir("**/kvasir_seg*/images/val", "**/*[Kk]vasir*[Ss][Ee][Gg]*/**/images/val"),
}

print("=" * 95)
print("§4.3 跨任务图像重叠审计")
print("=" * 95)
missing = [k for k, v in DIRS.items() if v is None]
if missing:
    for k in missing:
        print(f"      未找到 {k}")
    print("      >>> 请设置 BACKBONE_CALIB_ROOT 指向包含数据集的工作目录")
    sys.exit(0)
for k, v in DIRS.items():
    n = len([f for f in os.listdir(v) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"      {k:10s} {v}  ({n} 张)")


def files_of(d):
    return sorted(f for f in glob.glob(os.path.join(d, "*"))
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))


def md5_set(d):
    out = {}
    for f in files_of(d):
        with open(f, "rb") as fh:
            out.setdefault(hashlib.md5(fh.read()).hexdigest(), []).append(os.path.basename(f))
    return out


def phash_set(d):
    out = {}
    for f in files_of(d):
        h = str(imagehash.phash(Image.open(f).convert("RGB")))
        out.setdefault(h, []).append(os.path.basename(f))
    return out


print("\n--- MD5 内容哈希 ---")
md5 = {k: md5_set(v) for k, v in DIRS.items()}
overlaps_md5 = {}
for (a, b), want in PAPER.items():
    shared = set(md5[a]) & set(md5[b])
    overlaps_md5[(a, b)] = shared
    CHECK(f"{a} n {b}  (MD5)", len(shared), want)

if HAVE_HASH:
    print("\n--- 感知哈希 (pHash, 距离 0) ---")
    ph = {k: phash_set(v) for k, v in DIRS.items()}
    for (a, b), want in PAPER.items():
        shared = set(ph[a]) & set(ph[b])
        CHECK(f"{a} n {b}  (pHash)", len(shared), want)
else:
    for (a, b) in PAPER:
        SKIP(f"{a} n {b}  (pHash)", "imagehash/PIL 不可用")

print("\n--- 被排除的两张检测验证图 ---")
shared = overlaps_md5[("cls-train", "det-val")]
found = sorted({n for h in shared for n in md5["det-val"][h]})
print(f"      审计命中: {found}")
try:
    recorded = sorted(json.load(open(METRICS_DIR / "excluded_val_images.json")))
    print(f"      已记录  : {recorded}")
    CHECK("与 excluded_val_images.json 一致", int(found == recorded), 1)
except FileNotFoundError:
    SKIP("excluded_val_images.json 比对", "文件不存在")

print("\n" + "=" * 95)
print(f"  PASS {_pass}   FAIL {_fail}   SKIP {_skip}")
print("=" * 95)
