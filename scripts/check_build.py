from __future__ import annotations
# --- BOOTSTRAP: make repo root importable when running as a script ---
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
# --- /BOOTSTRAP ---
import argparse
import os
import csv
import numpy as np
from PIL import Image
import torch

from scripts.utils.io_utils import ensure_dir
from scripts.utils.errors import run_safe
from scripts.utils.coco_utils import load_coco, build_category_name_to_id
from scripts.utils.model_loader import load_model, hf_predict_pil, yolo_predict_on_paths

def parse_args():
    p = argparse.ArgumentParser(description="(1) Smoke-check: load models and run a tiny inference.")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--out", default="runs/check_build.csv")
    p.add_argument("--coco-ann", default="", help="Optional COCO annotations JSON for label mapping.")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(os.path.dirname(args.out))

    coco_name_to_id = {}
    if args.coco_ann:
        coco = load_coco(args.coco_ann)
        coco_name_to_id = build_category_name_to_id(coco)

    img = (np.random.rand(args.imgsz, args.imgsz, 3) * 255).astype(np.uint8)
    pil = Image.fromarray(img)

    rows = []
    for m in args.models:
        def job():
            lm = load_model(m, device=args.device, fp16=args.fp16, coco_name_to_id=coco_name_to_id)
            try:
                if lm.kind == "ultralytics":
                    tmp_path = "/tmp/smoke.jpg"
                    pil.save(tmp_path)
                    it = yolo_predict_on_paths(lm, [tmp_path], imgsz=args.imgsz, conf=args.conf, iou=args.iou, batch=1)
                    _ = next(iter(it))
                else:
                    _ = hf_predict_pil(lm, pil, conf=args.conf)
            finally:
                lm.close()

        r = run_safe(m, "smoke", job)
        gpu_mem = ""
        if torch.cuda.is_available():
            try:
                gpu_mem = str(torch.cuda.memory_allocated() // (1024*1024))
            except Exception:
                gpu_mem = ""
        rows.append({
            "model": m,
            "stage": r.stage,
            "status": r.status,
            "seconds": f"{r.seconds:.3f}",
            "gpu_mem_mb_after": gpu_mem,
            "message": r.message or "",
        })
        print(f"[{r.status}] {m} ({r.seconds:.2f}s)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)

    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
