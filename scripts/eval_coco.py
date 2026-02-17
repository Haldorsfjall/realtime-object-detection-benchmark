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
from typing import List, Dict, Any
from PIL import Image
import numpy as np

from scripts.utils.io_utils import ensure_dir, sanitize_model_id, write_json
from scripts.utils.errors import run_safe
from scripts.utils.coco_utils import load_coco, build_image_maps, build_category_name_to_id, coco_eval_from_predictions
from scripts.utils.model_loader import load_model, hf_predict_pil, yolo_predict_on_paths

def parse_args():
    p = argparse.ArgumentParser(description="(2) COCO eval: generate COCO-format predictions and run COCOeval.")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--coco-images", required=True)
    p.add_argument("--coco-ann", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--out", default="runs/coco_metrics.csv")
    p.add_argument("--pred-dir", default="runs/preds")
    return p.parse_args()

def iter_yolo_results_chunked(lm, img_paths, imgsz: int, conf: float, iou: float, batch: int, chunk_mult: int = 16):
    # Ultralytics can be sensitive to huge lists; run in chunks to be safer.
    # Also disable any saving/GUI operations for pure benchmarking.
    chunk = max(int(batch), 1) * max(int(chunk_mult), 1)
    for i in range(0, len(img_paths), chunk):
        part = img_paths[i:i+chunk]
        results = lm.yolo.predict(
            source=part,
            stream=False,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=0 if lm.device.startswith("cuda") else "cpu",
            half=bool(lm.fp16 and lm.device.startswith("cuda")),
            verbose=False,
            batch=batch,
            save=False,
            show=False,
        )
        for r in results:
            yield r

def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    return np.stack([x1, y1, (x2-x1), (y2-y1)], axis=1)

def main():
    args = parse_args()
    ensure_dir(os.path.dirname(args.out))
    ensure_dir(args.pred_dir)

    coco = load_coco(args.coco_ann)
    file_to_id, _ = build_image_maps(coco)
    coco_name_to_id = build_category_name_to_id(coco)

    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)
    img_paths: List[str] = []
    img_id_by_path: Dict[str,int] = {}
    for im in imgs:
        fn = im["file_name"]
        pth = os.path.join(args.coco_images, fn)
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"Image not found: {pth}")
        img_paths.append(pth)
        img_id_by_path[pth] = im["id"]

    rows = []
    for model_id in args.models:
        safe_name = sanitize_model_id(model_id)
        pred_path = os.path.join(args.pred_dir, f"{safe_name}_predictions.json")
        metrics_holder = {}

        def job():
            nonlocal metrics_holder
            lm = load_model(model_id, device=args.device, fp16=args.fp16, coco_name_to_id=coco_name_to_id)
            preds: List[Dict[str,Any]] = []
            try:
                if lm.kind == "ultralytics":
                    for r in iter_yolo_results_chunked(lm, img_paths, imgsz=args.imgsz, conf=args.conf, iou=args.iou, batch=args.batch):
                        img_path = str(getattr(r, "path", ""))
                        if not img_path:
                            continue
                        base = os.path.basename(img_path)
                        img_id = img_id_by_path.get(img_path)
                        if img_id is None:
                            img_id = file_to_id.get(base)
                        if img_id is None:
                            continue
                        boxes = r.boxes
                        if boxes is None or boxes.shape[0] == 0:
                            continue
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        scores = boxes.conf.detach().cpu().numpy()
                        labels = boxes.cls.detach().cpu().numpy().astype(int)
                        xywh = xyxy_to_xywh(xyxy)
                        for b, s, lab in zip(xywh, scores, labels):
                            cat_id = (lm.label_to_cat_id or {}).get(int(lab))
                            if cat_id is None:
                                continue
                            preds.append({"image_id": int(img_id), "category_id": int(cat_id), "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "score": float(s)})
                else:
                    for pth in img_paths:
                        img_id = img_id_by_path[pth]
                        pil = Image.open(pth).convert("RGB")
                        boxes_xyxy, scores, labels = hf_predict_pil(lm, pil, conf=args.conf)
                        if boxes_xyxy.shape[0] == 0:
                            continue
                        xywh = xyxy_to_xywh(boxes_xyxy)
                        for b, s, lab in zip(xywh, scores, labels):
                            cat_id = (lm.label_to_cat_id or {}).get(int(lab))
                            if cat_id is None:
                                continue
                            preds.append({"image_id": int(img_id), "category_id": int(cat_id), "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "score": float(s)})
            finally:
                lm.close()

            write_json(pred_path, preds)
            metrics_holder = coco_eval_from_predictions(coco, pred_path)

        r = run_safe(model_id, "coco", job)
        if r.status == "OK":
            row = {"model": model_id, "status": r.status, "seconds": f"{r.seconds:.1f}", "pred_json": pred_path}
            row.update({k: f"{v:.4f}" for k, v in metrics_holder.items()})
            row["error"] = ""
        else:
            row = {"model": model_id, "status": r.status, "seconds": f"{r.seconds:.1f}", "pred_json": pred_path,
                   "mAP@[.5:.95]":"", "AP50":"", "AP75":"", "AP_small":"", "AP_medium":"", "AP_large":"",
                   "error": r.message or ""}
        rows.append(row)
        if r.status != "OK":
            print(f"  error: {r.message}")
        print(f"[{r.status}] {model_id} ({r.seconds/60.0:.1f} min) -> {pred_path}")

    fieldnames = ["model","status","seconds","mAP@[.5:.95]","AP50","AP75","AP_small","AP_medium","AP_large","pred_json","error"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        wri.writerows(rows)

    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
