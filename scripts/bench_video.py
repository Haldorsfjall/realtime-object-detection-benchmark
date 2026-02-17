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
import cv2
import numpy as np
from PIL import Image
import torch

from scripts.utils.io_utils import ensure_dir, sanitize_model_id
from scripts.utils.errors import run_safe
from scripts.utils.timing import cuda_sync_if_needed
from scripts.utils.coco_utils import load_coco, build_category_name_to_id
from scripts.utils.model_loader import load_model, hf_predict_pil

def parse_args():
    p = argparse.ArgumentParser(description="(3) Video benchmark: FPS + latency quantiles (e2e).")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--out", default="runs/video_bench.csv")
    p.add_argument("--save-frame-times", action="store_true")
    p.add_argument("--coco-ann", default="", help="Optional COCO ann json for mapping (not required for timing).")
    return p.parse_args()

def quantiles_ms(arr_ms: np.ndarray):
    if arr_ms.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.quantile(arr_ms, 0.50)),
            float(np.quantile(arr_ms, 0.95)),
            float(np.quantile(arr_ms, 0.99)))

def main():
    args = parse_args()
    ensure_dir(os.path.dirname(args.out))
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    coco_name_to_id = {}
    if args.coco_ann:
        coco = load_coco(args.coco_ann)
        coco_name_to_id = build_category_name_to_id(coco)

    rows = []
    for model_id in args.models:
        safe_name = sanitize_model_id(model_id)
        summary = {}
        frame_times = None

        def job():
            nonlocal summary, frame_times
            lm = load_model(model_id, device=args.device, fp16=args.fp16, coco_name_to_id=coco_name_to_id)
            try:
                cap = cv2.VideoCapture(args.video)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open video: {args.video}")

                decode_ms = []
                e2e_ms = []
                n = 0
                total = args.warmup + args.frames

                while n < total:
                    t0 = cv2.getTickCount()
                    ok, frame = cap.read()
                    t1 = cv2.getTickCount()
                    if not ok:
                        break

                    dm = (t1 - t0) * 1000.0 / cv2.getTickFrequency()
                    decode_ms.append(dm)

                    if args.device.startswith("cuda") and torch.cuda.is_available():
                        cuda_sync_if_needed(args.device)

                    t_start = cv2.getTickCount()

                    if lm.kind == "ultralytics":
                        _ = lm.yolo.predict(
                            source=frame,
                            imgsz=args.imgsz,
                            conf=args.conf,
                            iou=args.iou,
                            device=0 if lm.device.startswith("cuda") else "cpu",
                            half=bool(lm.fp16 and lm.device.startswith("cuda")),
                            verbose=False,
                            save=False,
                            show=False,
                        )
                    else:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(rgb)
                        _ = hf_predict_pil(lm, pil, conf=args.conf)

                    if args.device.startswith("cuda") and torch.cuda.is_available():
                        cuda_sync_if_needed(args.device)

                    t_end = cv2.getTickCount()
                    em = (t_end - t_start) * 1000.0 / cv2.getTickFrequency()

                    if n >= args.warmup:
                        e2e_ms.append(em)

                    n += 1

                cap.release()

                e2e = np.array(e2e_ms, dtype=np.float64)
                dec = np.array(decode_ms[args.warmup:args.warmup+len(e2e_ms)], dtype=np.float64)

                p50, p95, p99 = quantiles_ms(e2e)
                fps = float(len(e2e) / (np.sum(e2e) / 1000.0)) if len(e2e) > 0 else float("nan")
                dec_mean = float(np.mean(dec)) if dec.size else float("nan")

                summary = {
                    "fps_mean": fps,
                    "e2e_p50_ms": p50,
                    "e2e_p95_ms": p95,
                    "e2e_p99_ms": p99,
                    "decode_mean_ms": dec_mean,
                    "frames_measured": int(len(e2e)),
                }
                frame_times = e2e
            finally:
                try:
                    lm.close()
                except Exception:
                    pass

        r = run_safe(model_id, "video", job)
        if r.status == "OK":
            row = {"model": model_id, "status": r.status, "seconds": f"{r.seconds:.1f}",
                   "FPS_mean": f"{summary['fps_mean']:.2f}",
                   "e2e_p50_ms": f"{summary['e2e_p50_ms']:.2f}",
                   "e2e_p95_ms": f"{summary['e2e_p95_ms']:.2f}",
                   "e2e_p99_ms": f"{summary['e2e_p99_ms']:.2f}",
                   "decode_mean_ms": f"{summary['decode_mean_ms']:.3f}",
                   "frames_measured": str(summary["frames_measured"]),
                   "frame_times_csv": "", "error": ""}
            if args.save_frame_times and frame_times is not None:
                ft_path = f"runs/video_frames_{safe_name}.csv"
                with open(ft_path, "w", newline="", encoding="utf-8") as f:
                    wri = csv.writer(f)
                    wri.writerow(["frame_idx","e2e_ms"])
                    for i, v in enumerate(frame_times.tolist()):
                        wri.writerow([i, f"{v:.4f}"])
                row["frame_times_csv"] = ft_path
        else:
            row = {"model": model_id, "status": r.status, "seconds": f"{r.seconds:.1f}",
                   "FPS_mean":"", "e2e_p50_ms":"", "e2e_p95_ms":"", "e2e_p99_ms":"", "decode_mean_ms":"", "frames_measured":"", "frame_times_csv":"", "error": r.message or ""}
        rows.append(row)
        if r.status != "OK":
            print(f"  error: {r.message}")
        print(f"[{r.status}] {model_id} ({r.seconds:.1f}s)")

    fieldnames = ["model","status","seconds","FPS_mean","e2e_p50_ms","e2e_p95_ms","e2e_p99_ms","decode_mean_ms","frames_measured","frame_times_csv","error"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        wri.writerows(rows)

    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
