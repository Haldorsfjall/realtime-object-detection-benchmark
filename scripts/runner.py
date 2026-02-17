from __future__ import annotations
# --- BOOTSTRAP: make repo root importable when running as a script ---
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
# --- /BOOTSTRAP ---
import argparse
import subprocess
import sys
import os

def parse_args():
    p = argparse.ArgumentParser(description="Run smoke + COCO + video sequentially.")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--coco-images", default="data/coco/val2017")
    p.add_argument("--coco-ann", default="data/coco/annotations/instances_val2017.json")
    p.add_argument("--video", default="data/vtest.avi")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--skip-coco", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    return p.parse_args()

def main():
    a = parse_args()

    _ENV = os.environ.copy()
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _ENV['PYTHONPATH'] = _ROOT + (os.pathsep + _ENV['PYTHONPATH'] if _ENV.get('PYTHONPATH') else '')

    if not a.skip_smoke:
        cmd = [sys.executable, "scripts/check_build.py", "--device", a.device]
        if a.fp16: cmd.append("--fp16")
        cmd += ["--imgsz", str(a.imgsz), "--conf", str(max(a.conf, 0.001)), "--iou", str(a.iou), "--models"] + a.models
        print("== smoke ==")
        subprocess.run(cmd, check=False, env=_ENV)

    if not a.skip_coco:
        cmd = [sys.executable, "scripts/eval_coco.py", "--coco-images", a.coco_images, "--coco-ann", a.coco_ann, "--device", a.device]
        if a.fp16: cmd.append("--fp16")
        cmd += ["--imgsz", str(a.imgsz), "--conf", str(a.conf), "--iou", str(a.iou), "--models"] + a.models
        print("== coco ==")
        subprocess.run(cmd, check=False, env=_ENV)

    if not a.skip_video:
        cmd = [sys.executable, "scripts/bench_video.py", "--video", a.video, "--device", a.device]
        if a.fp16: cmd.append("--fp16")
        cmd += ["--imgsz", str(a.imgsz), "--conf", str(a.conf), "--iou", str(a.iou), "--warmup", str(a.warmup), "--frames", str(a.frames), "--models"] + a.models
        print("== video ==")
        subprocess.run(cmd, check=False, env=_ENV)

if __name__ == "__main__":
    main()
