from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForObjectDetection

@dataclass
class LoadedModel:
    model_id: str
    kind: str  # 'ultralytics' | 'hf'
    device: str
    fp16: bool
    yolo: Optional[Any] = None
    processor: Optional[Any] = None
    hf_model: Optional[Any] = None
    label_to_cat_id: Optional[Dict[int,int]] = None

    def close(self) -> None:
        self.yolo = None
        self.processor = None
        self.hf_model = None
        self.label_to_cat_id = None

def infer_kind(model_id: str) -> str:
    if model_id.endswith(".pt"):
        return "ultralytics"
    if "/" in model_id:
        return "hf"
    raise ValueError(f"Unsupported model id format: {model_id}")

def to_device_str(device: str) -> str:
    if device in ("cuda", "cuda:0") and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def build_label_mapping_from_names(names: Dict[int,str], coco_name_to_id: Dict[str,int]) -> Dict[int,int]:
    mapping: Dict[int,int] = {}
    for k, name in names.items():
        try:
            kk = int(k)
        except Exception:
            continue
        n = str(name).strip().lower()
        if coco_name_to_id and n in coco_name_to_id:
            mapping[kk] = coco_name_to_id[n]
    return mapping

def load_model(model_id: str, device: str, fp16: bool, coco_name_to_id: Dict[str,int]) -> LoadedModel:
    device = to_device_str(device)
    kind = infer_kind(model_id)

    if kind == "ultralytics":
        m = YOLO(model_id)
        names = m.names if isinstance(m.names, dict) else {i: n for i, n in enumerate(m.names)}
        label_to_cat_id = build_label_mapping_from_names(names, coco_name_to_id) if coco_name_to_id else {}
        return LoadedModel(
            model_id=model_id,
            kind=kind,
            device=device,
            fp16=fp16,
            yolo=m,
            label_to_cat_id=label_to_cat_id,
        )

    # Hugging Face (RT-DETRv2 family checkpoints)
    # Use AutoModelForObjectDetection so Transformers picks the correct class from config.
    try:
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    except TypeError:
        # Older versions may not support use_fast kwarg
        processor = AutoImageProcessor.from_pretrained(model_id)

    hf_model = AutoModelForObjectDetection.from_pretrained(model_id)
    hf_model.to(device)
    hf_model.eval()
    if fp16 and device.startswith("cuda"):
        hf_model.half()

    id2label = getattr(hf_model.config, "id2label", {}) or {}
    cleaned: Dict[int, str] = {}
    for k, v in id2label.items():
        try:
            kk = int(k)
        except Exception:
            continue
        vv = str(v).strip().lower()
        if vv in ("background", "no object", "n/a", "none"):
            continue
        cleaned[kk] = vv

    label_to_cat_id = build_label_mapping_from_names(cleaned, coco_name_to_id) if coco_name_to_id else {}
    return LoadedModel(
        model_id=model_id,
        kind=kind,
        device=device,
        fp16=fp16,
        processor=processor,
        hf_model=hf_model,
        label_to_cat_id=label_to_cat_id,
    )


def yolo_predict_on_paths(lm: LoadedModel, img_paths: List[str], imgsz: int, conf: float, iou: float, batch: int):
    assert lm.yolo is not None
    return lm.yolo.predict(
        source=img_paths,
        stream=True,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=0 if lm.device.startswith("cuda") else "cpu",
        half=bool(lm.fp16 and lm.device.startswith("cuda")),
        verbose=False,
        batch=batch,
    )

@torch.inference_mode()
def hf_predict_pil(lm: LoadedModel, pil: Image.Image, conf: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert lm.processor is not None and lm.hf_model is not None
    inputs = lm.processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(lm.device) for k, v in inputs.items()}
    if lm.fp16 and lm.device.startswith("cuda") and "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].half()

    outputs = lm.hf_model(**inputs)
    w, h = pil.size
    res = lm.processor.post_process_object_detection(outputs, threshold=conf, target_sizes=[(h, w)])[0]
    boxes = res["boxes"].detach().cpu().numpy()
    scores = res["scores"].detach().cpu().numpy()
    labels = res["labels"].detach().cpu().numpy()
    return boxes, scores, labels
