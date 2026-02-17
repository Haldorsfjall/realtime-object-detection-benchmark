from __future__ import annotations
import os
from typing import Dict, Any, Tuple

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_coco(coco_ann: str) -> COCO:
    if not os.path.isfile(coco_ann):
        raise FileNotFoundError(f"COCO annotation file not found: {coco_ann}")
    return COCO(coco_ann)

def build_image_maps(coco: COCO) -> Tuple[Dict[str,int], Dict[int,Dict[str,Any]]]:
    imgs = coco.loadImgs(coco.getImgIds())
    file_to_id = {im["file_name"]: im["id"] for im in imgs}
    id_to_img = {im["id"]: im for im in imgs}
    return file_to_id, id_to_img

def build_category_name_to_id(coco: COCO) -> Dict[str,int]:
    cats = coco.loadCats(coco.getCatIds())
    return {c["name"].strip().lower(): c["id"] for c in cats}

def coco_eval_from_predictions(coco_gt: COCO, pred_json_path: str):
    coco_dt = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    s = coco_eval.stats
    return {
        "mAP@[.5:.95]": float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "AP_small": float(s[3]),
        "AP_medium": float(s[4]),
        "AP_large": float(s[5]),
    }
