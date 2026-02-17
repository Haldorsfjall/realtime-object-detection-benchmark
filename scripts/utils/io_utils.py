from __future__ import annotations
import os
import re
import json
from typing import Any

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def sanitize_model_id(model_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)
    return s.strip("_")

def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)
