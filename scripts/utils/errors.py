from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time
import gc

import torch


@dataclass
class RunResult:
    model: str
    stage: str
    status: str
    seconds: float
    message: Optional[str] = None


def is_oom(e: BaseException) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return True
    return False


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


def classify_exception(e: BaseException) -> str:
    s = str(e).lower()

    if is_oom(e):
        return "OOM"
    if isinstance(e, FileNotFoundError):
        return "DATA_NOT_FOUND"
    if isinstance(e, PermissionError):
        return "PERMISSION"
    if isinstance(e, ConnectionError):
        return "NETWORK"
    if isinstance(e, OSError):
        if "cannot identify image file" in s or "image file is truncated" in s:
            return "BAD_MEDIA"
        if "certificate" in s or "ssl" in s:
            return "NETWORK_SSL"
        return "OS_ERROR"
    return "RUNTIME_ERROR"


def run_safe(model: str, stage: str, fn):
    t0 = time.perf_counter()
    try:
        fn()
        return RunResult(model=model, stage=stage, status="OK", seconds=time.perf_counter() - t0)
    except Exception as e:
        return RunResult(
            model=model,
            stage=stage,
            status=classify_exception(e),
            seconds=time.perf_counter() - t0,
            message=str(e)[:1200],
        )
    finally:
        cleanup_cuda()
