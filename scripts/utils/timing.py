from __future__ import annotations
import torch

def cuda_sync_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
