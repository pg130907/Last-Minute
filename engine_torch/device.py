"""Torch device selection with the same three-way fallback as bootstrap.

Prefer CUDA (the deploy target — RTX 5060), then MPS (Apple Silicon dev
machines), then CPU. Precision is float32 everywhere (spec §5).
"""
from __future__ import annotations

import torch

from common.bootstrap import get_logger

_log = get_logger("device_torch")


def pick_torch_device(use_gpu: bool = True) -> torch.device:
    if not use_gpu:
        _log.info("torch device selected", extra={"device": "cpu", "requested": "cpu"})
        return torch.device("cpu")
    if torch.cuda.is_available():
        _log.info("torch device selected", extra={"device": "cuda", "requested": "cuda"})
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        _log.warning(
            "cuda unavailable; using mps",
            extra={"device": "mps", "requested": "cuda", "reason": "cuda_unavailable_mps_present"},
        )
        return torch.device("mps")
    _log.warning(
        "cuda unavailable; falling back to cpu",
        extra={"device": "cpu", "requested": "cuda", "reason": "cuda_unavailable"},
    )
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Backend-appropriate barrier — used only for benchmark timing, never inside
    the QPSO loop (spec §5: no sync in loop)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
