"""G0 — Bootstrap: seeds, logging, device.

The three primitives every module in mc-gqpso depends on. Import from here,
never re-implement.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from typing import Optional

import numpy as np

_LOGGERS_CONFIGURED: set[str] = set()
_ROOT_CONFIGURED = False


def set_all_seeds(seed: int) -> None:
    """Four-way seed contract: python-random, numpy, torch cpu, torch cuda.

    Must be called at every entry point (CLI, API request handler, test setup,
    benchmark run). Torch is imported lazily so the numpy-only engine has no
    hard torch dependency.
    """
    if seed is None:
        raise ValueError("seed must be an int, got None")
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # noqa: WPS433 (lazy import is intentional)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class _JsonFormatter(logging.Formatter):
    """Structured JSON records so downstream tooling can parse iteration logs.

    Any extra kwargs passed via `logger.info("...", extra={...})` are folded
    into the record — the QPSO loop uses this for {iteration, best_cost,
    wall_time} lines.
    """

    _STD_ATTRS = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in self._STD_ATTRS or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def _configure_root() -> None:
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return
    root = logging.getLogger("mc_gqpso")
    root.setLevel(os.environ.get("MC_GQPSO_LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.propagate = False
    _ROOT_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced JSON logger under mc_gqpso.<name>."""
    _configure_root()
    full = name if name.startswith("mc_gqpso") else f"mc_gqpso.{name}"
    logger = logging.getLogger(full)
    _LOGGERS_CONFIGURED.add(full)
    return logger


def pick_device(use_gpu: bool) -> str:
    """Return 'cuda' if requested AND available, else 'cpu' with a warning.

    Uses the string form so callers that don't import torch (numpy engine,
    tests) can still branch on it.
    """
    log = get_logger("device")
    if not use_gpu:
        log.info("device selected", extra={"device": "cpu", "requested": "cpu"})
        return "cpu"
    try:
        import torch
    except ImportError:
        log.warning(
            "torch not installed; falling back to cpu",
            extra={"device": "cpu", "requested": "cuda", "reason": "torch_missing"},
        )
        return "cpu"
    if torch.cuda.is_available():
        log.info("device selected", extra={"device": "cuda", "requested": "cuda"})
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Apple Silicon fallback for dev machines. Same residency / no-sync
        # discipline applies; the spec targets CUDA at deploy time.
        log.warning(
            "cuda unavailable; using mps",
            extra={"device": "mps", "requested": "cuda", "reason": "cuda_unavailable_mps_present"},
        )
        return "mps"
    log.warning(
        "cuda unavailable; falling back to cpu",
        extra={"device": "cpu", "requested": "cuda", "reason": "cuda_unavailable"},
    )
    return "cpu"


def wallclock() -> float:
    """Monotonic seconds since some fixed point — for wall_time logging."""
    return time.perf_counter()
