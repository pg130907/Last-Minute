"""G0 smoke test — seeds reproducible, logger emits JSON, device fallback logged."""
from __future__ import annotations

import io
import json
import logging
import random
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from common.bootstrap import get_logger, pick_device, set_all_seeds  # noqa: E402


def _capture_root_stream() -> io.StringIO:
    """Swap the root handler's stream for an in-memory buffer."""
    root = logging.getLogger("mc_gqpso")
    buf = io.StringIO()
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = buf
    return buf


def test_seeds_reproducible() -> None:
    set_all_seeds(42)
    a = (random.random(), np.random.rand(3).tolist())
    set_all_seeds(42)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b, f"seeds not reproducible: {a} vs {b}"


def test_seeds_differ_across_seeds() -> None:
    set_all_seeds(1)
    a = np.random.rand(5).tolist()
    set_all_seeds(2)
    b = np.random.rand(5).tolist()
    assert a != b


def test_logger_emits_json() -> None:
    log = get_logger("smoke")
    buf = _capture_root_stream()
    log.info("iteration tick", extra={"iteration": 3, "best_cost": 12.5, "wall_time_s": 0.01})
    line = buf.getvalue().strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["msg"] == "iteration tick"
    assert rec["iteration"] == 3
    assert rec["best_cost"] == 12.5
    assert rec["wall_time_s"] == 0.01
    assert rec["logger"].startswith("mc_gqpso.")


def test_device_fallback_logged() -> None:
    buf = _capture_root_stream()
    dev = pick_device(use_gpu=True)
    # No CUDA on dev boxes → either mps (Apple Silicon) or cpu; both count as
    # non-cuda fallbacks and MUST log a warning.
    assert dev in ("cpu", "mps"), f"unexpected device: {dev}"
    lines = [json.loads(x) for x in buf.getvalue().strip().splitlines() if x]
    fallback = [r for r in lines if r["level"] == "WARNING" and r.get("requested") == "cuda"]
    assert fallback, f"expected fallback warning; got {lines}"

    buf2 = _capture_root_stream()
    assert pick_device(use_gpu=False) == "cpu"
    lines2 = [json.loads(x) for x in buf2.getvalue().strip().splitlines() if x]
    assert any(r["level"] == "INFO" and r.get("device") == "cpu" for r in lines2)


if __name__ == "__main__":
    test_seeds_reproducible()
    test_seeds_differ_across_seeds()
    test_logger_emits_json()
    test_device_fallback_logged()
    print("bootstrap smoke: OK")
