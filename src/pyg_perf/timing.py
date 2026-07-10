"""A timing harness that is correct on both CPU and GPU.

The classic GPU-benchmark bug is timing async CUDA work without synchronizing -- you
measure launch latency, not compute. `sync(device)` calls `torch.cuda.synchronize()` on
CUDA and is a no-op on CPU, so `time_calls()` reports real wall time on either device.

Torch is imported lazily so this module (and its tests) load without torch installed.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass


def sync(device: str) -> None:
    """Block until queued device work finishes -- required before/after timing CUDA."""
    if device == "cuda":
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


@dataclass(frozen=True)
class Timing:
    """Summary of repeated timings, in milliseconds."""

    label: str
    device: str
    runs: int
    mean_ms: float
    median_ms: float
    min_ms: float
    p90_ms: float

    def as_row(self) -> dict[str, object]:
        return {
            "label": self.label,
            "device": self.device,
            "runs": self.runs,
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
        }


def summarize(label: str, device: str, samples_ms: list[float]) -> Timing:
    """Turn a list of per-call millisecond samples into a Timing summary."""
    if not samples_ms:
        raise ValueError("no timing samples")
    ordered = sorted(samples_ms)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return Timing(
        label=label,
        device=device,
        runs=len(samples_ms),
        mean_ms=statistics.fmean(samples_ms),
        median_ms=statistics.median(samples_ms),
        min_ms=ordered[0],
        p90_ms=p90,
    )


def time_calls(
    fn: Callable[[], object],
    device: str,
    *,
    warmup: int = 3,
    runs: int = 20,
    label: str = "call",
) -> Timing:
    """Time `fn` `runs` times after `warmup` untimed iterations, synchronizing the device
    around each measured call so CUDA async execution is not mis-measured."""
    for _ in range(max(0, warmup)):
        fn()
    sync(device)
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return summarize(label, device, samples)
