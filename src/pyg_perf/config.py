"""Run configuration + honest device resolution.

Torch-independent by design: this module imports without torch installed, so the config
logic is testable on any host (including a GPU-less CI box). Device resolution never
*pretends* CUDA is present -- if you ask for cuda and it is not available, it says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def cuda_available() -> bool:
    """True only if torch is installed AND reports a usable CUDA device. Never guesses."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@dataclass(frozen=True)
class Resolved:
    """The device the run will actually use, and an honest note if it differs from asked."""

    device: str  # "cpu" | "cuda"
    note: str


def resolve_device(prefer: str = "auto") -> Resolved:
    """Resolve a device honestly. `auto` picks cuda when present else cpu; an explicit
    `cuda` request on a host without CUDA falls back to cpu WITH a loud note (never silent)."""
    prefer = prefer.lower()
    has_cuda = cuda_available()
    if prefer in ("cuda", "gpu"):
        if has_cuda:
            return Resolved("cuda", "using CUDA as requested")
        return Resolved("cpu", "CUDA requested but not available on this host -- fell back to CPU")
    if prefer == "cpu":
        return Resolved("cpu", "using CPU as requested")
    # auto
    return Resolved("cuda" if has_cuda else "cpu", "auto-selected")


@dataclass(frozen=True)
class RunConfig:
    """Every knob the benchmarks/profilers expose, with safe CPU-friendly defaults."""

    device: str = "auto"  # auto | cpu | cuda
    num_nodes: int = 10_000
    avg_degree: int = 10
    num_features: int = 64
    hidden: int = 128
    num_classes: int = 8
    batch_size: int = 512
    num_neighbors: tuple[int, ...] = (10, 10)  # NeighborLoader fan-out per hop
    epochs: int = 2
    steps: int = 20
    amp: bool = False  # mixed precision (CUDA only; ignored on CPU)
    compile: bool = False  # torch.compile(dynamic=True)
    seed: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def resolved(self) -> Resolved:
        return resolve_device(self.device)
