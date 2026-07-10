"""pyg-perf-lab -- a PyCharm + PyTorch Geometric performance-engineering package.

Profile, benchmark, and tune GNN training on CPU/GPU, with honest evidence: the CPU paths
run on any host; the CUDA-only paths (AMP, PrefetchLoader, Nsight) light up on a GPU box.
Nothing is claimed as measured that was not actually run -- see docs/IMPLEMENTATION.md.
"""

from pyg_perf.config import RunConfig, resolve_device

__all__ = ["RunConfig", "resolve_device"]
__version__ = "0.1.0"
