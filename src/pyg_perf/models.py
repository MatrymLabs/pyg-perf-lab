"""A small, representative GNN (GraphSAGE) -- the benchmark subject.

Kept deliberately simple: two SAGE layers. The point of this package is measuring and
tuning the *pipeline* (loaders, transfers, allocator, AMP, compile), not chasing accuracy.
Torch/PyG imported lazily; `build_model` needs the `bench` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyg_perf.config import RunConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


def build_model(cfg: RunConfig) -> torch.nn.Module:
    """A 2-layer GraphSAGE sized from the config."""
    try:
        import torch
        from torch_geometric.nn import SAGEConv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'build_model needs torch + torch_geometric -- pip install -e ".[bench]"'
        ) from exc

    class SAGE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = SAGEConv(cfg.num_features, cfg.hidden)
            self.conv2 = SAGEConv(cfg.hidden, cfg.num_classes)

        def forward(self, x: Any, edge_index: Any) -> Any:
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)

    return SAGE()
