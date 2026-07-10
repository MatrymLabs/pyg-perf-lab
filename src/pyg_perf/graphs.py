"""Deterministic synthetic graphs -- so benchmarks run anywhere, no dataset download.

Torch/PyG are imported lazily inside the function: the module loads without them, but
`make_graph()` requires the `bench` extra (`pip install -e ".[bench]"`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyg_perf.config import RunConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch_geometric.data import Data


def make_graph(cfg: RunConfig) -> Data:
    """Build a reproducible random graph (`num_nodes`, ~`avg_degree` edges/node) with node
    features and labels, as a PyG `Data` object on CPU. Seeded for repeatable benchmarks."""
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - exercised only without the bench extra
        raise RuntimeError(
            "make_graph needs torch + torch_geometric -- install the bench extra: "
            'pip install -e ".[bench]"'
        ) from exc

    g = torch.Generator().manual_seed(cfg.seed)
    n, d = cfg.num_nodes, cfg.avg_degree
    num_edges = n * d
    # Random directed edges; PyG handles the rest. Deterministic given the seed.
    edge_index = torch.randint(0, n, (2, num_edges), generator=g)
    x = torch.randn(n, cfg.num_features, generator=g)
    y = torch.randint(0, cfg.num_classes, (n,), generator=g)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[: int(0.8 * n)] = True
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
