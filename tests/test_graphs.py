"""Tests for the deterministic synthetic-graph builder (CPU, no dataset download)."""

from __future__ import annotations

import torch

from pyg_perf.config import RunConfig
from pyg_perf.graphs import make_graph


def _small() -> RunConfig:
    return RunConfig(num_nodes=20, avg_degree=3, num_features=4, num_classes=2, seed=7)


def test_make_graph_has_the_configured_shapes() -> None:
    g = make_graph(_small())
    assert g.x.shape == (20, 4)  # num_nodes x num_features
    assert g.y.shape == (20,)
    assert g.edge_index.shape == (2, 20 * 3)  # num_nodes * avg_degree edges
    assert g.train_mask.sum().item() == 16  # 0.8 * num_nodes


def test_make_graph_is_reproducible_for_a_given_seed() -> None:
    a, b = make_graph(_small()), make_graph(_small())
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_index, b.edge_index)
    assert torch.equal(a.y, b.y)


def test_a_different_seed_changes_the_graph() -> None:
    a = make_graph(RunConfig(num_nodes=20, avg_degree=3, num_features=4, seed=1))
    b = make_graph(RunConfig(num_nodes=20, avg_degree=3, num_features=4, seed=2))
    assert not torch.equal(a.x, b.x)
