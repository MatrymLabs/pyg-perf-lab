"""Tests for the benchmark subject (a 2-layer GraphSAGE), exercised on CPU."""

from __future__ import annotations

import torch

from pyg_perf.config import RunConfig
from pyg_perf.graphs import make_graph
from pyg_perf.models import build_model


def test_build_model_forward_maps_nodes_to_class_logits() -> None:
    cfg = RunConfig(num_nodes=20, avg_degree=3, num_features=4, hidden=8, num_classes=3, seed=1)
    g = make_graph(cfg)
    model = build_model(cfg)
    out = model(g.x, g.edge_index)
    assert out.shape == (20, 3)  # num_nodes x num_classes
    assert out.dtype == torch.float32


def test_build_model_is_a_torch_module_with_two_sage_layers() -> None:
    cfg = RunConfig(num_features=4, hidden=8, num_classes=2)
    model = build_model(cfg)
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "conv1") and hasattr(model, "conv2")
