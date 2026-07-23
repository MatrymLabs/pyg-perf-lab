"""CPU smoke tests for the benchmark runners.

The loader benchmark and a single train step run fast on CPU and are exercised here. The full
`run_profile` path wraps the torch profiler (seconds per run, GPU-oriented) and is left to real
hardware runs -- so those lines stay honestly uncovered rather than paying a slow test for them.
"""

from __future__ import annotations

import torch

from pyg_perf.config import RunConfig
from pyg_perf.graphs import make_graph
from pyg_perf.models import build_model
from pyg_perf.runners import run_loader_bench, train_step


def _tiny() -> RunConfig:
    return RunConfig(
        device="cpu",
        num_nodes=40,
        avg_degree=3,
        num_features=4,
        hidden=8,
        num_classes=2,
        steps=2,
        epochs=1,
        batch_size=16,
        num_neighbors=(3, 3),
        seed=1,
    )


def test_train_step_returns_a_finite_loss_on_cpu() -> None:
    cfg = _tiny()
    g = make_graph(cfg)
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = train_step(model, g, optimizer, "cpu", amp=False)
    assert isinstance(loss, float)
    assert loss >= 0.0 and loss == loss  # finite (not NaN)


def test_run_loader_bench_yields_timings_and_notes_on_cpu() -> None:
    timings, notes = run_loader_bench(_tiny())
    assert len(timings) >= 1
    assert all(hasattr(t, "as_row") for t in timings)  # rows serialize to the JSON table
    assert isinstance(notes, list)
