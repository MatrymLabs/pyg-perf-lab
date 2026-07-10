"""Tests for the timing harness (no torch required for the summary math)."""

from __future__ import annotations

import pytest

from pyg_perf.timing import summarize, sync, time_calls


def test_summarize_computes_stats() -> None:
    t = summarize("x", "cpu", [10.0, 20.0, 30.0, 40.0])
    assert t.runs == 4
    assert t.min_ms == 10.0
    assert t.mean_ms == 25.0
    assert t.median_ms == 25.0
    assert t.p90_ms in (30.0, 40.0)  # p90 of 4 samples


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no timing"):
        summarize("x", "cpu", [])


def test_sync_is_a_noop_on_cpu() -> None:
    # Must not import torch or raise when device is cpu.
    sync("cpu")


def test_time_calls_measures_a_pure_python_callable() -> None:
    calls = {"n": 0}

    def work() -> None:
        calls["n"] += 1
        sum(range(1000))

    t = time_calls(work, "cpu", warmup=2, runs=5, label="work")
    assert t.runs == 5
    assert calls["n"] == 7  # 2 warmup + 5 measured
    assert t.mean_ms >= 0.0


def test_as_row_is_json_ready() -> None:
    row = summarize("neighbor_loader", "cpu", [1.0, 2.0]).as_row()
    assert row["label"] == "neighbor_loader"
    assert set(row) == {"label", "device", "runs", "mean_ms", "median_ms", "min_ms", "p90_ms"}
