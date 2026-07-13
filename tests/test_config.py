"""Tests for config + honest device resolution (no torch required)."""

from __future__ import annotations

import pytest

from pyg_perf.config import RunConfig, cuda_available, resolve_device


def test_cpu_request_is_honored() -> None:
    r = resolve_device("cpu")
    assert r.device == "cpu"


def test_cuda_request_without_a_gpu_falls_back_loudly() -> None:
    # On a CPU-only host, asking for cuda must fall back to cpu WITH a note -- never silent,
    # never a pretend "cuda". (On a real GPU host this test still holds: it only asserts the
    # fallback branch, which is what CI without a GPU exercises.)
    if cuda_available():
        assert resolve_device("cuda").device == "cuda"
    else:
        r = resolve_device("cuda")
        assert r.device == "cpu"
        assert "not available" in r.note


def test_auto_matches_actual_availability() -> None:
    r = resolve_device("auto")
    assert r.device == ("cuda" if cuda_available() else "cpu")


def test_unknown_device_fails_loud() -> None:
    # A typo like "cudaa" must not silently auto-select. The module's promise is
    # "never pretends" -- an unrecognized request raises, it does not guess.
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("cudaa")


def test_config_defaults_are_cpu_friendly() -> None:
    cfg = RunConfig()
    assert cfg.device == "auto"
    assert cfg.amp is False and cfg.compile is False  # opt-in, not default
    assert cfg.resolved().device in ("cpu", "cuda")
