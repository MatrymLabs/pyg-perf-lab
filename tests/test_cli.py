"""Tests for the CLI arg parsing + entry dispatch (runners mocked -- no hardware needed)."""

from __future__ import annotations

import pytest

from pyg_perf import cli
from pyg_perf.config import RunConfig


def test_parser_has_sensible_defaults() -> None:
    args = cli._parser("t").parse_args([])
    assert args.device == "auto"
    assert args.num_nodes == 10_000
    assert args.seed == 0
    assert args.amp is False


def test_config_maps_parsed_args_onto_a_runconfig() -> None:
    args = cli._parser("t").parse_args(
        ["--device", "cpu", "--num-nodes", "50", "--steps", "3", "--amp", "--compile"]
    )
    cfg = cli._config(args)
    assert isinstance(cfg, RunConfig)
    assert cfg.device == "cpu" and cfg.num_nodes == 50 and cfg.steps == 3
    assert cfg.amp is True and cfg.compile is True


def test_profile_main_prints_the_runner_result(monkeypatch, capsys) -> None:
    import pyg_perf.runners as runners

    monkeypatch.setattr(runners, "run_profile", lambda cfg: "PROFILE-OUTPUT")
    assert cli.profile_main(["--device", "cpu", "--num-nodes", "20"]) == 0
    assert "PROFILE-OUTPUT" in capsys.readouterr().out


def test_bench_main_prints_notes_and_a_json_table(monkeypatch, capsys) -> None:
    import pyg_perf.runners as runners

    # empty timings -> a "[]" table; a note line -> printed with a "# " prefix
    monkeypatch.setattr(runners, "run_loader_bench", lambda cfg: ([], ["cpu run, no GPU"]))
    assert cli.bench_main(["--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "cpu run, no GPU" in out and "[]" in out


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_config_resolves_a_device_without_hardware(device) -> None:
    args = cli._parser("t").parse_args(["--device", device])
    cfg = cli._config(args)
    assert cfg.resolved().device in ("cpu", "cuda")  # resolves; CPU on this host
