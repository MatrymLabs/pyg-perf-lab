"""Skip the torch/PyG bench tests when the bench extra (torch) is not installed.

The fast CI lanes deliberately install only `[dev]` (no torch, no heavy path), so these files --
which exercise `graphs`/`models`/`runners`/`cli` on a CPU torch -- are collected only where torch
is present (local dev, forge-audit's env, a bench run). The torch-free `config`/`timing` tests
always run. This keeps the fast gates torch-free by design while the coverage they measure with
torch present reaches ~80%.
"""

import importlib.util

if importlib.util.find_spec("torch") is None:
    collect_ignore_glob = [
        "test_graphs.py",
        "test_models.py",
        "test_cli.py",
        "test_runners.py",
    ]
