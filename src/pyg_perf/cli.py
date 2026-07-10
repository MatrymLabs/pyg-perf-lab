"""CLI entry points: `pyg-profile` and `pyg-bench` (also runnable as scripts/*.py).

Parse a few knobs into a RunConfig, run, and print. These require the bench extra
(torch + torch_geometric); config/timing logic is unit-tested without them.
"""

from __future__ import annotations

import argparse
import json

from pyg_perf.config import RunConfig


def _parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="PyG performance tooling")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-nodes", type=int, default=10_000)
    p.add_argument("--avg-degree", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--compile", action="store_true", help="torch.compile(dynamic=True)")
    p.add_argument("--seed", type=int, default=0)
    return p


def _config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        device=args.device,
        num_nodes=args.num_nodes,
        avg_degree=args.avg_degree,
        batch_size=args.batch_size,
        steps=args.steps,
        amp=args.amp,
        compile=args.compile,
        seed=args.seed,
    )


def profile_main(argv: list[str] | None = None) -> int:
    args = _parser("pyg-profile").parse_args(argv)
    cfg = _config(args)
    r = cfg.resolved()
    print(f"# {r.note}")
    from pyg_perf.runners import run_profile

    print(run_profile(cfg))
    return 0


def bench_main(argv: list[str] | None = None) -> int:
    args = _parser("pyg-bench").parse_args(argv)
    cfg = _config(args)
    r = cfg.resolved()
    print(f"# {r.note}")
    from pyg_perf.runners import run_loader_bench

    rows = [t.as_row() for t in run_loader_bench(cfg)]
    print(json.dumps(rows, indent=2))
    return 0
