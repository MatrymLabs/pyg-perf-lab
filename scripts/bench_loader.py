#!/usr/bin/env python3
"""Runnable: benchmark data loading (full-batch vs NeighborLoader vs PrefetchLoader).

    python scripts/bench_loader.py --device cpu --batch-size 512 --steps 20
    python scripts/bench_loader.py --device cuda                  # adds PrefetchLoader

Needs the bench extra: pip install -e ".[bench]"
"""

import sys

from pyg_perf.cli import bench_main

if __name__ == "__main__":
    raise SystemExit(bench_main(sys.argv[1:]))
