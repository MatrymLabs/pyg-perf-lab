#!/usr/bin/env python3
"""Runnable: profile GNN training steps with torch.profiler.

    python scripts/profile_step.py --device cpu --steps 12
    python scripts/profile_step.py --device cuda --amp --compile   # on a GPU host

Needs the bench extra: pip install -e ".[bench]"
"""

import sys

from pyg_perf.cli import profile_main

if __name__ == "__main__":
    raise SystemExit(profile_main(sys.argv[1:]))
