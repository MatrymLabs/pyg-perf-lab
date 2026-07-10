#!/usr/bin/env bash
# Nsight Compute: per-kernel efficiency (occupancy, memory throughput). GPU host only.
# Usage: external_tools/ncu.sh python scripts/profile_step.py --device cuda
set -euo pipefail
command -v ncu >/dev/null 2>&1 || { echo "ncu (Nsight Compute) not installed — GPU host only."; exit 127; }
# --set full is expensive; scope to a few steps in the script when using this.
exec ncu --set full --target-processes all -o "ncu_$(date -u +%Y%m%dT%H%M%SZ)" "$@"
