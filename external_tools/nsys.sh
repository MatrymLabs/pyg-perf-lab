#!/usr/bin/env bash
# Nsight Systems: capture a CUDA timeline (copies, gaps, sync). GPU host only.
# Usage: external_tools/nsys.sh python scripts/profile_step.py --device cuda
# Wire into PyCharm: Settings > Tools > External Tools > program=this script.
set -euo pipefail
command -v nsys >/dev/null 2>&1 || { echo "nsys (Nsight Systems) not installed — GPU host only."; exit 127; }
exec nsys profile --trace=cuda,osrt,nvtx --output "nsys_$(date -u +%Y%m%dT%H%M%SZ)" "$@"
