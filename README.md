# pyg-perf-lab

**A PyCharm + PyTorch Geometric performance-engineering package** - runnable profilers and
benchmarks for GNN training, with PyCharm run-configs, Nsight wrappers, CI (GitHub + GitLab),
and a layered tuning guide. Built to a strict rule: **nothing is claimed as measured that was
not actually run.**

> **Verification status (honest by design).**
> - **CPU paths - run and verified** on the author's host (aarch64 Linux, no GPU): the
>   config/timing logic, the `full_batch_forward` benchmark, and `torch.profiler` on CPU.
>   See the dated snapshot in [docs/sample_cpu_run.md](docs/sample_cpu_run.md) - the
>   profiler measured `aten::scatter_add_` as the dominant cost (the classic GNN
>   gather-scatter hotspot).
> - **Neighbor sampling** (`NeighborLoader`) needs a compiled backend (`pyg-lib` or
>   `torch-sparse`); when it's absent the tool **reports and skips** that row rather than
>   crashing - honest degradation, not a silent gap.
> - **GPU / CUDA / Nsight paths - authored, not yet verified here** (no GPU on this host).
>   Written from the official docs to run on a CUDA box and marked *(verify on a CUDA
>   host)* throughout. See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## Install

The base install is light (no torch - so lint/type/tests run anywhere):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # or: make env
make check                  # ruff + mypy + pytest (no torch needed)
```

To run the benchmarks, install a **torch build for your platform** first
(see https://pytorch.org/get-started/), then:

```bash
pip install -e ".[bench]"   # torch + torch-geometric
```

## Run

```bash
# CPU (runs anywhere)
python scripts/bench_loader.py --device cpu --num-nodes 5000 --steps 15
python scripts/profile_step.py --device cpu --steps 12

# GPU host - verify on your CUDA box
python scripts/bench_loader.py --device cuda            # adds PrefetchLoader
python scripts/profile_step.py --device cuda --amp --compile
```

## What's inside

| Path | What |
|------|------|
| `src/pyg_perf/config.py` | Run config + **honest device resolution** (asks cuda, reports fallback - never pretends) |
| `src/pyg_perf/timing.py` | Timing harness with **CUDA sync** (no async mis-measurement) |
| `src/pyg_perf/{graphs,models,runners}.py` | Synthetic graph · GraphSAGE · `torch.profiler` + `NeighborLoader`/`PrefetchLoader` + AMP + `torch.compile` |
| `scripts/{profile_step,bench_loader}.py` | Runnable entry points |
| `.idea/runConfigurations/` | PyCharm one-click run configs (CPU + GPU) |
| `external_tools/{nsys,ncu}.sh` | Nsight Systems / Compute wrappers *(GPU host)* |
| `.github/workflows/ci.yml`, `.gitlab-ci.yml` | CPU matrix (runs) + documented self-hosted GPU job |
| `docs/IMPLEMENTATION.md` | The layered method + the five common PyG slowdowns + sources |

## Method (one line)

Shorten the loop in PyCharm → **measure before guessing** (`torch.profiler`) → fix data
loading and batching before kernel tuning → escalate to Nsight only when the evidence says
you're bottlenecked below Python. Full guide in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

MIT.
