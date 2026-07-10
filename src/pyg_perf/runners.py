"""The runnable core: profile a training step, and benchmark data loading.

Everything device-aware. On CPU (this Pi) the profiler and NeighborLoader benchmark run
for real; the CUDA-only paths (AMP, PrefetchLoader(device=cuda), Nsight) light up on a GPU
host. Torch/PyG imported lazily -- run these with the `bench` extra installed.

References (official docs):
- torch.profiler: https://pytorch.org/docs/stable/profiler.html
- PyG NeighborLoader: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.loader.NeighborLoader.html
- PrefetchLoader: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.loader.PrefetchLoader.html
- torch.compile: https://pytorch.org/docs/stable/generated/torch.compile.html
"""

from __future__ import annotations

from typing import Any

from pyg_perf.config import RunConfig
from pyg_perf.graphs import make_graph
from pyg_perf.models import build_model
from pyg_perf.timing import Timing, time_calls


def _prepare(cfg: RunConfig) -> tuple[Any, Any, Any, str]:
    """Build (data, model, optimizer, device); apply compile if requested. Needs bench extra."""
    import torch

    device = cfg.resolved().device
    torch.manual_seed(cfg.seed)
    data = make_graph(cfg).to(device)
    model = build_model(cfg).to(device)
    if cfg.compile:
        # dynamic=True: graph sizes vary batch to batch under neighbor sampling.
        model = torch.compile(model, dynamic=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return data, model, optimizer, device


def train_step(model: Any, data: Any, optimizer: Any, device: str, amp: bool) -> float:
    """One AMP-aware training step over `data`. AMP is honored only on CUDA."""
    import torch
    import torch.nn.functional as F

    model.train()
    optimizer.zero_grad(set_to_none=True)
    use_amp = amp and device == "cuda"
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else _nullctx()
    with ctx:
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _nullctx() -> Any:
    from contextlib import nullcontext

    return nullcontext()


def run_profile(cfg: RunConfig) -> str:
    """Profile `cfg.steps` training steps with torch.profiler; return the key-averages table.
    Also writes a Chrome trace next to the run for the PyCharm/Chrome trace viewer."""
    from torch.profiler import ProfilerActivity, profile, schedule

    data, model, optimizer, device = _prepare(cfg)
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    sched = schedule(wait=1, warmup=1, active=max(1, cfg.steps - 2), repeat=1)
    with profile(activities=activities, schedule=sched, record_shapes=True) as prof:
        for _ in range(cfg.steps):
            train_step(model, data, optimizer, device, cfg.amp)
            prof.step()

    trace = f"trace_{device}.json"
    try:
        prof.export_chrome_trace(trace)
    except Exception:  # pragma: no cover - trace export is best-effort
        trace = "(trace export skipped)"

    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=15)
    header = (
        f"# torch.profiler device={device} amp={cfg.amp} compile={cfg.compile}\n# trace: {trace}\n"
    )
    return header + table


def run_loader_bench(cfg: RunConfig) -> list[Timing]:
    """Benchmark data loading: full-batch forward vs a NeighborLoader mini-batch loop
    (and PrefetchLoader on CUDA). Returns Timing rows for comparison."""
    import torch
    from torch_geometric.loader import NeighborLoader

    data, model, _opt, device = _prepare(cfg)
    results: list[Timing] = []

    # 1) Full-batch forward (the naive baseline).
    def full_batch() -> None:
        with torch.no_grad():
            model(data.x, data.edge_index)

    results.append(time_calls(full_batch, device, runs=cfg.steps, label="full_batch_forward"))

    # 2) NeighborLoader mini-batches (the scalable path).
    loader = NeighborLoader(
        data,
        num_neighbors=list(cfg.num_neighbors),
        batch_size=cfg.batch_size,
        input_nodes=data.train_mask,
    )
    it = iter(loader)

    def neighbor_step() -> None:
        nonlocal it
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.to(device)
        with torch.no_grad():
            model(batch.x, batch.edge_index)

    results.append(time_calls(neighbor_step, device, runs=cfg.steps, label="neighbor_loader"))

    # 3) PrefetchLoader overlaps host->device copy with compute -- CUDA only.
    if device == "cuda":
        from torch_geometric.loader import PrefetchLoader

        ploader = PrefetchLoader(
            NeighborLoader(
                data,
                num_neighbors=list(cfg.num_neighbors),
                batch_size=cfg.batch_size,
                input_nodes=data.train_mask,
            ),
            device=torch.device("cuda"),
        )
        pit = iter(ploader)

        def prefetch_step() -> None:
            nonlocal pit
            try:
                batch = next(pit)
            except StopIteration:
                pit = iter(ploader)
                batch = next(pit)
            with torch.no_grad():
                model(batch.x, batch.edge_index)

        results.append(time_calls(prefetch_step, device, runs=cfg.steps, label="prefetch_loader"))

    return results
