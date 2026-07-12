# GPU scaling study - loader strategy vs graph and batch size

> Captured 2026-07-12 on Windows 11, NVIDIA GeForce RTX 4090 (24 GB), torch 2.11.0+cu128,
> torch-geometric 2.8.0, pyg-lib 0.7.0. Synthetic seeded graphs (64 features, avg degree 10),
> GraphSAGE 2-layer, `no_grad` forward, 20 timed steps per cell, medians in ms.
> A dated snapshot - reproduce with the commands below.

## Sweep 1: graph size (batch_size=512)

`python scripts/bench_loader.py --device cuda --num-nodes <N> --batch-size 512 --steps 20`

| nodes | full_batch_forward | neighbor_loader | prefetch_loader |
|---|---|---|---|
| 5,000 | 1.0 | 3.1 | 3.7 |
| 50,000 | 1.8 | 5.4 | 7.2 |
| 200,000 | 10.4 | 6.1 | 8.7 |
| 500,000 | 26.8 | 7.0 | 10.1 |
| 1,000,000 | 56.0 | 7.7 | 10.6 |

**Finding 1 - the full-batch/sampling crossover sits near ~10^5 nodes on this GPU.**
Below ~100k nodes, just running the whole graph through the model is fastest (the 4090
eats it whole). Above it, full-batch grows linearly with graph size while NeighborLoader
stays nearly flat (its cost tracks batch work, not graph size): 7.2x faster at 1M nodes.

## Sweep 2: batch size (num_nodes=500,000)

`python scripts/bench_loader.py --device cuda --num-nodes 500000 --batch-size <B> --steps 20`

| batch | full_batch_forward | neighbor_loader | prefetch_loader |
|---|---|---|---|
| 512 | 28.0 | 6.8 | 10.7 |
| 2,048 | 28.0 | 19.9 | 26.4 |
| 8,192 | 24.3 | 59.0 | 78.3 |
| 32,768 | 25.4 | 165.3 | 199.8 |

**Finding 2 - CPU-side neighbor sampling is the wall.** Sampled mini-batch cost grows
superlinearly with batch size (6.8 -> 165 ms) while full-batch stays flat (~25 ms): past
batch ~4-8k on this graph, sampling costs more than not sampling. The sampler runs on
CPU; the GPU is idle waiting for it. On a real workload this is the signal for
`pyg-lib`-accelerated sampling, more loader workers, or larger-than-memory reasons to
sample at all.

**Finding 3 (methodological) - PrefetchLoader never wins in this harness, and cannot.**
Prefetch overlaps host-to-device copy with compute, but this benchmark times the loader
serially (`next(it)` then a forward, one step at a time), so there is nothing to overlap
- the per-batch `pin_memory` cost is pure overhead, and it grows with the payload
(+3.9 ms at batch 512, +34 ms at batch 32k). A serial timing harness structurally cannot
show prefetch's benefit; demonstrating it requires an overlapped training loop (long
compute per step, workers feeding batches concurrently). Filed as an honest limitation
of `run_loader_bench`, not a verdict on PrefetchLoader.

## Practical reading (this hardware, this model family)

- Graph fits in VRAM and < ~100k nodes: **full-batch**, skip the loader entirely.
- Larger graphs, small batches (<= ~2k): **NeighborLoader**, and keep batches small.
- Bigger batches wanted: profile the sampler first - it, not the GPU, is the bottleneck.
- PrefetchLoader: only with a real overlapped training loop; benchmark it there.
