# Sample verified run — CPU (this host)

> Captured 2026-07-10 on the author's host: **aarch64 Linux (Raspberry Pi), no GPU**,
> torch 2.13.0 (CPU), torch-geometric 2.8.0. A dated snapshot, not a live claim —
> reproduce with the commands below. GPU numbers await a run on the CUDA box.

## `python scripts/bench_loader.py --device cpu --num-nodes 5000 --steps 15`

```
# using CPU as requested
# neighbor_loader SKIPPED -- 'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'. Install pyg-lib or torch-sparse.
[
  {
    "label": "full_batch_forward",
    "device": "cpu",
    "runs": 15,
    "mean_ms": 68.847,
    "median_ms": 68.729,
    "min_ms": 47.313,
    "p90_ms": 81.765
  }
]
```

The `neighbor_loader` row is honestly **skipped**: PyG's neighbor sampler needs a
compiled backend (`pyg-lib` or `torch-sparse`), which has no aarch64 wheel for this
torch build. The tool reports that and continues rather than crashing.

## `python scripts/profile_step.py --device cpu --steps 10` (top rows)

```
# using CPU as requested
# torch.profiler device=cpu amp=False compile=False
# trace: trace_cpu.json
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                          ProfilerStep*         1.97%      56.358ms       100.00%        2.867s     358.422ms             8  
                                     aten::scatter_add_        29.93%     858.253ms        29.93%     858.313ms      21.458ms            40  
                                               aten::mm        23.26%     666.948ms        23.26%     667.065ms      10.423ms            64  
                                     aten::index_select        14.89%     426.923ms        14.90%     427.340ms      26.709ms            16  
autograd::engine::evaluate_function: IndexSelectBack...         0.27%       7.684ms        12.03%     344.994ms      43.124ms             8  
       autograd::engine::evaluate_function: MmBackward0         0.03%     787.908us        11.14%     319.438ms      19.965ms            16  
                                            MmBackward0         0.04%       1.209ms        11.11%     318.650ms      19.916ms            16  
    autograd::engine::evaluate_function: AddmmBackward0         0.28%       7.934ms        10.75%     308.141ms      19.259ms            16  
```

**Interpretation:** `aten::scatter_add_` dominates self-CPU time, with
`index_select`/`gather` close behind — the textbook GNN message-passing gather-scatter
cost. On a real workload this is the signal to try `SparseTensor` (see IMPLEMENTATION.md).
