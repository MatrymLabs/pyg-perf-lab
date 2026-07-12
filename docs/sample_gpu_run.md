# Sample verified run - CUDA (RTX 4090)

> Captured 2026-07-12 on Windows 11, **NVIDIA GeForce RTX 4090 (24 GB)**,
> torch 2.11.0+cu128, torch-geometric 2.8.0, pyg-lib 0.7.0+pt211cu128.
> A dated snapshot, not a live claim — reproduce with the commands below.

## `python scripts/bench_loader.py --device cuda --batch-size 512 --steps 20`

```
# using CUDA as requested
[
  {
    "label": "full_batch_forward",
    "device": "cuda",
    "runs": 20,
    "mean_ms": 0.409,
    "median_ms": 0.403,
    "min_ms": 0.358,
    "p90_ms": 0.463
  },
  {
    "label": "neighbor_loader",
    "device": "cuda",
    "runs": 20,
    "mean_ms": 2.234,
    "median_ms": 2.198,
    "min_ms": 1.869,
    "p90_ms": 2.335
  },
  {
    "label": "prefetch_loader",
    "device": "cuda",
    "runs": 20,
    "mean_ms": 4.239,
    "median_ms": 4.016,
    "min_ms": 1.161,
    "p90_ms": 4.573
  }
]
```

**Interpretation:** Full-batch forward on the 4090 is **0.4 ms** — roughly 170x faster
than the Pi's 68 ms CPU baseline. NeighborLoader adds sampling overhead (2.2 ms) which
dominates at this scale. PrefetchLoader's async H2D copy doesn't pay off on this small
synthetic graph (4.0 ms > 2.2 ms) because the pinning + stream overhead exceeds the
transfer savings — on larger real-world graphs with bigger feature tensors, the overlap
would justify the cost.

## `python scripts/profile_step.py --device cuda --amp --steps 12` (top rows)

```
# torch.profiler device=cuda amp=True compile=False
# trace: trace_cuda.json
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                          ProfilerStep*         0.00%       0.000us         0.00%       0.000us       0.000us      35.063ms       686.73%      35.063ms       3.506ms            10
                               Optimizer.step#Adam.step         0.00%       0.000us         0.00%       0.000us       0.000us       2.102ms        41.16%       2.102ms     210.164us            10
                                           aten::gather         0.89%     439.000us         1.70%     839.600us      27.987us       1.483ms        29.04%       1.483ms      49.425us            30
                                     aten::index_select         0.88%     435.600us         2.34%       1.160ms      57.975us       0.000us         0.00%     988.214us      49.411us            20
                                       aten::index_add_         0.17%      82.100us         0.29%     142.500us      14.250us     799.097us        15.65%     799.097us      79.910us            10
                                     aten::scatter_add_         0.93%     461.300us         1.63%     806.800us      20.170us     717.531us        14.05%     717.531us      17.938us            40
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 49.519ms
Self CUDA time total: 5.106ms
```

**Interpretation:** With AMP (float16), self-CUDA time is just **5.1 ms** for 10 profiled
steps — sub-millisecond per step. The gather/scatter ops still dominate CUDA self-time
(~43% combined), consistent with the CPU profile. `torch.compile` was skipped because
Triton is not available on Windows; that optimization can be verified on Linux.

## Chrome trace

The profiler exported `trace_cuda.json` (3.1 MB) — load it in `chrome://tracing` or
PyCharm's trace viewer for flame-graph analysis of the CUDA kernel timeline.
