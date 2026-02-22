# Experiment Results — UMI Event Detection Benchmark

## Experiment 1: Ray Batch — SSD I/O + Python Dict Processing

**Setup**: Ray remote tasks, one per episode file. Each task reads from SSD (CSV or Parquet), converts to Python dicts, runs HSM row-by-row. "GPU" mode reserves a GPU slot but processes on CPU with Python dicts — GPU is idle.

**Hardware**: AMD 16c/32t + 2x NVIDIA 3090/3090Ti

**Dataset**: 50 episodes, 19,354 total rows, 23 float columns (low-dim sensor data extracted from UMI cup-in-the-wild Zarr)

**HSM events detected per pass**: 153 (51 grasp, 75 lift, 27 place)

| Strategy | Scale | HW | Rows/sec | Events/sec | p50 (ms) | p99 (ms) | Wall (s) |
|----------|------:|---:|---------:|-----------:|---------:|---------:|---------:|
| large-few | 1x | CPU | 7,012 | 55.4 | 19.9 | 52.7 | 2.8 |
| large-few | 100x | CPU | 331,976 | 2,624.4 | 683.9 | 1,971.5 | 5.8 |
| large-few | 1000x | CPU | 1,452,890 | 11,485.6 | 4,807.7 | 7,056.8 | 13.3 |
| large-few | 10000x | CPU | 1,990,235 | 15,733.5 | 45,871.3 | 69,019.2 | 97.2 |
| large-few | 1x | GPU | 755 | 6.0 | 18.8 | 24.2 | 25.6 |
| large-few | 100x | GPU | 60,176 | 475.7 | 277.9 | 389.6 | 32.2 |
| large-few | 1000x | GPU | 204,444 | 1,616.2 | 2,668.7 | 3,770.2 | 94.7 |
| large-few | 10000x | GPU | 265,472 | 2,098.6 | 27,323.0 | 38,023.6 | 729.0 |
| small-many | 1x | CPU | 6,946 | 54.9 | 23.4 | 90.0 | 2.8 |
| small-many | 100x | CPU | 304,089 | 2,403.9 | 693.2 | 2,048.2 | 6.4 |
| small-many | 1000x | CPU | 1,458,796 | 11,532.3 | 4,808.6 | 7,164.7 | 13.3 |
| small-many | 10000x | CPU | 2,017,271 | 15,947.2 | 44,729.1 | 66,752.7 | 95.9 |
| small-many | 1x | GPU | 749 | 5.9 | 22.4 | 55.7 | 25.8 |
| small-many | 100x | GPU | 59,351 | 469.2 | 286.2 | 408.1 | 32.6 |
| small-many | 1000x | GPU | 204,063 | 1,613.2 | 2,672.1 | 3,936.3 | 94.8 |
| small-many | 10000x | GPU | 266,661 | 2,108.0 | 26,767.3 | 38,439.9 | 725.8 |

### Key Findings

1. **CPU dominates GPU at all scales** — 32 CPU threads process 50 episodes in parallel vs only 2 GPU workers (one per GPU). GPU reserves a slot but doesn't actually compute on GPU.
2. **small-many (CSV) and large-few (Parquet) perform nearly identically** — at this dataset size, file format doesn't matter.
3. **CPU peaks at ~2M rows/sec** at 10000x replay (193M rows in ~96s).
4. **GPU bottleneck is parallelism** — only 2 workers vs 32. Not a GPU compute issue since GPU isn't used for compute at all.
5. **At 10000x**: 193M rows simulates ~18 GB of sensor data throughput.

### Limitation

The "GPU" mode in Experiment 1 is misleading — it reserves GPU hardware but runs pure Python on CPU. Data flows: SSD → CPU RAM (Polars) → Python dicts → HSM. The GPU is completely idle.

---

## Experiment 2: GPU-Resident Tensor Processing

**Setup**: Pre-load all episodes once, pre-compute signals (accel_mag, gripper_delta, z_accel) as vectorized array ops, then feed pre-computed floats to `HSMDetector.step_precomputed()`. Two modes:
- **cpu-tensor**: numpy arrays, vectorized numpy signal computation, 32 CPU Ray workers
- **gpu-tensor**: torch CUDA tensors, vectorized GPU signal computation, transfer to CPU for HSM iteration, 2 GPU-pinned Ray workers

**Hardware**: AMD 16c/32t + 2x NVIDIA 3090/3090Ti

**Dataset**: 50 episodes, 19,354 total rows (same as Exp 1). Replay scales simulate higher data volumes.

**HSM events detected per pass**: 153 (51 grasp, 75 lift, 27 place) — matches Experiment 1

| Mode | Scale | Rows/sec | Events/sec | p50 (ms) | p99 (ms) | Load (s) | Compute (s) | Wall (s) |
|------|------:|---------:|-----------:|---------:|---------:|---------:|------------:|---------:|
| cpu-tensor | 1x | 4,865 | 38.5 | 0.9 | 1.8 | 0.3 | 4.0 | 4.3 |
| cpu-tensor | 100x | 495,187 | 3,914.6 | 91.5 | 144.0 | 0.3 | 3.9 | 4.2 |
| cpu-tensor | 1000x | 3,561,412 | 28,154.2 | 919.0 | 1,295.9 | 0.3 | 5.4 | 5.7 |
| cpu-tensor | 10000x | 8,442,565 | 66,741.4 | 9,361.8 | 13,907.7 | 0.3 | 22.9 | 23.2 |
| gpu-tensor | 1x | 814 | 6.4 | 0.7 | 0.9 | 3.9 | 23.8 | 27.7 |
| gpu-tensor | 100x | 77,528 | 612.9 | 52.2 | 78.7 | 2.5 | 25.0 | 27.4 |
| gpu-tensor | 1000x | 514,598 | 4,068.1 | 519.7 | 738.9 | 2.5 | 37.6 | 40.1 |
| gpu-tensor | 10000x | 1,211,645 | 9,578.5 | 5,238.7 | 7,539.0 | 2.3 | 159.7 | 162.1 |

### Key Findings

1. **cpu-tensor is 4.2x faster than Exp 1 CPU at peak** — 8.4M rows/sec vs 2.0M (Exp 1) at 10000x. Pre-computing signals as numpy vectors + `step_precomputed()` eliminates per-row dict creation and string key scanning overhead.
2. **gpu-tensor is 4.6x faster than Exp 1 GPU** — 1.2M rows/sec vs 265K (Exp 1) at 10000x. GPU computes signals vectorized, but the 2-worker parallelism bottleneck remains.
3. **cpu-tensor still dominates gpu-tensor by ~7x at scale** — the parallelism advantage of 32 CPU workers vs 2 GPU workers overwhelms GPU compute gains. The HSM state machine is inherently sequential per-episode.
4. **Load time is negligible for cpu-tensor** (~0.3s), significant for gpu-tensor (~2.5s due to CUDA init + H2D transfer). At scale, load time is amortized.
5. **Per-episode latency is much lower in Exp 2** — cpu-tensor p50 at 1x is 0.9ms vs 23ms (Exp 1), showing the benefit of skipping SSD I/O + Polars + dict conversion per task.
6. **At 10000x cpu-tensor**: 193M rows processed in 22.9s compute time — equivalent to ~18 GB sensor throughput at 8.4M rows/sec.

### Exp 1 vs Exp 2 Comparison (10000x scale)

| Metric | Exp 1 CPU | Exp 2 cpu-tensor | Exp 1 GPU | Exp 2 gpu-tensor |
|--------|----------:|-----------------:|----------:|-----------------:|
| Rows/sec | 2,017,271 | 8,442,565 | 266,661 | 1,211,645 |
| Wall (s) | 95.9 | 23.2 | 725.8 | 162.1 |
| Speedup | 1.0x | **4.1x** | 1.0x | **4.5x** |

### What the GPU actually does in Experiment 2

Unlike Exp 1 where GPU was completely idle, in Exp 2 gpu-tensor mode the GPU genuinely computes:
- Signal pre-computation (accel magnitude, gripper delta, z-accel) runs as vectorized torch CUDA ops
- Pre-computed signal arrays are transferred back to CPU for sequential HSM iteration
- The bottleneck is still the sequential Python HSM loop — GPU wins on vectorized prep but can't help with the stateful per-timestep logic
