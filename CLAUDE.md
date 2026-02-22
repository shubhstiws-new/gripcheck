# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

5D benchmark comparing **Ray vs PyFlink** for real-time event detection on Stanford UMI robotics data. The goal is to answer: "For this robotics scenario + 3 critical events → use Ray or Flink?" The 5 dimensions swept are: hardware, framework, dataset strategy, operation mode (batch/streaming), and replay scale (1× / 100× / 1000× / 10000×).

## Commands

### Setup and dependency management

```bash
# Install dependencies (uses uv, Python 3.11)
uv sync

# Run any script
uv run python <script>.py
```

### Pipeline execution (order matters)

```bash
# Step 1: Extract Zarr → two dataset strategies
uv run python data_gen.py
# Writes: data/small_many/episode_NNNN.csv, data/large_few/episode_NNNN.parquet

# Step 2: Run Ray batch baseline
uv run python ray_batch.py --strategy small-many
uv run python ray_batch.py --strategy large-few --gpu --replay-scale 100
uv run python ray_batch.py --strategy both   # runs both strategies

# Start Ray head node (needed for multi-GPU)
ray start --head --num-gpus=2
```

### ray_batch.py CLI options

| Flag | Values | Default |
|------|--------|---------|
| `--strategy` | `small-many`, `large-few`, `both` | `both` |
| `--gpu` | flag | off |
| `--replay-scale` | `1`, `100`, `1000`, `10000` | `1` |

Results are saved to `results/ray_batch_<strategy>_x<scale>[_gpu].json`.

## Architecture

### Core files (the actual benchmark)

- **`hsm.py`** — Shared Hierarchical State Machine. Framework-agnostic, row-by-row stateful event detector. `HSMDetector.step(row_dict)` returns a list of `Event` dataclasses. Used identically by Ray and (future) Flink pipelines. Detects 3 events: `grasp`, `lift`, `place`.
- **`data_gen.py`** — Reads the UMI Zarr replay buffer (`data/raw/cup_in_the_wild.zarr`), auto-discovers low-dim sensor arrays (skipping camera/image arrays), and writes two output strategies. Synthetic acceleration is derived via finite-diff of EEF position when IMU is absent.
- **`ray_batch.py`** — Ray parallel episode processing. Dispatches one Ray remote task per episode file. `GPUHSMActor`/`CPUHSMActor` and `process_episode_remote_gpu`/`process_episode_remote` handle GPU vs CPU routing. `BenchResult` collects rows/sec, events/sec, p50/p99 latency, CPU/mem.

### Data flow

```
Zarr replay buffer (cup_in_the_wild.zarr)
  └─ data_gen.py
       ├─ data/small_many/*.csv       (per-episode, ~1–10 MB each)
       └─ data/large_few/*.parquet    (per-episode, with episode_id col)
            └─ ray_batch.py / future flink_pipeline.py
                 └─ hsm.py (HSMDetector per episode)
                      └─ results/*.json
```

### Zarr structure

- `meta/episode_ends` — cumulative end indices (one int per episode)
- `data/*` — flat arrays across all timesteps (e.g., `robot_eef_pos`, `robot_gripper_width`, `camera0_rgb`)
- Low-dim keys (1D or 2D with ≤10 cols) are selected; 3D+ image arrays are skipped

### Planned files (not yet created)

Per `OVERALL_PLAN.md` / `NEXT_STEPS.md`:
- `experiments/sweep.py` — Typer CLI sweeping all 4 combos per strategy (Week 2)
- `flink_pipeline.py` — Identical HSM logic via PyFlink `KeyedProcessFunction` (Week 3)
- `analysis.ipynb` — Load `results.parquet`, heatmaps, export `rules.json` (Week 4)

### `universal_manipulation_interface/`

Cloned from `real-stanford/universal_manipulation_interface`. Contains diffusion policy training code, SLAM pipeline scripts, and calibration utilities for the physical robot. This is **reference/data-source material**, not part of the benchmark itself. Don't modify it.

## Key conventions

- `HSMDetector` is episode-scoped; instantiate one per episode, never share across episodes.
- `replay_scale` simulates higher-rate streaming by replaying each episode N times — episode IDs are offset by `rep * 10000` to keep them unique.
- `HSMConfig` thresholds were calibrated for the UMI cup-in-the-wild dataset (EEF accel derived via finite-diff, not raw IMU).
- Results accumulate in `results/` as individual JSON files per run; a future `analysis.ipynb` will merge them into `results.parquet`.
