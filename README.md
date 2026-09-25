# GripCheck

**Quality scoring for robot manipulation demonstrations, built on the Stanford UMI dataset.**

Imitation-learning policies are only as good as the demonstrations they are trained on. Teleoperated datasets routinely contain aborted attempts, missed grasps, and episodes in which nothing happens. GripCheck screens each episode before training. It detects the manipulation events that should be present (grasp, lift, place), checks basic motion sanity, and writes a per-episode quality table that a training job can filter on.

```python
episodes = pl.read_parquet("results/episodes_scored.parquet")
train_ids = episodes.filter(pl.col("quality_score") >= 0.6)["episode_id"]
```

---

## Pipeline

```
UMI replay buffer (Zarr)
   │  data_gen.py   extract low-dimensional signals per episode; derive acceleration
   ▼
per-episode Parquet / CSV
   │  hsm.py        state machine: IDLE → GRASPING → LIFTING → TRANSPORTING → PLACING
   ▼
event sequence per episode
   │  scorer.py     pattern score + sanity checks → one row per episode
   ▼
episodes_scored.parquet
```

| Stage | Detail |
|---|---|
| Extraction | Reads the flat Zarr arrays and slices them by `meta/episode_ends`. Skips camera frames and keeps only low-dimensional signals (end-effector pose, gripper width). Derives acceleration by double finite difference of position. |
| Event detection | Per-episode stateful detector. **Grasp**: acceleration spike while the gripper closes. **Lift**: upward acceleration with a stable grip. **Place**: downward acceleration while the gripper opens. The state machine suppresses out-of-order detections. Thresholds are in `HSMConfig`. |
| Scoring | Complete grasp → lift → place cycle = 1.0; grasp + lift = 0.6; grasp only = 0.3; none = 0.0. Episodes that are shorter than 50 steps, show under 1 cm of gripper movement, or show under 5 cm of end-effector travel score 0 and carry a named flag. |
| Output | One row per episode: event counts and sequence, completed cycles, score, flags, gripper range, travel distance, acceleration statistics, and processing time. |
| Parallel execution | `ray_batch.py` distributes episodes across Ray tasks and reports throughput and p50/p99 per-episode latency for two storage layouts (many small CSVs, fewer Parquet files). `ray_batch_gpu_resident.py` precomputes signals as NumPy or CUDA tensors and uses a faster state-machine entry point. |

## Performance results

Measured on one workstation (AMD 16-core / 32-thread CPU, 2× RTX 3090) with 50 episodes (19,354 rows). The episodes were replayed up to 10,000× (193M rows) to measure throughput at volume. Full tables: [`results/EXPERIMENT_RESULTS.md`](results/EXPERIMENT_RESULTS.md).

| Approach (10,000× replay) | Workers | Rows / s | Wall time |
|---|---:|---:|---:|
| Row-by-row, Python dicts, CPU | 32 | 2.0M | 96 s |
| Row-by-row, Python dicts, "GPU" actors (GPU idle) | 2 | 0.27M | 726 s |
| Precomputed signals, NumPy, CPU | 32 | **8.4M** | **23 s** |
| Precomputed signals, CUDA tensors | 2 | 1.2M | 162 s |

1. **Removing per-row Python overhead mattered more than hardware.** Vectorising the signal computation and passing plain floats to the state machine gave a 4.1× speedup on the same CPU.
2. **The GPU did not help this workload.** The state machine is sequential within an episode, so the bottleneck is worker count (32 CPU workers vs. 2 GPU workers), not arithmetic. The GPU accelerates only the signal precomputation.
3. **Storage layout was not a factor at this size.** CSV and Parquet layouts performed within 10% of each other at every scale.
4. **Scale reference.** 8.4M rows/s would process the full 699K-step dataset in under 0.1 s of compute. A 50,000-episode corpus (about 25M rows) would take about 3 s on one workstation.

## Dataset

[Universal Manipulation Interface (UMI)](https://umi-gripper.github.io/), `cup_in_the_wild`: 1,447 human demonstrations of cup pick-and-place, about 699K timesteps at about 10 Hz. Only about 87 MB of the dataset is low-dimensional sensor data. Camera frames make up most of the 19 GB archive and are not read.

```bash
wget https://real.stanford.edu/umi/data/zarr_datasets/cup_in_the_wild.zarr.zip
unzip cup_in_the_wild.zarr.zip -d data/raw/
```

## Getting started

```bash
uv sync
uv run pytest                               # unit tests (synthetic episodes, no dataset needed)

uv run python data_gen.py                   # Zarr → data/small_many, data/large_few
uv run python scorer.py                     # → results/episodes_scored.parquet
uv run python ray_batch.py --strategy both  # throughput and latency baseline
```

## Repository layout

```
data_gen.py        Zarr extraction and signal derivation
hsm.py             Event detection state machine and thresholds
scorer.py          Episode scoring and output table
ray_batch.py       Ray-based parallel execution and performance measurement
ray_batch_gpu_resident.py  Precomputed-signal variant (NumPy / CUDA)
results/           Recorded benchmark results
tests/             Unit tests for detection, cycle counting and scoring
docs/design.md     Requirements, design decisions and open questions
```

## Current status and limitations

| Item | Status |
|---|---|
| Extraction, detection, Ray execution | Implemented; run on 50 episodes (see performance results) |
| Episode scorer and output table | Implemented; covered by unit tests on scripted synthetic episodes. Not yet run on the full dataset. |
| Detector accuracy | **Not yet measured.** Thresholds were tuned by inspecting UMI signals, not fitted to labels. On the 50-episode subset the detector produced 51 grasps, 75 lifts and 27 places, which suggests lifts are over-detected and places under-detected. The next step is to hand-label event timesteps on about 50 episodes and report per-event precision and recall. |
| Effect on training | Not yet measured. The intended experiment is to train a small policy on all episodes versus filtered episodes and compare validation loss. |
| Streaming comparison | A Ray vs. Apache Flink streaming comparison was scoped but not built. |

## Capabilities developed

Built over February 21–22, 2026 as a first project with robot-learning data.

- Working with Zarr replay buffers and episode-indexed flat arrays
- Translating a physical task description into a testable state machine
- Designing a data-quality contract between a data pipeline and a model-training consumer
- Profiling per-episode Python overhead against Ray task-dispatch cost

## Acknowledgements

Dataset: Chi et al., *Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots*, RSS 2024.
