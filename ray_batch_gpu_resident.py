"""
ray_batch_gpu_resident.py – Experiment 2: GPU-resident tensor processing.

Two modes compared:
  A) cpu-tensor: numpy arrays, vectorized signal pre-computation, step_precomputed()
  B) gpu-tensor: torch CUDA tensors, vectorized GPU signal ops, transfer to CPU for HSM

Both modes pre-load all episode data once, pre-compute accel_mag / gripper_delta /
z_accel as vectors, then iterate HSM with step_precomputed().

Usage:
    uv run python ray_batch_gpu_resident.py --mode cpu-tensor
    uv run python ray_batch_gpu_resident.py --mode gpu-tensor --replay-scale 1000
    uv run python ray_batch_gpu_resident.py --mode both --replay-scale 10000
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import psutil
import ray
import typer

from hsm import HSMConfig, HSMDetector

app = typer.Typer()

SMALL_MANY_DIR = Path("data/small_many")
RESULTS_DIR = Path("results")

# Column names from data_gen.py output
ACCEL_COLS = ["robot0_eef_accel_0", "robot0_eef_accel_1", "robot0_eef_accel_2"]
GRIPPER_COL = "robot0_gripper_width_0"
Z_ACCEL_COL = "robot0_eef_accel_2"


@dataclasses.dataclass
class BenchResult:
    mode: str  # "cpu-tensor" or "gpu-tensor"
    replay_scale: int
    n_episodes: int
    total_rows: int
    total_events: int
    events_by_type: dict[str, int]
    load_time_s: float
    compute_time_s: float
    wall_time_s: float
    rows_per_sec: float
    events_per_sec: float
    latencies_ms: list[float]
    p50_latency_ms: float
    p99_latency_ms: float
    cpu_percent: float
    mem_mb: float


# ── Data loading ─────────────────────────────────────────────────────

def load_all_episodes_numpy() -> dict[int, dict[str, np.ndarray]]:
    """Load all episodes into numpy arrays with pre-computed signals."""
    import polars as pl

    files = sorted(SMALL_MANY_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {SMALL_MANY_DIR}. Run data_gen.py first.")

    episodes = {}
    for f in files:
        episode_id = int(f.stem.split("_")[-1])
        df = pl.read_csv(str(f))

        accel = df.select(ACCEL_COLS).to_numpy().astype(np.float32)
        gripper = df[GRIPPER_COL].to_numpy().astype(np.float32)
        z_accel = df[Z_ACCEL_COL].to_numpy().astype(np.float32)

        # Vectorized signal pre-computation
        accel_mag = np.sqrt((accel ** 2).sum(axis=1)).astype(np.float32)
        gripper_delta = np.zeros_like(gripper)
        gripper_delta[1:] = gripper[1:] - gripper[:-1]

        episodes[episode_id] = {
            "accel_mag": accel_mag,
            "gripper": gripper,
            "gripper_delta": gripper_delta,
            "z_accel": z_accel,
            "n_rows": len(df),
        }
    return episodes


def load_all_episodes_gpu() -> dict[int, dict[str, np.ndarray]]:
    """Load episodes, pre-compute signals on GPU, transfer back to numpy."""
    import polars as pl
    import torch

    device = torch.device("cuda:0")
    files = sorted(SMALL_MANY_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {SMALL_MANY_DIR}. Run data_gen.py first.")

    episodes = {}
    for f in files:
        episode_id = int(f.stem.split("_")[-1])
        df = pl.read_csv(str(f))

        accel_np = df.select(ACCEL_COLS).to_numpy().astype(np.float32)
        gripper_np = df[GRIPPER_COL].to_numpy().astype(np.float32)

        # Transfer to GPU
        accel_t = torch.from_numpy(accel_np).to(device)
        gripper_t = torch.from_numpy(gripper_np).to(device)

        # Vectorized GPU signal computation
        accel_mag_t = torch.sqrt((accel_t ** 2).sum(dim=1))
        gripper_delta_t = torch.zeros_like(gripper_t)
        gripper_delta_t[1:] = gripper_t[1:] - gripper_t[:-1]
        z_accel_t = accel_t[:, 2]  # z-component

        # Transfer pre-computed signals back to CPU numpy
        episodes[episode_id] = {
            "accel_mag": accel_mag_t.cpu().numpy(),
            "gripper": gripper_t.cpu().numpy(),
            "gripper_delta": gripper_delta_t.cpu().numpy(),
            "z_accel": z_accel_t.cpu().numpy(),
            "n_rows": len(df),
        }

    torch.cuda.synchronize()
    return episodes


# ── Ray remote task ──────────────────────────────────────────────────

@ray.remote
def process_episode_precomputed(
    episode_id: int,
    accel_mag: np.ndarray,
    gripper: np.ndarray,
    gripper_delta: np.ndarray,
    z_accel: np.ndarray,
    n_rows: int,
    replay_scale: int = 1,
) -> dict:
    """Process one episode using pre-computed signals and step_precomputed()."""
    import time as _time
    import dataclasses as _dc
    from hsm import HSMDetector

    t0 = _time.perf_counter()

    total_events = []
    total_rows_processed = 0

    for rep in range(replay_scale):
        detector = HSMDetector(episode_id=episode_id + rep * 10000)
        for i in range(n_rows):
            detector.step_precomputed(
                float(accel_mag[i]),
                float(gripper[i]),
                float(gripper_delta[i]),
                float(z_accel[i]),
            )
        total_events.extend(detector.get_all_events())
        total_rows_processed += n_rows

    elapsed_ms = (_time.perf_counter() - t0) * 1000

    summary = {"grasp": 0, "lift": 0, "place": 0}
    for ev in total_events:
        summary[ev.event_type] = summary.get(ev.event_type, 0) + 1

    return {
        "episode_id": episode_id,
        "n_rows": total_rows_processed,
        "n_events": len(total_events),
        "summary": summary,
        "latency_ms": elapsed_ms,
    }


@ray.remote(num_gpus=1)
def process_episode_precomputed_gpu(
    episode_id: int,
    accel_mag: np.ndarray,
    gripper: np.ndarray,
    gripper_delta: np.ndarray,
    z_accel: np.ndarray,
    n_rows: int,
    replay_scale: int = 1,
) -> dict:
    """GPU-pinned version of pre-computed episode processing."""
    import time as _time
    from hsm import HSMDetector

    t0 = _time.perf_counter()

    total_events = []
    total_rows_processed = 0

    for rep in range(replay_scale):
        detector = HSMDetector(episode_id=episode_id + rep * 10000)
        for i in range(n_rows):
            detector.step_precomputed(
                float(accel_mag[i]),
                float(gripper[i]),
                float(gripper_delta[i]),
                float(z_accel[i]),
            )
        total_events.extend(detector.get_all_events())
        total_rows_processed += n_rows

    elapsed_ms = (_time.perf_counter() - t0) * 1000

    summary = {"grasp": 0, "lift": 0, "place": 0}
    for ev in total_events:
        summary[ev.event_type] = summary.get(ev.event_type, 0) + 1

    return {
        "episode_id": episode_id,
        "n_rows": total_rows_processed,
        "n_events": len(total_events),
        "summary": summary,
        "latency_ms": elapsed_ms,
    }


# ── Main benchmark runner ───────────────────────────────────────────

def run_benchmark(mode: str, replay_scale: int = 1) -> BenchResult:
    """Run the GPU-resident or CPU-tensor benchmark."""
    ray.init(ignore_reinit_error=True)
    proc = psutil.Process()

    # Phase 1: Load and pre-compute signals
    t_load_start = time.perf_counter()
    if mode == "gpu-tensor":
        episodes = load_all_episodes_gpu()
    else:
        episodes = load_all_episodes_numpy()
    t_load_end = time.perf_counter()
    load_time = t_load_end - t_load_start

    # Put pre-computed arrays into Ray object store
    episode_refs = {}
    for eid, data in episodes.items():
        episode_refs[eid] = {
            "accel_mag": ray.put(data["accel_mag"]),
            "gripper": ray.put(data["gripper"]),
            "gripper_delta": ray.put(data["gripper_delta"]),
            "z_accel": ray.put(data["z_accel"]),
            "n_rows": data["n_rows"],
        }

    cpu_before = proc.cpu_percent(interval=None)
    mem_before = proc.memory_info().rss / 1e6

    # Phase 2: Dispatch Ray tasks
    t_compute_start = time.perf_counter()

    remote_fn = process_episode_precomputed_gpu if mode == "gpu-tensor" else process_episode_precomputed
    futures = []
    for eid, refs in episode_refs.items():
        futures.append(remote_fn.remote(
            eid,
            refs["accel_mag"],
            refs["gripper"],
            refs["gripper_delta"],
            refs["z_accel"],
            refs["n_rows"],
            replay_scale,
        ))
    results = ray.get(futures)

    t_compute_end = time.perf_counter()
    compute_time = t_compute_end - t_compute_start
    wall_time = (t_load_end - t_load_start) + compute_time

    cpu_after = proc.cpu_percent(interval=0.1)
    mem_after = proc.memory_info().rss / 1e6

    # Aggregate
    total_rows = sum(r["n_rows"] for r in results)
    total_events = sum(r["n_events"] for r in results)
    latencies = [r["latency_ms"] for r in results]
    events_by_type: dict[str, int] = {"grasp": 0, "lift": 0, "place": 0}
    for r in results:
        for k, v in r["summary"].items():
            events_by_type[k] = events_by_type.get(k, 0) + v

    bench = BenchResult(
        mode=mode,
        replay_scale=replay_scale,
        n_episodes=len(results),
        total_rows=total_rows,
        total_events=total_events,
        events_by_type=events_by_type,
        load_time_s=load_time,
        compute_time_s=compute_time,
        wall_time_s=wall_time,
        rows_per_sec=total_rows / compute_time if compute_time > 0 else 0,
        events_per_sec=total_events / compute_time if compute_time > 0 else 0,
        latencies_ms=latencies,
        p50_latency_ms=float(np.percentile(latencies, 50)) if latencies else 0,
        p99_latency_ms=float(np.percentile(latencies, 99)) if latencies else 0,
        cpu_percent=(cpu_before + cpu_after) / 2,
        mem_mb=max(mem_before, mem_after),
    )

    ray.shutdown()
    return bench


def save_results(bench: BenchResult):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = dataclasses.asdict(bench)
    out["latencies_ms"] = f"[{len(bench.latencies_ms)} values]"
    tag = f"ray_gpu_resident_{bench.mode.replace('-', '_')}_x{bench.replay_scale}"
    path = RESULTS_DIR / f"{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}")


def print_bench(bench: BenchResult):
    print(f"\n{'='*60}")
    print(f"  Experiment 2 — {bench.mode} @ {bench.replay_scale}x")
    print(f"{'='*60}")
    print(f"  Episodes:       {bench.n_episodes}")
    print(f"  Replay scale:   {bench.replay_scale}x")
    print(f"  Total rows:     {bench.total_rows:,}")
    print(f"  Total events:   {bench.total_events}")
    print(f"    grasp: {bench.events_by_type.get('grasp', 0)}")
    print(f"    lift:  {bench.events_by_type.get('lift', 0)}")
    print(f"    place: {bench.events_by_type.get('place', 0)}")
    print(f"  Load time:      {bench.load_time_s:.3f} s")
    print(f"  Compute time:   {bench.compute_time_s:.3f} s")
    print(f"  Wall time:      {bench.wall_time_s:.3f} s")
    print(f"  Rows/sec:       {bench.rows_per_sec:,.0f}  (compute only)")
    print(f"  Events/sec:     {bench.events_per_sec:,.1f}")
    print(f"  p50 latency:    {bench.p50_latency_ms:.1f} ms")
    print(f"  p99 latency:    {bench.p99_latency_ms:.1f} ms")
    print(f"  CPU:            {bench.cpu_percent:.1f}%")
    print(f"  Mem:            {bench.mem_mb:.0f} MB")
    print(f"{'='*60}\n")


@app.command()
def run(
    mode: str = typer.Option("both", help="Mode: cpu-tensor, gpu-tensor, or both"),
    replay_scale: int = typer.Option(1, help="Replay each episode N times (1/100/1000/10000)"),
):
    """Run the GPU-resident / CPU-tensor event detection benchmark."""
    modes = ["cpu-tensor", "gpu-tensor"] if mode == "both" else [mode]

    for m in modes:
        print(f"\nRunning Experiment 2 '{m}' @ {replay_scale}x ...")
        bench = run_benchmark(m, replay_scale=replay_scale)
        print_bench(bench)
        save_results(bench)


if __name__ == "__main__":
    app()
