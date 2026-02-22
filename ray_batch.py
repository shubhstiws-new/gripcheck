"""
ray_batch.py – Ray Data batch baseline for UMI event detection.

Reads both dataset strategies (small-many CSV, large-few Parquet),
runs the shared HSM detector, and collects performance metrics.

Supports GPU actors and replay_scale to emulate high-rate streaming.

Usage:
    uv run python ray_batch.py --strategy small-many
    uv run python ray_batch.py --strategy large-few --gpu --replay-scale 100
    uv run python ray_batch.py --strategy both
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import psutil
import ray
import typer

from hsm import HSMConfig, HSMDetector, Event

app = typer.Typer()

SMALL_MANY_DIR = Path("data/small_many")
LARGE_FEW_DIR = Path("data/large_few")
RESULTS_DIR = Path("results")


@dataclasses.dataclass
class BenchResult:
    strategy: str
    replay_scale: int
    gpu: bool
    n_episodes: int
    total_rows: int
    total_events: int
    events_by_type: dict[str, int]
    wall_time_s: float
    rows_per_sec: float
    events_per_sec: float
    latencies_ms: list[float]   # per-episode processing time
    p50_latency_ms: float
    p99_latency_ms: float
    cpu_percent: float
    mem_mb: float


# ── GPU Actor for map_batches ────────────────────────────────────────

@ray.remote(num_gpus=1)
class GPUHSMActor:
    """Ray actor that processes batches on a GPU-assigned worker."""

    def __init__(self):
        import torch  # noqa: F401 – ensures GPU is claimed
        self.device = "gpu"

    def process_batch(self, batch: dict) -> dict:
        """Process a batch of rows through the HSM. Returns event counts."""
        from hsm import HSMDetector
        import dataclasses as _dc

        results = []
        # batch is a dict of columns; iterate row-wise
        n_rows = len(next(iter(batch.values())))
        episode_ids = batch.get("episode_id", list(range(n_rows)))

        # Group by episode
        episodes: dict[int, list[dict]] = {}
        for i in range(n_rows):
            row = {k: v[i] for k, v in batch.items() if k != "episode_id"}
            eid = int(episode_ids[i]) if "episode_id" in batch else 0
            episodes.setdefault(eid, []).append(row)

        all_events = []
        for eid, rows in episodes.items():
            detector = HSMDetector(episode_id=eid)
            for row in rows:
                detector.step(row)
            all_events.extend([_dc.asdict(e) for e in detector.get_all_events()])

        return {"events": all_events, "n_rows": n_rows}


@ray.remote
class CPUHSMActor:
    """Ray actor that processes batches on CPU workers."""

    def process_batch(self, batch: dict) -> dict:
        from hsm import HSMDetector
        import dataclasses as _dc

        n_rows = len(next(iter(batch.values())))
        episode_ids = batch.get("episode_id", list(range(n_rows)))

        episodes: dict[int, list[dict]] = {}
        for i in range(n_rows):
            row = {k: v[i] for k, v in batch.items() if k != "episode_id"}
            eid = int(episode_ids[i]) if "episode_id" in batch else 0
            episodes.setdefault(eid, []).append(row)

        all_events = []
        for eid, rows in episodes.items():
            detector = HSMDetector(episode_id=eid)
            for row in rows:
                detector.step(row)
            all_events.extend([_dc.asdict(e) for e in detector.get_all_events()])

        return {"events": all_events, "n_rows": n_rows}


# ── Per-episode remote task (simple parallelism) ────────────────────

@ray.remote
def process_episode_remote(path: str, fn_name: str, replay_scale: int = 1) -> dict:
    """Process one episode file through HSM, optionally replaying N times."""
    import time as _time
    import polars as _pl
    import dataclasses as _dc
    from pathlib import Path as _Path
    from hsm import HSMDetector

    t0 = _time.perf_counter()
    if fn_name == "csv":
        df = _pl.read_csv(path)
    else:
        df = _pl.read_parquet(path)

    episode_id = int(_Path(path).stem.split("_")[-1])
    drop_cols = [c for c in ["episode_id"] if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    rows = df.to_dicts()

    # Replay scale: process the same episode N times to emulate higher data rates
    total_events = []
    total_rows_processed = 0
    for rep in range(replay_scale):
        detector = HSMDetector(episode_id=episode_id + rep * 10000)
        for row in rows:
            detector.step(row)
        total_events.extend(detector.get_all_events())
        total_rows_processed += len(rows)

    elapsed_ms = (_time.perf_counter() - t0) * 1000

    summary = {"grasp": 0, "lift": 0, "place": 0}
    for ev in total_events:
        summary[ev.event_type] = summary.get(ev.event_type, 0) + 1

    return {
        "episode_id": episode_id,
        "n_rows": total_rows_processed,
        "n_events": len(total_events),
        "events": [_dc.asdict(e) for e in total_events[:100]],  # cap for serialization
        "summary": summary,
        "latency_ms": elapsed_ms,
    }


@ray.remote(num_gpus=1)
def process_episode_remote_gpu(path: str, fn_name: str, replay_scale: int = 1) -> dict:
    """GPU-pinned version: claims a GPU slot for the worker."""
    import time as _time
    import polars as _pl
    import dataclasses as _dc
    from pathlib import Path as _Path
    from hsm import HSMDetector

    t0 = _time.perf_counter()
    if fn_name == "csv":
        df = _pl.read_csv(path)
    else:
        df = _pl.read_parquet(path)

    episode_id = int(_Path(path).stem.split("_")[-1])
    drop_cols = [c for c in ["episode_id"] if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    rows = df.to_dicts()

    total_events = []
    total_rows_processed = 0
    for rep in range(replay_scale):
        detector = HSMDetector(episode_id=episode_id + rep * 10000)
        for row in rows:
            detector.step(row)
        total_events.extend(detector.get_all_events())
        total_rows_processed += len(rows)

    elapsed_ms = (_time.perf_counter() - t0) * 1000

    summary = {"grasp": 0, "lift": 0, "place": 0}
    for ev in total_events:
        summary[ev.event_type] = summary.get(ev.event_type, 0) + 1

    return {
        "episode_id": episode_id,
        "n_rows": total_rows_processed,
        "n_events": len(total_events),
        "events": [_dc.asdict(e) for e in total_events[:100]],
        "summary": summary,
        "latency_ms": elapsed_ms,
    }


# ── Main benchmark runner ───────────────────────────────────────────

def run_ray_batch(strategy: str, gpu: bool = False, replay_scale: int = 1) -> BenchResult:
    """Run Ray batch pipeline on the given dataset strategy."""
    ray.init(ignore_reinit_error=True)
    proc = psutil.Process()

    if strategy == "small-many":
        data_dir = SMALL_MANY_DIR
        files = sorted(data_dir.glob("*.csv"))
        fn_name = "csv"
    else:
        data_dir = LARGE_FEW_DIR
        files = sorted(data_dir.glob("*.parquet"))
        fn_name = "parquet"

    if not files:
        raise FileNotFoundError(f"No files in {data_dir}. Run data_gen.py first.")

    file_paths = [str(f.resolve()) for f in files]

    cpu_before = proc.cpu_percent(interval=None)
    mem_before = proc.memory_info().rss / 1e6

    t_start = time.perf_counter()

    # Dispatch to GPU or CPU remote tasks
    remote_fn = process_episode_remote_gpu if gpu else process_episode_remote
    futures = [remote_fn.remote(p, fn_name, replay_scale) for p in file_paths]
    results = ray.get(futures)

    t_end = time.perf_counter()
    wall_time = t_end - t_start

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
        strategy=strategy,
        replay_scale=replay_scale,
        gpu=gpu,
        n_episodes=len(results),
        total_rows=total_rows,
        total_events=total_events,
        events_by_type=events_by_type,
        wall_time_s=wall_time,
        rows_per_sec=total_rows / wall_time if wall_time > 0 else 0,
        events_per_sec=total_events / wall_time if wall_time > 0 else 0,
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
    tag = f"ray_batch_{bench.strategy.replace('-', '_')}_x{bench.replay_scale}"
    if bench.gpu:
        tag += "_gpu"
    path = RESULTS_DIR / f"{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}")


def print_bench(bench: BenchResult):
    gpu_tag = " [GPU]" if bench.gpu else " [CPU]"
    print(f"\n{'='*60}")
    print(f"  Ray Batch Baseline — {bench.strategy} @ {bench.replay_scale}x{gpu_tag}")
    print(f"{'='*60}")
    print(f"  Episodes:       {bench.n_episodes}")
    print(f"  Replay scale:   {bench.replay_scale}x")
    print(f"  Total rows:     {bench.total_rows:,}")
    print(f"  Total events:   {bench.total_events}")
    print(f"    grasp: {bench.events_by_type.get('grasp', 0)}")
    print(f"    lift:  {bench.events_by_type.get('lift', 0)}")
    print(f"    place: {bench.events_by_type.get('place', 0)}")
    print(f"  Wall time:      {bench.wall_time_s:.3f} s")
    print(f"  Rows/sec:       {bench.rows_per_sec:,.0f}")
    print(f"  Events/sec:     {bench.events_per_sec:,.1f}")
    print(f"  p50 latency:    {bench.p50_latency_ms:.1f} ms")
    print(f"  p99 latency:    {bench.p99_latency_ms:.1f} ms")
    print(f"  CPU:            {bench.cpu_percent:.1f}%")
    print(f"  Mem:            {bench.mem_mb:.0f} MB")
    print(f"{'='*60}\n")


@app.command()
def run(
    strategy: str = typer.Option("both", help="Dataset strategy: small-many, large-few, or both"),
    gpu: bool = typer.Option(False, help="Use GPU actors (requires CUDA GPUs)"),
    replay_scale: int = typer.Option(1, help="Replay each episode N times (1/100/1000/10000)"),
):
    """Run the Ray batch event detection benchmark."""
    strategies = ["small-many", "large-few"] if strategy == "both" else [strategy]

    for strat in strategies:
        print(f"\nRunning Ray batch on '{strat}' @ {replay_scale}x {'[GPU]' if gpu else '[CPU]'}...")
        bench = run_ray_batch(strat, gpu=gpu, replay_scale=replay_scale)
        print_bench(bench)
        save_results(bench)


if __name__ == "__main__":
    app()
