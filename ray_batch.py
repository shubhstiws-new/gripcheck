"""
ray_batch.py – Ray Data batch baseline for UMI event detection.

Reads both dataset strategies (small-many CSV, large-few Parquet),
runs the shared HSM detector, and collects performance metrics.

Usage:
    uv run python ray_batch.py --strategy small-many
    uv run python ray_batch.py --strategy large-few
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


def process_episode_csv(path: str) -> dict:
    """Process a single CSV episode file through the HSM."""
    t0 = time.perf_counter()
    df = pl.read_csv(path)
    episode_id = int(Path(path).stem.split("_")[-1])

    detector = HSMDetector(episode_id=episode_id)
    rows = df.to_dicts()
    for row in rows:
        detector.step(row)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    events = detector.get_all_events()
    return {
        "episode_id": episode_id,
        "n_rows": len(rows),
        "n_events": len(events),
        "events": [dataclasses.asdict(e) for e in events],
        "summary": detector.summary(),
        "latency_ms": elapsed_ms,
    }


def process_episode_parquet(path: str) -> dict:
    """Process a single Parquet episode file through the HSM."""
    t0 = time.perf_counter()
    df = pl.read_parquet(path)
    episode_id = int(df["episode_id"][0]) if "episode_id" in df.columns else 0

    detector = HSMDetector(episode_id=episode_id)
    rows = df.drop("episode_id").to_dicts()
    for row in rows:
        detector.step(row)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    events = detector.get_all_events()
    return {
        "episode_id": episode_id,
        "n_rows": len(rows),
        "n_events": len(events),
        "events": [dataclasses.asdict(e) for e in events],
        "summary": detector.summary(),
        "latency_ms": elapsed_ms,
    }


def run_ray_batch(strategy: str) -> BenchResult:
    """Run Ray batch pipeline on the given dataset strategy."""
    ray.init(ignore_reinit_error=True)
    proc = psutil.Process()

    if strategy == "small-many":
        data_dir = SMALL_MANY_DIR
        files = sorted(data_dir.glob("*.csv"))
        process_fn = process_episode_csv
    else:
        data_dir = LARGE_FEW_DIR
        files = sorted(data_dir.glob("*.parquet"))
        process_fn = process_episode_parquet

    if not files:
        raise FileNotFoundError(f"No files in {data_dir}. Run data_gen.py first.")

    file_paths = [str(f) for f in files]

    # Create Ray dataset from file paths and map processing
    cpu_before = proc.cpu_percent(interval=None)
    mem_before = proc.memory_info().rss / 1e6

    t_start = time.perf_counter()

    # Use Ray tasks for parallel per-episode processing
    @ray.remote
    def process_remote(path: str, fn_name: str) -> dict:
        # Re-import inside Ray worker
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

        detector = HSMDetector(episode_id=episode_id)
        rows = df.to_dicts()
        for row in rows:
            detector.step(row)

        elapsed_ms = (_time.perf_counter() - t0) * 1000
        events = detector.get_all_events()
        return {
            "episode_id": episode_id,
            "n_rows": len(rows),
            "n_events": len(events),
            "events": [_dc.asdict(e) for e in events],
            "summary": detector.summary(),
            "latency_ms": elapsed_ms,
        }

    fn_name = "csv" if strategy == "small-many" else "parquet"
    futures = [process_remote.remote(p, fn_name) for p in file_paths]
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
    # Remove raw latencies list from JSON for readability
    out["latencies_ms"] = f"[{len(bench.latencies_ms)} values]"
    path = RESULTS_DIR / f"ray_batch_{bench.strategy.replace('-', '_')}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}")


def print_bench(bench: BenchResult):
    print(f"\n{'='*60}")
    print(f"  Ray Batch Baseline — {bench.strategy}")
    print(f"{'='*60}")
    print(f"  Episodes:       {bench.n_episodes}")
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
):
    """Run the Ray batch event detection benchmark."""
    strategies = ["small-many", "large-few"] if strategy == "both" else [strategy]

    for strat in strategies:
        print(f"\nRunning Ray batch on '{strat}' strategy...")
        bench = run_ray_batch(strat)
        print_bench(bench)
        save_results(bench)


if __name__ == "__main__":
    app()
