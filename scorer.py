"""
scorer.py – Episode quality scoring for UMI demonstrations.

Runs the HSM event detector over each episode, summarises the detected
event sequence and basic motion statistics, and assigns a 0–1 quality score
that downstream training jobs can filter on.

Usage:
    uv run python scorer.py --episodes-dir data/large_few --out results/episodes_scored.parquet
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import typer

from hsm import Event, HSMConfig, HSMDetector

# Event-pattern scores. A complete grasp → lift → place cycle is a clean demonstration;
# partial cycles still carry signal for grasp / lift training.
SCORE_COMPLETE = 1.0
SCORE_GRASP_LIFT = 0.6
SCORE_GRASP_ONLY = 0.3
SCORE_NO_EVENTS = 0.0

# Episodes that fail these checks score 0 regardless of detected events.
MIN_STEPS = 50              # shorter episodes are typically aborted demonstrations
MIN_GRIPPER_RANGE_M = 0.01  # gripper never meaningfully opened or closed
MIN_TRAVEL_M = 0.05         # end effector barely moved

GRIPPER_COL = "robot0_gripper_width_0"
POS_COLS = [f"robot0_eef_pos_{i}" for i in range(3)]
ACCEL_COLS = [f"robot0_eef_accel_{i}" for i in range(3)]


def count_complete_cycles(event_types: list[str]) -> int:
    """Count ordered grasp → lift → place subsequences, without reusing events."""
    cycles, expected = 0, "grasp"
    for ev in event_types:
        if ev == expected:
            if ev == "place":
                cycles += 1
                expected = "grasp"
            else:
                expected = "lift" if ev == "grasp" else "place"
        elif ev == "grasp":
            # A new grasp restarts the cycle (e.g. a regrasp after a fumble).
            expected = "lift"
    return cycles


def pattern_score(event_types: list[str]) -> float:
    if count_complete_cycles(event_types) > 0:
        return SCORE_COMPLETE
    if "grasp" in event_types:
        grasp_at = event_types.index("grasp")
        if "lift" in event_types[grasp_at:]:
            return SCORE_GRASP_LIFT
        return SCORE_GRASP_ONLY
    return SCORE_NO_EVENTS


def score_episode(df: pl.DataFrame, episode_id: int, config: HSMConfig | None = None) -> dict:
    """Detect events in one episode and return a single scored summary row."""
    t0 = time.perf_counter()

    detector = HSMDetector(episode_id=episode_id, config=config)
    signal_cols = [c for c in df.columns if c != "episode_id"]
    for row in df.select(signal_cols).iter_rows(named=True):
        detector.step(row)
    events: list[Event] = detector.get_all_events()
    event_types = [e.event_type for e in events]

    n_steps = df.height
    gripper = df[GRIPPER_COL].to_numpy()
    pos = df.select(POS_COLS).to_numpy()
    accel_mag = np.linalg.norm(df.select(ACCEL_COLS).to_numpy(), axis=1)

    gripper_range = float(gripper.max() - gripper.min()) if n_steps else 0.0
    travel = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if n_steps > 1 else 0.0

    flags = []
    if n_steps < MIN_STEPS:
        flags.append("too_short")
    if gripper_range < MIN_GRIPPER_RANGE_M:
        flags.append("gripper_static")
    if travel < MIN_TRAVEL_M:
        flags.append("no_motion")

    score = SCORE_NO_EVENTS if flags else pattern_score(event_types)

    return {
        "episode_id": episode_id,
        "n_timesteps": n_steps,
        "n_events": len(events),
        "n_grasps": event_types.count("grasp"),
        "n_lifts": event_types.count("lift"),
        "n_places": event_types.count("place"),
        "event_sequence": ",".join(event_types),
        "n_complete_cycles": count_complete_cycles(event_types),
        "quality_score": score,
        "quality_flags": ",".join(flags),
        "gripper_range": gripper_range,
        "eef_travel_dist": travel,
        "mean_accel_mag": float(accel_mag.mean()) if n_steps else 0.0,
        "max_accel_mag": float(accel_mag.max()) if n_steps else 0.0,
        "processing_time_ms": (time.perf_counter() - t0) * 1000,
    }


def score_directory(episodes_dir: Path, config: HSMConfig | None = None) -> pl.DataFrame:
    """Score every per-episode Parquet file produced by data_gen.py."""
    rows = []
    for path in sorted(episodes_dir.glob("episode_*.parquet")):
        episode_id = int(path.stem.split("_")[1])
        rows.append(score_episode(pl.read_parquet(path), episode_id, config))
    return pl.DataFrame(rows)


def main(
    episodes_dir: Path = typer.Option(Path("data/large_few")),
    out: Path = typer.Option(Path("results/episodes_scored.parquet")),
) -> None:
    scored = score_directory(episodes_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(out)

    print(f"Scored {scored.height} episodes → {out}")
    print(scored.group_by("quality_score").len().sort("quality_score", descending=True))
    flagged = scored.filter(pl.col("quality_flags") != "")
    print(f"Flagged by quality checks: {flagged.height}")


if __name__ == "__main__":
    typer.run(main)
