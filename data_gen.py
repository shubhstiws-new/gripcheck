"""
data_gen.py – Extract UMI Zarr replay buffer into two dataset strategies:
  - small-many: per-episode CSVs (1–10 MB each)
  - large-few:  per-episode Parquet files (partitioned, later merged)

Adapts to the actual UMI Zarr structure:
  meta/episode_ends  →  cumulative end indices
  data/*             →  flat arrays across all timesteps
"""

import zarr
import polars as pl
import numpy as np
from pathlib import Path

ZARR_PATH = Path("data/raw/cup_in_the_wild.zarr")
DATA_DIR = Path("data")
SMALL_MANY = DATA_DIR / "small_many"
LARGE_FEW = DATA_DIR / "large_few"
MAX_EPISODES = 50  # subset for speed


def discover_keys(root: zarr.Group) -> dict[str, tuple]:
    """List all data arrays and their shapes, skipping unavailable codecs."""
    data_grp = root["data"]
    result = {}
    for k in data_grp.array_keys():
        try:
            result[k] = data_grp[k].shape
        except ValueError:
            print(f"  Skipping {k} (codec unavailable)")
    return result


def extract_umi_to_strategies():
    for d in (SMALL_MANY, LARGE_FEW):
        d.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(ZARR_PATH), mode="r")
    episode_ends = np.array(root["meta/episode_ends"])
    n_episodes = min(MAX_EPISODES, len(episode_ends))

    # Discover available data keys
    key_shapes = discover_keys(root)
    print(f"Found {len(key_shapes)} data arrays, {len(episode_ends)} total episodes")
    for k, s in key_shapes.items():
        print(f"  {k}: {s}")

    # Select low-dim keys we care about for event detection
    # Skip high-dim arrays like camera images
    selected_keys = []
    for k, shape in key_shapes.items():
        if len(shape) == 1:
            selected_keys.append(k)  # scalar per timestep (e.g. timestamp)
        elif len(shape) == 2 and shape[1] <= 10:
            selected_keys.append(k)  # low-dim vector (pos, rot, gripper, etc.)
        # skip camera/image arrays (3+ dims or large 2nd dim)

    print(f"\nSelected {len(selected_keys)} low-dim keys for extraction:")
    for k in selected_keys:
        print(f"  {k}: {key_shapes[k]}")

    data_grp = root["data"]
    for i in range(n_episodes):
        start = 0 if i == 0 else int(episode_ends[i - 1])
        end = int(episode_ends[i])
        ep_len = end - start

        # Build columns dict
        columns: dict[str, np.ndarray | list] = {}
        columns["timestep"] = np.arange(ep_len)

        for k in selected_keys:
            arr = np.array(data_grp[k][start:end])
            if arr.ndim == 1:
                columns[k] = arr
            elif arr.ndim == 2:
                for j in range(arr.shape[1]):
                    columns[f"{k}_{j}"] = arr[:, j]

        df = pl.DataFrame(columns)

        # Derive synthetic accel proxy from EEF velocity (finite diff of position)
        # This gives us an "IMU-like" signal for event detection
        pos_cols = [c for c in df.columns if "eef_pos" in c and "vel" not in c and "wrt" not in c]
        for col in pos_cols:
            vel = df[col].diff().fill_null(0.0)
            acc = vel.diff().fill_null(0.0)
            df = df.with_columns(acc.alias(col.replace("eef_pos", "eef_accel")))

        # Small-many: per-episode CSV
        csv_path = SMALL_MANY / f"episode_{i:04d}.csv"
        df.write_csv(csv_path)

        # Large-few: per-episode Parquet with episode_id column
        pq_path = LARGE_FEW / f"episode_{i:04d}.parquet"
        df.with_columns(pl.lit(i).alias("episode_id")).write_parquet(pq_path)

        print(f"  Episode {i:4d}: {ep_len:6d} steps, {len(df.columns)} cols → {csv_path.name} / {pq_path.name}")

    # Summary
    csv_sizes = [f.stat().st_size for f in SMALL_MANY.glob("*.csv")]
    pq_sizes = [f.stat().st_size for f in LARGE_FEW.glob("*.parquet")]
    print(f"\nDone! {n_episodes} episodes extracted.")
    print(f"  small-many CSVs:  {len(csv_sizes)} files, total {sum(csv_sizes)/1e6:.1f} MB")
    print(f"  large-few Parquet: {len(pq_sizes)} files, total {sum(pq_sizes)/1e6:.1f} MB")


if __name__ == "__main__":
    extract_umi_to_strategies()
