# WEEK 1 – Setup, UMI Data, Shared HSM, Ray Batch Baseline
DO EXACTLY IN ORDER. First numbers today.

## Step 1 – Project setup (5 min)
mkdir robotic-umi-event-bench && cd robotic-umi-event-bench
uv init --python 3.11
uv add ray[default,data] polars pyarrow pandas typer pynvml psutil zarr tqdm

## Step 2 – Download UMI data (Stanford robotics)
git clone https://github.com/real-stanford/universal_manipulation_interface.git --depth 1
cd universal_manipulation_interface
# Example demo (quick start)
wget --recursive --no-parent --no-host-directories --cut-dirs=2 --relative --reject="index.html*" https://real.stanford.edu/umi/data/example_demo_session/
# Main cup-in-the-wild dataset (processed Zarr replay buffer)
wget https://real.stanford.edu/umi/data/zarr_datasets/cup_in_the_wild.zarr.zip
unzip cup_in_the_wild.zarr.zip -d data/raw/
cd ..

## Step 3 – data_gen.py (create & run)
```python
# data_gen.py
import zarr
import polars as pl
from pathlib import Path
import numpy as np

DATA_DIR = Path("data")
(DATA_DIR / "small_many").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "large_few").mkdir(parents=True, exist_ok=True)

def extract_umi_to_strategies():
    root = zarr.open("universal_manipulation_interface/data/raw/cup_in_the_wild.zarr", mode="r")
    # Replay buffer structure: episodes with obs/action timeseries (IMU, gripper, pose)
    for i in range(min(50, len(root["episode_ends"]))):  # subset for speed
        start = 0 if i == 0 else root["episode_ends"][i-1]
        end = root["episode_ends"][i]
        # Extract key timeseries (adapt keys to your Zarr; typical: accel, gyro, gripper, pose)
        accel = np.array(root["observations/accel"][start:end])  # example
        gripper = np.array(root["observations/gripper"][start:end])
        df = pl.DataFrame({
            "timestamp": np.arange(start, end),
            "accel_x": accel[:,0], "accel_y": accel[:,1], "accel_z": accel[:,2],
            "gripper_width": gripper,
            # add gyro, pose etc. as available
        })
        # Small-many: per-episode CSV
        df.write_csv(f"data/small_many/episode_{i:04d}.csv")
        # Large-few: append to partitioned Parquet
        df.with_columns(pl.lit(i).alias("episode_id")).write_parquet(
            f"data/large_few/episode_{i:04d}.parquet"
        )
    print("✅ UMI data extracted to small-many + large-few")

if __name__ == "__main__":
    extract_umi_to_strategies()