# hsm.py
import numpy as np

class UMI_HSM:
    def __init__(self):
        self.state = "IDLE"
        self.imu_buffer = []  # rolling window

    def process(self, row: dict) -> list:
        accel = np.array([row["accel_x"], row["accel_y"], row["accel_z"]])
        self.imu_buffer.append(accel)
        if len(self.imu_buffer) > 20:
            self.imu_buffer.pop(0)
        events = []
        # 1. Grasp/Contact
        if row["gripper_width"] < 0.02 and np.std(self.imu_buffer) > 0.5 and self.state == "IDLE":
            events.append("Grasp/Contact")
            self.state = "GRASPING"
        # 2. Lift/Transport
        elif self.state == "GRASPING" and accel[2] > 1.5:  # upward accel
            events.append("Lift/Transport")
            self.state = "TRANSPORT"
        # 3. Place/Release
        elif self.state == "TRANSPORT" and accel[2] < -1.0 and row["gripper_width"] > 0.05:
            events.append("Place/Release")
            self.state = "IDLE"
        return events


Step 5 – ray_pipeline.py (batch baseline – skeleton)
Create with Ray Data .map_batches using your GPU actors + UMI_HSM.

Step 6 – First test
ray start --head --num-gpus=2
uv run python -c "
import ray
from ray_pipeline import run_batch
ray.init()
run_batch(dataset_strategy='small_many', gpu=True, replay_scale=50)
"
Prints throughput + detected events count.
Finish Week 1 → commit + say “Week 1 done” for WEEK_2 exact file.


---

### File 3: WEEK_2_RAY_PIPELINE_AND_SWEEP.md

```markdown
# WEEK 2 – Full Ray Sweep on your 3090 hardware
Create experiments/sweep.py (Typer CLI) – exactly the 4 combos per strategy.

Variables:
--dataset_strategy small_many|large_few
--mode batch|streaming
--gpu
--replay_scale 1|100|1000|10000

Streaming: Ray + Redpanda (docker-compose optional).
HSM processes live rows → count the 3 events.

Run overnight: `uv run experiments/sweep.py --framework ray --all`
Results append to results.parquet.

Week 2 ends with Ray numbers locked.

# WEEK 3 – PyFlink side-by-side (identical logic)
1. uv add apache-flink
2. flink_pipeline.py – copy hsm.py into KeyedProcessFunction (stateful HSM) or Table + MATCH_RECOGNIZE.
3. Same sweep.py with --framework flink
4. docker-compose up -d only for streaming (Redpanda + MinIO)

Run identical combos. Compare side-by-side in Week 4 notebook.

# WEEK 4 – Analysis + Decision Rules + Edge Port
analysis.ipynb:
- Load results.parquet
- Heatmaps + pivot on the 5D
- Export rules.json (e.g. {"small_many + streaming + gpu": "Ray", ...})

Edge port (side-step):
- Containerize winning Ray job
- Deploy on Jetson Orin Nano (K3s one-liner) or real robot board
- Same HSM, same code, measure power/latency on physical robot stream

Final deliverable: Your automated decision framework (no more confusion).

When done → rent cloud cluster and run on full UMI or your own robot logs.