# Robotic UMI Event Detection Benchmark – Overall Plan (Stanford Robotics Dataset)
Goal: Run a lean 5D experiment on real Stanford robot manipulation data to answer:
"For this robotics scenario + 3 critical events → use Ray or Flink?"

## Dataset (fixed for laser focus)
- Universal Manipulation Interface (UMI) – real-stanford lab (Stanford University, Bay Area)
- In-the-wild human demos (handheld gripper + GoPro) transferred to robot policies
- Active unsolved problem: scaling manipulation to unstructured real-world environments (Stanford 2024–2026 frontier)
- High-frequency timeseries: GoPro IMU (accel/gyro), gripper/joint states, SLAM poses
- Two strategies: small-many (1–10 MB CSVs per demo episode) vs large-few (100 MB+ Parquet partitioned by task/session)

## 3 Critical Events (HSM detects these)
1. Grasp/Contact – IMU spike + gripper closure
2. Lift/Transport – vertical accel + stable hold
3. Place/Release – deceleration + gripper open

## 5D Matrix (auto-swept)
- Hardware: your 16c/32t AMD + 2×3090 (later edge robot board)
- Framework: Ray (first) vs PyFlink
- Dataset strategy: small-many vs large-few
- Operation: batch vs streaming replay
- Replay scale: 1× / 100× / 1 000× / 10 000× (emulates high-rate streaming)

## Weekly Roadmap (4 weeks, <2 h/day)
Week 1: Env + UMI data extraction + shared HSM + Ray batch baseline
Week 2: Ray streaming + full sweep on your hardware
Week 3: PyFlink identical logic + comparison
Week 4: Analysis → automated decision rules + edge port

## Success Metric
Events/s, p99 latency for the 3 events, GPU/CPU util. Produce rules.json policy.

Alignment
- Jim Fan: GPU-accelerated embodied robotics pipelines
- Yann LeCun: minimal-overhead temporal reasoning on sensor streams

Start with WEEK_1_SETUP_AND_DATA.md exactly.