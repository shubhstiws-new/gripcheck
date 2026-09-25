# GripCheck – Design Notes

## Problem statement

Imitation-learning pipelines train on thousands of teleoperated episodes. Some fraction are unusable: the operator fumbles, the gripper fails to close, or the arm stalls mid-trajectory. Training on these episodes wastes compute and degrades the policy. Discarding good episodes also costs training signal. The pipeline therefore needs a transparent, per-episode quality signal that the training owner can threshold.

## Data

`cup_in_the_wild.zarr` stores all episodes as flat, concatenated arrays. `meta/episode_ends` holds cumulative end indices: episode *i* spans `[ends[i-1], ends[i])`.

| Array | Shape | Use |
|---|---|---|
| `robot0_eef_pos` | [T, 3] | End-effector position (m); source for velocity and acceleration |
| `robot0_eef_rot_axis_angle` | [T, 3] | End-effector orientation |
| `robot0_gripper_width` | [T, 1] | Finger separation (m); about 0.08 open, about 0.01 closed |
| `robot0_demo_start_pose`, `robot0_demo_end_pose` | [T, 6] | Constant per episode; metadata only |
| `camera0_rgb` | [T, 224, 224, 3] | Not read |

Episode lengths range from 28 to 1,605 steps (mean 483), sampled at about 10 Hz.

## Design decisions

1. **Sequential inside an episode, parallel across episodes.** The state machine is inherently sequential, because each step depends on the previous state. Episodes are independent, so parallelism is applied at the episode level. For very short episodes, task-dispatch overhead can exceed compute, so batching small episodes together is preferable to one task per episode.
2. **Transparent scoring over a learned score.** Without labels, a rule-based score with named flags (`too_short`, `gripper_static`, `no_motion`) is auditable. The training owner can see why an episode was excluded and override the threshold.
3. **Configuration over constants.** Detection thresholds live in `HSMConfig`, because gripper geometry and sampling rates differ across datasets and robots.
4. **Reader isolation.** Extraction is the only Zarr-specific stage. Other formats (RLDS, LeRobot Parquet) should require only a new reader that emits the same per-episode columns.

## Output contract

`episodes_scored.parquet`, one row per episode:

| Column | Type | Meaning |
|---|---|---|
| `episode_id` | int | Episode index |
| `n_timesteps` | int | Episode length |
| `n_events`, `n_grasps`, `n_lifts`, `n_places` | int | Detected event counts |
| `event_sequence` | str | Ordered events, for example `grasp,lift,place` |
| `n_complete_cycles` | int | Ordered grasp → lift → place cycles |
| `quality_score` | float | 0.0–1.0 |
| `quality_flags` | str | Sanity checks that failed |
| `gripper_range`, `eef_travel_dist` | float | Motion evidence |
| `mean_accel_mag`, `max_accel_mag` | float | Smoothness evidence |
| `processing_time_ms` | float | Per-episode cost |

## Shared responsibilities and open questions

| Topic | Pipeline owner | Other party |
|---|---|---|
| Invalid input (NaNs, overlapping episode boundaries) | Detect and report episode IDs; do not silently repair | Data producer: fix the collection process |
| Score threshold | Publish the score distribution and flag breakdown | Training owner: choose the threshold |
| Event definitions for new hardware | Keep thresholds configurable; provide per-episode inspection | Domain expert: define the events |
| Scale beyond one machine | Benchmark and state resource needs | Platform team: provision the cluster |

## Planned validation

1. Hand-label grasp, lift and place timesteps for about 50 episodes; report per-event precision and recall with a ±3-step tolerance.
2. Compare the score distribution with a manual good/bad review of a random sample of 30 episodes.
3. Train a small behaviour-cloning policy on all episodes versus episodes with `quality_score ≥ 0.6`; compare validation loss.
