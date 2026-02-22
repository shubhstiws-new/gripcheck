"""
hsm.py – Shared Hierarchical State Machine for UMI event detection.

Detects 3 critical manipulation events from robot sensor timeseries:
  1. Grasp/Contact  – acceleration spike + gripper closing
  2. Lift/Transport  – upward acceleration + stable gripper hold
  3. Place/Release   – deceleration + gripper opening

Framework-agnostic: used identically by Ray and Flink pipelines.
Operates row-by-row with internal state (suitable for streaming).
"""

from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import Any

import numpy as np


class Phase(Enum):
    IDLE = auto()
    APPROACHING = auto()
    GRASPING = auto()
    LIFTING = auto()
    TRANSPORTING = auto()
    PLACING = auto()
    RELEASING = auto()


@dataclasses.dataclass
class Event:
    event_type: str          # "grasp", "lift", "place"
    timestep: int
    episode_id: int
    phase_from: str
    phase_to: str
    confidence: float        # 0-1 heuristic confidence
    details: dict[str, Any]  # signal values that triggered


# ── Tunable thresholds (calibrated for UMI cup-in-the-wild) ──────────
@dataclasses.dataclass
class HSMConfig:
    # Grasp detection
    accel_spike_thresh: float = 0.02      # EEF accel magnitude spike
    gripper_close_thresh: float = 0.06    # gripper width < this → closed
    gripper_closing_delta: float = -0.002 # gripper narrowing rate

    # Lift detection
    vertical_accel_thresh: float = 0.01   # upward accel (z-axis)
    stable_hold_width: float = 0.06       # gripper width for "holding"

    # Place detection
    decel_thresh: float = -0.01           # negative accel (slowing)
    gripper_open_thresh: float = 0.07     # gripper width > this → opening
    gripper_opening_delta: float = 0.002  # gripper widening rate


class HSMDetector:
    """Stateful per-episode event detector. Call `step()` for each timestep."""

    def __init__(self, episode_id: int = 0, config: HSMConfig | None = None):
        self.episode_id = episode_id
        self.cfg = config or HSMConfig()
        self.phase = Phase.IDLE
        self.events: list[Event] = []
        self.prev_gripper: float | None = None
        self.prev_accel_mag: float | None = None
        self._step_count = 0

    def step(self, row: dict[str, float]) -> list[Event]:
        """
        Process one timestep. Returns list of events detected at this step.

        Expected row keys (flexible naming, searches for patterns):
          - accel columns: anything with 'accel' in name (x, y, z components)
          - gripper column: anything with 'gripper' in name
          - vertical pos/accel: z-component of eef_pos or eef_accel
        """
        new_events: list[Event] = []
        ts = int(row.get("timestep", self._step_count))
        self._step_count += 1

        # Extract signals from row
        accel_cols = {k: v for k, v in row.items() if "accel" in k.lower()}
        gripper_cols = {k: v for k, v in row.items() if "gripper" in k.lower()}

        # Acceleration magnitude
        accel_vals = list(accel_cols.values())
        accel_mag = float(np.sqrt(sum(v**2 for v in accel_vals))) if accel_vals else 0.0

        # Gripper width (take first gripper column)
        gripper = float(list(gripper_cols.values())[0]) if gripper_cols else 0.0

        # Gripper delta
        gripper_delta = 0.0
        if self.prev_gripper is not None:
            gripper_delta = gripper - self.prev_gripper

        # Vertical accel (z-component, typically last accel col or one with 'z' or '_2')
        z_accel = 0.0
        for k, v in accel_cols.items():
            if k.endswith("_2") or "z" in k.lower():
                z_accel = float(v)
                break

        # ── State machine transitions ────────────────────────────
        cfg = self.cfg

        if self.phase in (Phase.IDLE, Phase.APPROACHING, Phase.RELEASING):
            # Detect GRASP: accel spike + gripper closing
            if (accel_mag > cfg.accel_spike_thresh
                    and gripper_delta < cfg.gripper_closing_delta
                    and gripper < cfg.gripper_close_thresh):
                confidence = min(1.0, accel_mag / (cfg.accel_spike_thresh * 3))
                ev = Event(
                    event_type="grasp",
                    timestep=ts,
                    episode_id=self.episode_id,
                    phase_from=self.phase.name,
                    phase_to=Phase.GRASPING.name,
                    confidence=confidence,
                    details={"accel_mag": accel_mag, "gripper": gripper,
                             "gripper_delta": gripper_delta},
                )
                new_events.append(ev)
                self.phase = Phase.GRASPING

        if self.phase == Phase.GRASPING:
            # Detect LIFT: upward accel + stable grip
            if (z_accel > cfg.vertical_accel_thresh
                    and gripper < cfg.stable_hold_width):
                confidence = min(1.0, z_accel / (cfg.vertical_accel_thresh * 3))
                ev = Event(
                    event_type="lift",
                    timestep=ts,
                    episode_id=self.episode_id,
                    phase_from=self.phase.name,
                    phase_to=Phase.LIFTING.name,
                    confidence=confidence,
                    details={"z_accel": z_accel, "gripper": gripper},
                )
                new_events.append(ev)
                self.phase = Phase.LIFTING

        if self.phase in (Phase.LIFTING, Phase.TRANSPORTING):
            # Detect PLACE: deceleration + gripper opening
            if (z_accel < cfg.decel_thresh
                    and gripper_delta > cfg.gripper_opening_delta):
                confidence = min(1.0, abs(z_accel) / abs(cfg.decel_thresh * 3))
                ev = Event(
                    event_type="place",
                    timestep=ts,
                    episode_id=self.episode_id,
                    phase_from=self.phase.name,
                    phase_to=Phase.PLACING.name,
                    confidence=confidence,
                    details={"z_accel": z_accel, "gripper": gripper,
                             "gripper_delta": gripper_delta},
                )
                new_events.append(ev)
                self.phase = Phase.PLACING
                # Reset to idle after place for next grasp cycle
                self.phase = Phase.IDLE

        # Transition GRASPING → TRANSPORTING if holding stable
        if self.phase == Phase.GRASPING and abs(gripper_delta) < 0.0005:
            self.phase = Phase.TRANSPORTING

        # Transition LIFTING → TRANSPORTING if accel stabilizes
        if self.phase == Phase.LIFTING and abs(z_accel) < cfg.vertical_accel_thresh * 0.5:
            self.phase = Phase.TRANSPORTING

        self.prev_gripper = gripper
        self.prev_accel_mag = accel_mag
        self.events.extend(new_events)
        return new_events

    def get_all_events(self) -> list[Event]:
        return list(self.events)

    def summary(self) -> dict[str, int]:
        counts = {"grasp": 0, "lift": 0, "place": 0}
        for ev in self.events:
            counts[ev.event_type] = counts.get(ev.event_type, 0) + 1
        return counts


def detect_events_batch(rows: list[dict[str, float]], episode_id: int = 0,
                        config: HSMConfig | None = None) -> list[Event]:
    """Convenience: run HSM over a full episode (list of row dicts)."""
    detector = HSMDetector(episode_id=episode_id, config=config)
    for row in rows:
        detector.step(row)
    return detector.get_all_events()
