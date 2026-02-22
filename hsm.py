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
    # Grasp detection – primary trigger is rapid gripper closure
    gripper_closing_delta: float = -0.002  # per-step gripper narrowing rate
    gripper_close_thresh: float = 0.075    # gripper width below this during closure
    accel_spike_thresh: float = 0.003      # EEF accel mag (boosts confidence, not required)

    # Lift detection – upward z-accel while gripper is closed
    vertical_accel_thresh: float = 0.002   # upward accel (z-axis)
    stable_hold_width: float = 0.070       # gripper width for "holding"

    # Place detection – primary trigger is rapid gripper opening
    gripper_opening_delta: float = 0.002   # per-step gripper widening rate
    gripper_open_thresh: float = 0.070     # gripper width above this during opening
    decel_thresh: float = -0.002           # negative z-accel (boosts confidence)

    # Cooldown: min steps between same-type events within an episode
    min_steps_between_events: int = 20


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
        self._last_event_step = -999

    def step_precomputed(
        self,
        accel_mag: float,
        gripper: float,
        gripper_delta: float,
        z_accel: float,
        timestep: int | None = None,
    ) -> list[Event]:
        """Fast path: accept pre-computed signal values, skip dict scanning."""
        ts = timestep if timestep is not None else self._step_count
        self._step_count += 1
        new_events: list[Event] = []

        cfg = self.cfg
        cooldown_ok = (self._step_count - self._last_event_step) > cfg.min_steps_between_events

        if self.phase in (Phase.IDLE, Phase.APPROACHING, Phase.RELEASING) and cooldown_ok:
            if (gripper_delta < cfg.gripper_closing_delta
                    and gripper < cfg.gripper_close_thresh):
                base_conf = min(1.0, abs(gripper_delta) / abs(cfg.gripper_closing_delta * 3))
                accel_bonus = 0.2 if accel_mag > cfg.accel_spike_thresh else 0.0
                confidence = min(1.0, base_conf + accel_bonus)
                ev = Event(
                    event_type="grasp", timestep=ts, episode_id=self.episode_id,
                    phase_from=self.phase.name, phase_to=Phase.GRASPING.name,
                    confidence=confidence,
                    details={"accel_mag": accel_mag, "gripper": gripper,
                             "gripper_delta": gripper_delta},
                )
                new_events.append(ev)
                self.phase = Phase.GRASPING
                self._last_event_step = self._step_count

        if self.phase in (Phase.GRASPING, Phase.TRANSPORTING) and cooldown_ok:
            if (z_accel > cfg.vertical_accel_thresh
                    and gripper < cfg.stable_hold_width):
                confidence = min(1.0, z_accel / (cfg.vertical_accel_thresh * 3))
                ev = Event(
                    event_type="lift", timestep=ts, episode_id=self.episode_id,
                    phase_from=self.phase.name, phase_to=Phase.LIFTING.name,
                    confidence=confidence,
                    details={"z_accel": z_accel, "gripper": gripper},
                )
                new_events.append(ev)
                self.phase = Phase.LIFTING
                self._last_event_step = self._step_count

        if self.phase in (Phase.LIFTING, Phase.TRANSPORTING) and cooldown_ok:
            if (gripper_delta > cfg.gripper_opening_delta
                    and gripper > cfg.gripper_open_thresh):
                base_conf = min(1.0, gripper_delta / (cfg.gripper_opening_delta * 3))
                decel_bonus = 0.2 if z_accel < cfg.decel_thresh else 0.0
                confidence = min(1.0, base_conf + decel_bonus)
                ev = Event(
                    event_type="place", timestep=ts, episode_id=self.episode_id,
                    phase_from=self.phase.name, phase_to=Phase.PLACING.name,
                    confidence=confidence,
                    details={"z_accel": z_accel, "gripper": gripper,
                             "gripper_delta": gripper_delta},
                )
                new_events.append(ev)
                self.phase = Phase.IDLE
                self._last_event_step = self._step_count

        if self.phase == Phase.GRASPING and abs(gripper_delta) < 0.0005:
            self.phase = Phase.TRANSPORTING
        if self.phase == Phase.LIFTING and abs(z_accel) < cfg.vertical_accel_thresh * 0.5:
            self.phase = Phase.TRANSPORTING

        self.prev_gripper = gripper
        self.prev_accel_mag = accel_mag
        self.events.extend(new_events)
        return new_events

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
        cooldown_ok = (self._step_count - self._last_event_step) > cfg.min_steps_between_events

        if self.phase in (Phase.IDLE, Phase.APPROACHING, Phase.RELEASING) and cooldown_ok:
            # Detect GRASP: rapid gripper closure is the primary trigger
            if (gripper_delta < cfg.gripper_closing_delta
                    and gripper < cfg.gripper_close_thresh):
                # Accel boosts confidence but is not required
                base_conf = min(1.0, abs(gripper_delta) / abs(cfg.gripper_closing_delta * 3))
                accel_bonus = 0.2 if accel_mag > cfg.accel_spike_thresh else 0.0
                confidence = min(1.0, base_conf + accel_bonus)
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
                self._last_event_step = self._step_count

        if self.phase in (Phase.GRASPING, Phase.TRANSPORTING) and cooldown_ok:
            # Detect LIFT: upward z-accel while gripper is closed
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
                self._last_event_step = self._step_count

        if self.phase in (Phase.LIFTING, Phase.TRANSPORTING) and cooldown_ok:
            # Detect PLACE: rapid gripper opening is the primary trigger
            if (gripper_delta > cfg.gripper_opening_delta
                    and gripper > cfg.gripper_open_thresh):
                base_conf = min(1.0, gripper_delta / (cfg.gripper_opening_delta * 3))
                decel_bonus = 0.2 if z_accel < cfg.decel_thresh else 0.0
                confidence = min(1.0, base_conf + decel_bonus)
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
                self.phase = Phase.IDLE
                self._last_event_step = self._step_count

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
