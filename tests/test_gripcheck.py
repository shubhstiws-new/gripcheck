import polars as pl
import pytest

from hsm import detect_events_batch
from scorer import count_complete_cycles, pattern_score, score_episode

OPEN, CLOSED = 0.08, 0.03


def synthetic_episode(n_steps: int = 90, place: bool = True, lift: bool = True) -> pl.DataFrame:
    """A scripted pick-and-place: close gripper, lift, carry, lower and release.

    Events are spaced more than HSMConfig.min_steps_between_events (20) apart.
    """
    gripper, az, ax = [], [], []
    for t in range(n_steps):
        if t < 10:                      # approach, gripper open
            g, z, x = OPEN, 0.0, 0.0
        elif t < 15:                    # close with an acceleration spike
            g, z, x = OPEN - 0.01 * (t - 9), 0.0, 0.05
        elif t < 35:                    # settle grip
            g, z, x = CLOSED, 0.0, 0.0
        elif t < 40:                    # lift: upward acceleration, stable grip
            g, z, x = CLOSED, 0.05 if lift else 0.0, 0.0
        elif t < 60:                    # carry
            g, z, x = CLOSED, 0.0, 0.0
        elif t < 65 and place:          # lower and release
            g, z, x = CLOSED + 0.01 * (t - 59), -0.05, 0.0
        else:
            g, z, x = (OPEN if place else CLOSED), 0.0, 0.0
        gripper.append(g)
        az.append(z)
        ax.append(x)

    return pl.DataFrame({
        "timestep": list(range(n_steps)),
        "robot0_eef_pos_0": [0.002 * t for t in range(n_steps)],
        "robot0_eef_pos_1": [0.0] * n_steps,
        "robot0_eef_pos_2": [0.0] * n_steps,
        "robot0_gripper_width_0": gripper,
        "robot0_eef_accel_0": ax,
        "robot0_eef_accel_1": [0.0] * n_steps,
        "robot0_eef_accel_2": az,
    })


# ── event detection ─────────────────────────────────────────────────

def test_detects_full_cycle_in_order():
    events = detect_events_batch(synthetic_episode().iter_rows(named=True))
    assert [e.event_type for e in events] == ["grasp", "lift", "place"]
    assert [e.timestep for e in events] == [10, 35, 64]


def test_no_events_when_gripper_never_moves():
    df = synthetic_episode().with_columns(pl.lit(OPEN).alias("robot0_gripper_width_0"))
    assert detect_events_batch(df.iter_rows(named=True)) == []


# ── cycle counting and pattern scores ───────────────────────────────

@pytest.mark.parametrize(
    "events, cycles",
    [
        ([], 0),
        (["grasp", "lift", "place"], 1),
        (["grasp", "lift", "place", "grasp", "lift", "place"], 2),
        (["grasp", "grasp", "lift", "place"], 1),   # regrasp before lifting
        (["lift", "place"], 0),
        (["grasp", "place"], 0),
    ],
)
def test_count_complete_cycles(events, cycles):
    assert count_complete_cycles(events) == cycles


@pytest.mark.parametrize(
    "events, score",
    [
        (["grasp", "lift", "place"], 1.0),
        (["grasp", "lift"], 0.6),
        (["grasp"], 0.3),
        (["lift", "grasp"], 0.3),   # lift before any grasp does not count
        ([], 0.0),
    ],
)
def test_pattern_score(events, score):
    assert pattern_score(events) == score


# ── episode scoring ─────────────────────────────────────────────────

def test_clean_episode_scores_one():
    row = score_episode(synthetic_episode(), episode_id=7)
    assert row["episode_id"] == 7
    assert row["event_sequence"] == "grasp,lift,place"
    assert row["quality_score"] == 1.0
    assert row["quality_flags"] == ""


def test_missing_place_scores_partial():
    row = score_episode(synthetic_episode(place=False), episode_id=0)
    assert row["event_sequence"] == "grasp,lift"
    assert row["quality_score"] == 0.6


def test_short_episode_is_flagged_and_zeroed():
    row = score_episode(synthetic_episode(n_steps=40), episode_id=0)
    assert "too_short" in row["quality_flags"]
    assert row["quality_score"] == 0.0


def test_static_robot_is_flagged():
    df = synthetic_episode().with_columns(pl.lit(0.0).alias("robot0_eef_pos_0"))
    row = score_episode(df, episode_id=0)
    assert "no_motion" in row["quality_flags"]
    assert row["quality_score"] == 0.0
