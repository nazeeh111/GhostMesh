"""Role/pair/gain/selection contracts, not held-domain accuracy evidence."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import domain_amplitude_probe as a


def config():
    cfg = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/amplitude-person-e03-v1.json"
        ).read_text()
    )
    cfg.update(steps=4, checkpoint_every=2, adversarial_warmup_steps=1)
    return cfg


def fixture():
    rows = [
        ("E01", person, f"E01/{person}/{action}", end)
        for person in ["S01", "S02", "S03", "S05", "S06", "S07", "S08", "S10"]
        for action in ["A03", "A12"]
        for end in [39, 47, 55, 63]
    ]
    rows.extend([("E01", "S04", "E01/S04/A03", 39), ("E03", "S21", "E03/S21/A03", 39)])
    data = {
        key: np.array([r[i] for r in rows])
        for i, key in enumerate(("room", "person", "record", "end"))
    }
    data["train"] = (data["room"] == "E01") & (data["person"] != "S04")
    data["pose"] = (
        np.random.default_rng(3).normal(size=(len(rows), 17, 2)).astype(np.float32)
    )
    return data


def test_schedule_has_all_source_people_exact_lags_and_reproducible_gain_bounds():
    d = fixture()
    cfg = config()
    schedule, gains = a.make_schedule(d, cfg)
    again, gg = a.make_schedule(d, cfg)
    np.testing.assert_array_equal(schedule, again)
    np.testing.assert_array_equal(gains, gg)
    assert schedule.shape == (4, 8, 2) and d["train"][schedule].all()
    assert gains.shape == (4, 8, 3) and abs(gains).max() <= 6
    np.testing.assert_array_equal(
        d["record"][schedule[:, :, 0]], d["record"][schedule[:, :, 1]]
    )
    np.testing.assert_array_equal(
        d["end"][schedule[:, :, 0]] - d["end"][schedule[:, :, 1]], np.full((4, 8), 8)
    )
    assert all(len(set(d["person"][batch[:, 0]])) == 8 for batch in schedule)


def test_raw_reader_uses_only_its_source_role_and_causal_frame_support():
    rf = np.random.default_rng(4).normal(30, 2, (60, 6, 114, 10)).astype(np.float32)
    reader = a.RawReader(rf, np.array([39, 47]), np.array([True, False]))
    before = reader(np.array([0]))
    rf[40:] *= 100
    np.testing.assert_array_equal(reader(np.array([0])), before)
    with pytest.raises(ValueError, match="role"):
        reader(np.array([1]))
    with pytest.raises(ValueError, match="role"):
        reader(np.array([-1]))


def test_pose_statistics_ignore_all_nontraining_labels():
    d = fixture()
    before = a.pose_statistics(d, config())
    d["pose"][~d["train"]] = np.nan
    after = a.pose_statistics(d, config())
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_source_selection_honors_motion_then_error_then_earlier_step_and_rejects_undefined():
    def candidate(step, r2, nje):
        return dict(step=step, zero=dict(pooled=dict(motion_r2=r2, nje=nje)))

    rows = [
        candidate(0, 0.0, 0.1),
        candidate(500, 0.1, 0.2),
        candidate(1000, 0.1, 0.1),
        candidate(1500, 0.1, 0.1),
    ]
    assert a.select(rows, "zero")["step"] == 1000
    with pytest.raises(ValueError, match="finite"):
        a.select([candidate(0, None, 0.1)], "zero")


def test_invalid_recipe_cannot_change_source_split_or_recovery_range():
    cfg = config()
    a.validate_config(cfg)
    cfg["source_room"] = "E03"
    with pytest.raises(ValueError):
        a.validate_config(cfg)
    cfg = config()
    cfg["steps"] = 10000
    with pytest.raises(ValueError):
        a.validate_config(cfg)
