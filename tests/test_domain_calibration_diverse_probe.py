"""Coverage and deterministic endpoint checks for Diverse108, no real data."""
from hashlib import sha256
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("domain_calibration_diverse_probe", Path(__file__).parents[1] / "scripts/domain_calibration_diverse_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_reference_plan_is_record_only_and_causal_range():
    for room, offset in zip(probe.ROOMS, (0, 10, 20)):
        for person in range(offset + 1, offset + 6):
            for action in range(1, 28):
                record = f"{room}/S{person:02}/A{action:02}"
                expected = 31 + int(sha256(("ghostmesh-diverse108-v1|" + record).encode()).hexdigest(), 16) % 265
                assert probe.reference_end(record) == expected
                assert 31 <= expected <= 295
    with pytest.raises(ValueError, match="Forbidden"):
        probe.reference_end("E04/S36/A01")


def test_every_rotation_has_four_people_and_all_actions():
    subjects = np.repeat([f"S{i:02}" for i in range(1, 6)], 27)
    actions = np.tile(np.arange(1, 28), 5)
    held_count = np.zeros(135, dtype=int)
    for person in sorted(set(subjects)):
        mask = probe.rotation_mask(subjects, person)
        assert mask.sum() == 108
        assert person not in set(subjects[mask])
        for source in set(subjects[mask]):
            assert set(actions[mask & (subjects == source)]) == set(range(1, 28))
        held_count += ~mask
    np.testing.assert_array_equal(held_count, np.ones(135, dtype=int))
    with pytest.raises(ValueError, match="Unexpected"):
        probe.rotation_mask(subjects, "S06")
