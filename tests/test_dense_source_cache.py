import copy

import pytest

from scripts.build_dense_source_cache import source_records


def receipt():
    names = [
        f"E{room:02}/S{person:02}/A{action:02}"
        for room, first in [(1, 1), (2, 11)]
        for person in range(first, first + 10)
        for action in range(1, 28)
    ]
    return dict(
        config=dict(magnitude_units="log_magnitude_db", phase_units="radians"),
        frames=len(names) * 297,
        records=[dict(id=n, offset=i * 297, frames=297) for i, n in enumerate(names)],
    )


def test_dense_source_builder_rejects_units_overlap_and_held_room_injection():
    base = receipt()
    assert len(source_records(base)) == 540
    wrong = copy.deepcopy(base)
    wrong["config"]["magnitude_units"] = "linear"
    with pytest.raises(ValueError, match="units"):
        source_records(wrong)
    wrong = copy.deepcopy(base)
    wrong["records"][1]["offset"] = 0
    with pytest.raises(ValueError, match="overlap"):
        source_records(wrong)
    wrong = copy.deepcopy(base)
    wrong["records"][0]["id"] = "E03/S21/A01"
    with pytest.raises(ValueError, match="Exactsource"):
        source_records(wrong)
