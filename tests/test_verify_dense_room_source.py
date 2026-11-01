import copy

from scripts.verify_dense_room_source import gate


def test_dense_gate_rejects_one_room_reversal_despite_positive_pooled_motion():
    metric = dict(motion_r2=0.12, delta_r2=0.1, nje=0.05, pck05=0.7, pck20=0.96)
    primary = dict(
        pooled=metric,
        per_room={room: copy.deepcopy(metric) for room in ["E01", "E02"]},
        per_person={
            person: copy.deepcopy(metric) for person in ["S04", "S09", "S14", "S19"]
        },
        shuffled=dict(nje=0.08),
    )
    controls = [dict(pooled=dict(nje=0.06, pck05=0.6, pck20=0.94))]
    assert all(gate(primary, controls).values())
    primary["per_room"]["E02"]["motion_r2"] = -0.01
    result = gate(primary, controls)
    assert not result["both_source_rooms_positive"]
    assert result["body_motion_useful"] and result["nje_beats_controls"]
    primary["per_person"]["S04"]["motion_r2"] = -0.1
    primary["per_person"]["S09"]["motion_r2"] = -0.1
    assert not gate(primary, controls)["three_of_four_people_positive"]
