"""Source access, frame joins and qualification checks with software fixtures."""

import copy
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
import pytest

from scripts import domain_shape_probe as s


def fixture():
    rng = np.random.default_rng(2)
    n = 20
    meta = dict(
        room=np.repeat(["E01", "E03", "E01", "E03", "E02"], 4),
        person=np.repeat(["S01", "S21", "S04", "S24", "S11"], 4),
        record=np.repeat(
            ["E01/S01/A01", "E03/S21/A01", "E01/S04/A01", "E03/S24/A01", "E02/S11/A01"],
            4,
        ),
        action=np.repeat("A01", n),
        end=np.tile([39, 47, 55, 63], 5),
    )
    return (
        rng.normal(size=(n, 6)),
        rng.normal(size=(n, 136)),
        meta,
        np.arange(n) < 8,
        (np.arange(n) >= 8) & (np.arange(n) < 16),
    )


def test_shape_coefficients_use_only_train_and_never_access_e02_values():
    x, y, meta, tr, va = fixture()

    class Tracked:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, ids):
            assert (tr | va)[ids].all(), "E02 values accessed"
            return self.value[ids]

    states, _, _, _ = s.fit_source(Tracked(x), Tracked(y), meta, tr, va)
    x[va] += 100
    y[va] -= 100
    x[~(tr | va)], y[~(tr | va)] = np.nan, np.nan
    after, _, _, _ = s.fit_source(Tracked(x), Tracked(y), meta, tr, va)
    for name, state in states.items():
        for key, value in state.items():
            np.testing.assert_array_equal(value, after[name][key])
    wrong = tr.copy()
    wrong[-1] = True
    with pytest.raises(ValueError, match="No E02"):
        s.fit_source(Tracked(x), Tracked(y), meta, wrong, va)
    with pytest.raises(ValueError, match="people differ"):
        s.fit_source(x, y, meta, va, tr)


def test_shape_metrics_distinguish_motion_and_absolute_offset_without_query_fitting():
    _, y, meta, _, va = fixture()
    y = y[va]
    records = meta["record"][va]
    ends = meta["end"][va]
    perfect = s.score(y, y, records, ends)
    shifted = s.score(y + 10, y, records, ends)
    assert perfect["mse"] == 0 and perfect["record_centered_shape_r2"] == pytest.approx(
        1
    )
    assert shifted["record_centered_shape_r2"] == pytest.approx(1) and shifted[
        "mse"
    ] == pytest.approx(100)
    assert shifted["lag8_zero_reference_delta_r2"] == pytest.approx(1)
    static = s.score(np.zeros_like(y), y, records, ends)
    assert static["record_centered_shape_r2"] == pytest.approx(0)
    assert static["lag8_zero_reference_delta_r2"] == pytest.approx(0)
    flat = s.score(np.ones_like(y), np.ones_like(y), records, ends)
    assert (
        flat["record_centered_shape_r2"] is None
        and flat["lag8_zero_reference_delta_r2"] is None
    )


def annotation_fixture(tmp_path):
    annotations = tmp_path / "annotations"
    records = []
    entries = []
    names = []
    for room, first in [("E01", 1), ("E03", 21)]:
        for person in range(first, first + 10):
            for action in range(1, 28):
                name = f"{room}/S{person:02}/A{action:02}/ground_truth.npy"
                names.append(name)
                path = annotations / name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = b"SOFTWARE TEST ONLY nonnumeric fixture"
                path.write_bytes(raw)
                records.append(
                    dict(
                        path=name,
                        bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
                entries.append(
                    dict(name=name, size=len(raw), crc=zlib.crc32(raw) & 0xFFFFFFFF)
                )
    receipt = tmp_path / "receipt.json"
    inventory = tmp_path / "inventory.json"
    receipt.write_text(
        json.dumps(
            dict(
                status="CRC_verified_source_only_skeletons",
                rooms=["E01", "E03"],
                records=records,
            )
        )
    )
    inventory.write_text(json.dumps(entries))
    meta = dict(
        room=np.array([x.split("/")[0] for x in names]),
        record=np.array([x.removesuffix("/ground_truth.npy") for x in names]),
        end=39 + 8 * (np.arange(540) % 33),
    )
    return annotations, receipt, inventory, meta, names


def test_all_annotation_crc_checks_precede_numeric_decode(tmp_path, monkeypatch):
    annotations, receipt, inventory, meta, names = annotation_fixture(tmp_path)
    (annotations / names[0]).write_bytes(b"corrupted fixture bytes")
    monkeypatch.setattr(
        np,
        "load",
        lambda *a, **k: pytest.fail("CRC must reject before annotation decoding"),
    )
    with pytest.raises(ValueError, match="CRC/hash"):
        s.load_targets(meta, np.ones(540, dtype=bool), annotations, receipt, inventory)


def test_real_loader_frame_selection_uses_zero_based_endpoint_and_matching_record(
    tmp_path, monkeypatch
):
    annotations, receipt, inventory, meta, names = annotation_fixture(tmp_path)
    indices = {name: i for i, name in enumerate(names)}
    base = np.tile(np.arange(51).reshape(17, 3).astype(np.float32), (297, 1, 1))

    def labels(path, **kwargs):
        i = indices[str(Path(path).relative_to(annotations))]
        value = base.copy()
        value[:, 16, 0] += np.arange(297, dtype=np.float32) / 10 + i / 20
        return value

    monkeypatch.setattr(np, "load", labels)
    ids, target = s.load_targets(
        meta, np.ones(540, dtype=bool), annotations, receipt, inventory
    )
    np.testing.assert_array_equal(ids, np.arange(540))
    for i in [0, 1, 269, 270, 539]:
        np.testing.assert_array_equal(
            target[i],
            s.canonical_distances(labels(annotations / names[i])[meta["end"][i]]),
        )


def test_e02_annotation_mask_refused_before_receipt_or_file_access(tmp_path):
    data = dict(
        room=np.array(["E02"]), record=np.array(["E02/S11/A01"]), end=np.array([39])
    )
    with pytest.raises(ValueError, match="Only E01/E03"):
        s.load_targets(
            data,
            np.ones(1, dtype=bool),
            tmp_path,
            tmp_path / "absent",
            tmp_path / "absent",
        )


def gate_fixture():
    def selected(r2, mse):
        row = dict(
            record_centered_shape_r2=r2, lag8_zero_reference_delta_r2=r2, mse=mse
        )
        return dict(
            aligned=dict(
                pooled=row,
                per_room={x: row.copy() for x in ["E01", "E03"]},
                per_person={x: row.copy() for x in ["S04", "S09", "S24", "S29"]},
            ),
            shuffled=dict(pooled=dict(row, record_centered_shape_r2=-0.1)),
        )

    results = {
        a: dict(
            selections={f: selected(0.1, 0.2) for f in ["ordinary", "fixed_effects"]}
        )
        for a in s.ARMS
    }
    results["masked_temporal"]["selections"]["fixed_effects"] = selected(0.3, 0.1)
    controls = {
        x: dict(pooled=dict(record_centered_shape_r2=0, mse=0.3))
        for x in ["source_mean", "source_clock"]
    }
    return results, controls


def test_gate_compares_every_selected_ordinary_and_fixed_effects_control():
    results, controls = gate_fixture()
    assert all(s.decide_gate(results, controls).values())
    changed = copy.deepcopy(results)
    changed["random_frozen"]["selections"]["ordinary"]["aligned"]["pooled"][
        "record_centered_shape_r2"
    ] = 0.4
    assert not s.decide_gate(changed, controls)["beats_control_motion"]
    results["masked_temporal"]["selections"]["fixed_effects"]["aligned"]["pooled"][
        "record_centered_shape_r2"
    ] = None
    gate = s.decide_gate(results, controls)
    assert (
        not gate["useful_shape_motion"]
        and not gate["beats_control_motion"]
        and not gate["shuffle_worse"]
    )


def test_existing_or_nested_output_refused_before_data_reads(tmp_path):
    with pytest.raises(FileExistsError):
        s.run(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        s.run(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / "new-study")
