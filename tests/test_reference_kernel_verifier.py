"""Synthetic verifier fixtures; no scientific accuracy claims."""

import copy

import numpy as np
import pytest
from scipy.linalg.blas import dgemm

from ghostmesh.reference_kernel import ReferenceRBFKernel
from scripts import domain_reference_kernel_probe as study
from scripts.verify_reference_kernel_probe import (
    audit_duals,
    cross_kernel,
    reference_state,
    same_tree,
    verify,
)


def fixture():
    rng = np.random.default_rng(407)
    raw = rng.normal(size=(11, 5))
    ids = np.arange(8)
    labels = rng.normal(size=(8, 17, 2))
    base = rng.normal(size=(8, 17, 2))
    mean = rng.normal(size=(17, 2))
    permutation = np.roll(np.arange(8), 1)
    kernel = ReferenceRBFKernel.from_reference(raw[ids])
    state, _ = study.adaptations(kernel, labels, base, mean, permutation)
    return raw, ids, labels, base, mean, permutation, kernel, state


def test_saved_duals_match_independent_dense_system_and_reject_tampering():
    raw, ids, labels, base, mean, permutation, kernel, state = fixture()
    gram = kernel.cross_kernel(raw[ids])
    residual, difference, curves = audit_duals(
        gram, labels, base, mean, permutation, state
    )
    assert max(residual, difference) < 1e-12
    assert all(len(values) == 5 for values in curves.values())
    altered = {key: value.copy() for key, value in state.items()}
    altered["residual_krr"][2, 0, 0] += 0.1
    with pytest.raises(ValueError, match="Saved dual"):
        audit_duals(gram, labels, base, mean, permutation, altered)
    with pytest.raises(ValueError, match="Saved dual"):
        audit_duals(gram, labels + 1, base, mean, permutation, state)
    altered = {key: value.copy() for key, value in state.items()}
    altered["constant_residual"][0] += 1
    with pytest.raises(ValueError, match="Constant residual"):
        audit_duals(gram, labels, base, mean, permutation, altered)


def test_saved_reference_statistics_are_reference_only_and_replay_exactly():
    raw, ids, _, _, _, _, kernel, state = fixture()
    reference, gram = reference_state(raw, ids, kernel.state_dict())
    np.testing.assert_array_equal(gram, kernel.cross_kernel(raw[ids]))
    before = cross_kernel(raw, np.arange(8, 11), reference)
    np.testing.assert_array_equal(before, kernel.cross_kernel(raw[8:]))
    np.testing.assert_array_equal(
        dgemm(1.0, before, state["residual_krr"][0]),
        kernel.predict(raw[8:], state["residual_krr"][0]),
    )
    changed = raw.copy()
    changed[8:] += 1000
    unchanged, _ = reference_state(changed, ids, kernel.state_dict())
    np.testing.assert_array_equal(unchanged["mean"], reference["mean"])
    polluted = kernel.state_dict()
    polluted["input_mean"] = raw.mean(0)
    with pytest.raises(ValueError, match="Reference-only"):
        reference_state(raw, ids, polluted)


def test_nested_choice_excludes_held_room_and_checks_complete_json_trees():
    cases = []
    for source in study.ROOMS:
        for target in study.ROOMS:
            if source != target:
                curves = {
                    method: [
                        dict(**choice, source_validation_nje=float(index))
                        for index, choice in enumerate(study.candidate_grid(method))
                    ]
                    for method in study.METHODS
                }
                cases.append(dict(source=source, target=target, curves=curves))
    expected = study.select_outer(cases, "E01")
    changed = copy.deepcopy(cases)
    for case in changed:
        if "E01" in (case["source"], case["target"]):
            for curve in case["curves"].values():
                for row in curve:
                    row["source_validation_nje"] += 1e9
    same_tree(study.select_outer(changed, "E01"), expected)
    assert set(expected["residual_krr"]["source_directions"]) == {
        "E02->E03",
        "E03->E02",
    }
    with pytest.raises(ValueError):
        same_tree({"metric": 1.0, "extra": 0}, {"metric": 1.0})
    with pytest.raises(ValueError):
        same_tree({"metric": 1.01}, {"metric": 1.0})


def test_existing_verification_is_preserved_before_any_input_read(tmp_path):
    report = tmp_path / "verification.json"
    report.write_text("preserve me")
    with pytest.raises(FileExistsError):
        verify(None, None, None, tmp_path)
    assert report.read_text() == "preserve me"
