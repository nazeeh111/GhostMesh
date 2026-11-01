import numpy as np
import pytest

from scripts.domain_packet_transfer_probe import fit_readouts
from scripts.verify_packet_transfer_probe import verify, verify_systems


def fixture():
    rng = np.random.default_rng(91)
    design = rng.normal(size=(40, 7))
    y = rng.normal(size=(40, 104)) * 0.01
    y[:, :34] += 0.5
    train = np.arange(40) < 20
    val = (np.arange(40) >= 20) & (np.arange(40) < 30)
    model, _ = fit_readouts(design, y, train, val)
    return design, y, model, train, val


def test_independent_direct_equations_and_validation_reject_wrong_weights():
    z, y, model, train, val = fixture()
    residual, direct, criteria = verify_systems(z, y, model, train, val)
    assert max(residual, direct) < 1e-12
    assert len(criteria) == 4 and all(len(x) == 5 for x in criteria.values())
    altered = {key: value.copy() for key, value in model.items()}
    altered["weights"][2, 0, 0] += 0.01
    with pytest.raises(ValueError, match="direct ridge solve"):
        verify_systems(z, y, altered, train, val)


def test_validation_rejects_bad_selection_overlapping_roles_and_ignores_held_labels():
    z, y, model, train, val = fixture()
    before = verify_systems(z, y, model, train, val)
    changed = y.copy()
    changed[~(train | val)] += 1e9
    assert verify_systems(z, changed, model, train, val) == before
    altered = {key: value.copy() for key, value in model.items()}
    altered["selected_index"][0] = (altered["selected_index"][0] + 1) % 5
    with pytest.raises(ValueError, match="minimize source validation"):
        verify_systems(z, y, altered, train, val)
    with pytest.raises(ValueError, match="disjoint source roles"):
        verify_systems(z, y, model, train, train)


def test_existing_report_is_refused_before_any_input_read(tmp_path):
    report = tmp_path / "already.json"
    report.write_text("preserve me")
    with pytest.raises(FileExistsError):
        verify("raw", None, None, None, report)
    assert report.read_text() == "preserve me"
