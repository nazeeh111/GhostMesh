"""Fresh-cache RF guards and exact allocation, without annotation access."""

import numpy as np
import pytest

from scripts import build_moment_cache as c


def raw():
    rng = np.random.default_rng(9)
    # Independent complex receiver samples produce well-conditioned covariance.
    return np.concatenate(
        (
            rng.normal(30, 2, (297, 3, 114, 10)),
            rng.uniform(-np.pi, np.pi, (297, 3, 114, 10)),
        ),
        axis=1,
    ).astype(np.float32)


def test_forty_frame_guard_purges_every_affected_endpoint_and_never_bridges():
    x = raw()
    baseline, rejected = c.retained_endpoints(x)
    np.testing.assert_array_equal(baseline, np.arange(39, 297, 8))
    assert len(rejected) == 0
    x[80] = np.nan
    ends, rejected = c.retained_endpoints(x)
    expected = [e for e in baseline if not e - 39 <= 80 <= e]
    np.testing.assert_array_equal(ends, expected)
    np.testing.assert_array_equal(rejected, [80])


def test_covariance_rank_failure_is_not_repaired_with_identity_noise():
    x = raw()
    x[80, :3] = 30
    x[80, 3:] = 0
    ends, rejected = c.retained_endpoints(x)
    assert 80 in rejected and all(not e - 39 <= 80 <= e for e in ends)


def test_output_is_preserved_before_any_input_read(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    p = out / "keep"
    p.write_text("original")
    with pytest.raises(ValueError, match="Existing"):
        c.build(tmp_path / "nonexistent-raw", out)
    assert p.read_text() == "original"
