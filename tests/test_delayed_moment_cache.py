import numpy as np
import pytest

from scripts.build_delayed_moment_cache import history_indices, output_endpoints


def test_nominal_delay_and_complete_record_boundaries():
    ends = output_endpoints(np.array([], dtype=np.int64))
    assert np.array_equal(ends, np.arange(39, 293, 8))
    histories = history_indices(ends)
    assert histories.shape == (32, 32)
    assert histories[0, 0] == 12 and histories[-1, -1] == 291
    assert np.array_equal(histories[:, -1], ends + 4)


def test_common_guard_rejects_bad_frame_inside_future_delay_and_old_history():
    first = 39
    assert first not in output_endpoints(np.array([first + 4]))
    assert first not in output_endpoints(np.array([first - 39]))
    assert first in output_endpoints(np.array([first + 5]))
    assert len(output_endpoints(np.arange(297))) == 0


def test_invalid_clock_or_frame_roles_are_rejected():
    for bad in [np.array([40]), np.array([295]), np.array([39.0])]:
        with pytest.raises(ValueError):
            history_indices(bad)
    with pytest.raises(ValueError):
        output_endpoints(np.array([-1]))
