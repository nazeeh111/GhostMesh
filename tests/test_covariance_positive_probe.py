from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

spec = spec_from_file_location(
    "covariance_positive",
    Path(__file__).resolve().parents[1] / "scripts/domain_covariance_positive_probe.py",
)
probe = module_from_spec(spec)
spec.loader.exec_module(probe)


def test_lag32_plus_eight_frame_context_requires_forty_raw_frames():
    rng = np.random.default_rng(8)
    rf = rng.normal(size=(297, 6, 114, 10))
    rf[:, :3] += 40
    y = rng.uniform(size=(297, 17, 2))
    supports = []
    for split, block in probe.CONFIG["raw_blocks_half_open"].items():
        values, target, ends, support, _ = probe.block_examples(rf, y, *block)
        assert len(target) == probe.CONFIG["counts"][split]
        assert np.array_equal(support[:, -1], ends)
        assert support.min() >= block[0] and support.max() < block[1]
        assert support.shape[1] == 40
        assert all(value.shape[0] == len(target) for value in values.values())
        supports.append(set(support.ravel()))
    assert (
        not supports[0] & supports[1]
        and not supports[0] & supports[2]
        and not supports[1] & supports[2]
    )
