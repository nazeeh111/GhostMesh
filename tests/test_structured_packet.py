"""SOFTWARE TEST ONLY: model geometry/gradient contracts and analytical motion."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from ghostmesh.structured_packet import EDGES, StructuredPacketPose, structured_loss

SPEC = importlib.util.spec_from_file_location("structured_probe_test", Path(__file__).resolve().parents[1] / "scripts/domain_structured_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("channels", [3, 7])
def test_matched_capacity_source_initialization_and_finite_backward(channels):
    torch.manual_seed(2027)
    torch.set_num_threads(1)
    y = torch.rand(4, 17, 2) * .4 + .2
    x = torch.randn(2, 8, channels, 114, 10)
    counts = []
    for factorized in (False, True):
        model = StructuredPacketPose(factorized, channels=channels)
        model.initialize_from_source(y)
        out = model(x)
        np.testing.assert_allclose(out["pose"].detach(), np.broadcast_to(y.mean(0), (2, 17, 2)), atol=1e-7)
        loss = structured_loss(out, y[:2], factorized)
        loss.backward()
        assert torch.isfinite(loss)
        counts.append(sum(p.numel() for p in model.parameters()))
    assert counts[0] == counts[1] < 1000000
    assert len(EDGES) == 19


def test_root_translation_does_not_supervise_structure_encoder_decoder():
    torch.manual_seed(2027)
    torch.set_num_threads(1)
    y = torch.rand(4, 17, 2) * .4 + .2
    x = torch.randn(4, 8, 3, 114, 10)
    model = StructuredPacketPose(True)
    model.initialize_from_source(y)
    grads = []
    for shift in (0., .1):
        model.zero_grad(set_to_none=True)
        structured_loss(model(x), y + shift, True).backward()
        grads.append({k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None})
    for key in grads[0]:
        if not key.startswith("root_head"):
            torch.testing.assert_close(grads[0][key], grads[1][key], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(grads[0]["root_head.bias"], grads[1]["root_head.bias"])


def test_input_rejects_future_or_wrong_sensor_layout():
    model = StructuredPacketPose()
    with pytest.raises(ValueError, match="causal windows"):
        model(torch.rand(1, 9, 3, 114, 10))
    with pytest.raises(ValueError, match="causal windows"):
        model(torch.rand(1, 8, 7, 114, 10))


def test_motion_metric_perfect_static_and_inverted_known_traces():
    time = np.arange(8, dtype=float)[:, None, None]
    skeleton = np.arange(34, dtype=float).reshape(1, 17, 2) * .001
    y = skeleton + np.sin(time) * np.arange(1, 18)[None, :, None] * .001
    sequence = np.array(["SOFTWARE_TEST_ONLY"] * len(y))
    mean = np.broadcast_to(y.mean(0), y.shape)
    perfect, static, inverted = [probe.motion_metrics(p, y, sequence) for p in (y, mean, 2 * mean - y)]
    for component in ("absolute", "root", "relative"):
        assert perfect[component]["within_record_r2"] == pytest.approx(1)
        assert static[component]["within_record_r2"] == pytest.approx(0)
        assert inverted[component]["within_record_r2"] == pytest.approx(-3)


def test_phase_ablation_preserves_amplitude_and_original_circular_channels():
    amplitude = np.arange(2 * 8 * 3 * 114 * 10, dtype=np.float32).reshape(2, 8, 3, 114, 10)
    phase = np.full((2, 8, 4, 114, 10), -.5, dtype=np.float32)
    target = np.zeros((2, 17, 2), dtype=np.float32)
    x, _ = probe.RawSet(amplitude, target, np.array([1]), False, phase)[0]
    zero, _ = probe.RawSet(amplitude, target, np.array([1]), True, phase)[0]
    np.testing.assert_array_equal(x[:, :3], amplitude[1])
    np.testing.assert_array_equal(x[:, 3:], phase[1])
    np.testing.assert_array_equal(zero[:, :3], amplitude[1])
    assert torch.count_nonzero(zero[:, 3:]) == 0
    assert (phase == -.5).all()
