"""Metric/gradient evidence for the training-capacity diagnostic, no real data."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

spec = importlib.util.spec_from_file_location('overfit', Path(__file__).resolve().parents[1]/'scripts/domain_rf_overfit_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_motion_score_separates_mean_perfect_and_reversed_motion():
    rng = np.random.default_rng(1)
    y = rng.uniform(.2, .8, (16, 17, 2))
    static = np.broadcast_to(y.mean(0), y.shape)
    for component in ('absolute', 'root', 'relative'):
        assert abs(probe.score(static, y)['motion'][component]['within_record_r2']) < 1e-12
        assert probe.score(y, y)['motion'][component]['within_record_r2'] == 1
        inverted = 2*y.mean(0)-y
        assert abs(probe.score(inverted, y)['motion'][component]['within_record_r2']+3) < 1e-12
    assert probe.score(y, y)['nje'] == 0


def test_zero_initialized_readouts_deliberately_delay_encoder_gradient_one_update():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = probe.RFPose()
    y = torch.rand(2, 2, 17, 2)*.4+.2
    model.initialize_from_source(y)
    body, root = torch.randn(2, 2, 7, 114, 10), torch.randn(2, 2, 342)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    evidence = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        terms = probe.loss_components(model(body, root, return_sequence=True), y)
        loss = terms['relative_geometry']+terms['root_geometry']+.1*terms['bone']+.1*terms['uncertainty']
        loss.backward()
        evidence.append(probe.gradient_evidence(model))
        probe.clip_groups(model)
        optimizer.step()
    assert evidence[0]['body_encoder'] == evidence[0]['root_encoder'] == 0
    assert evidence[0]['body_head'] > 0 and evidence[0]['root_head'] > 0
    assert evidence[1]['body_encoder'] > 0 and evidence[1]['root_encoder'] > 0


def test_input_control_remaps_individual_time_frames_across_blocks():
    controls_spec = importlib.util.spec_from_file_location('controls', Path(__file__).resolve().parents[1]/'scripts/domain_rf_overfit_controls.py')
    controls = importlib.util.module_from_spec(controls_spec)
    controls_spec.loader.exec_module(controls)
    x = torch.arange(12).reshape(2, 3, 2)
    order = np.array([5, 0, 4, 1, 3, 2])
    changed = controls.remap_frames(x, order)
    assert changed.shape == x.shape
    torch.testing.assert_close(changed.reshape(6, 2), x.reshape(6, 2)[order])
    torch.testing.assert_close(controls.remap_frames(x, np.arange(6)), x)
