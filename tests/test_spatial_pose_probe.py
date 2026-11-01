"""Synthetic software checks, never evidence of CSI pose accuracy."""

import numpy as np
import pytest
import torch

from ghostmesh.spatial_pose import SpatialPose, pose_objective
from scripts import domain_spatial_token_probe as probe


def model(global_pool=False):
    torch.manual_seed(2027)
    return SpatialPose(torch.zeros(17, 2), torch.ones(17, 2), global_pool=global_pool)


def test_joint_outputs_have_own_hip_origin_and_identical_input_zero_difference():
    network = model().eval()
    with torch.no_grad():
        network.output[-1].weight.normal_(std=0.1)
        tokens = torch.randn(3, 95, 48)
        output = network(tokens)
        assert output.shape == (3, 17, 2)
        torch.testing.assert_close(output[:, [11, 12]].mean(1), torch.zeros(3, 2))
        torch.testing.assert_close(
            network(tokens) - network(tokens), torch.zeros_like(output), atol=0, rtol=0
        )


def test_global_ablation_has_identical_parameter_count_and_uses_only_token_mean():
    spatial, global_head = model(), model(True)
    assert sum(x.numel() for x in spatial.parameters()) == sum(
        x.numel() for x in global_head.parameters()
    )
    for key, value in spatial.state_dict().items():
        torch.testing.assert_close(value, global_head.state_dict()[key], atol=0, rtol=0)
    with torch.no_grad():
        global_head.output[-1].weight.normal_(std=0.1)
        tokens = torch.randn(3, 95, 48)
        reference = global_head(tokens)
        # Preserve the mean while relocating a strong local feature.
        changed = tokens.clone()
        changed[:, 0] += 20
        changed[:, 1] -= 20
        torch.testing.assert_close(
            global_head(changed), reference, atol=2e-6, rtol=2e-6
        )


def test_motion_objective_matches_analytic_equation_and_endpoint_swap():
    torch.manual_seed(3)
    current, previous, y, z = [torch.randn(5, 17, 2) for _ in range(4)]
    scale, ds = torch.rand(17, 2) + 0.1, torch.rand(17, 2) + 0.1
    absolute = 0.5 * (
        ((current - y) / scale).square().mean()
        + ((previous - z) / scale).square().mean()
    )
    expected = 0.25 * absolute + 0.75 * (
        ((current - previous - y + z) / ds).square().mean()
    )
    torch.testing.assert_close(
        pose_objective(current, previous, y, z, scale, ds, "motion"), expected
    )
    torch.testing.assert_close(
        pose_objective(current, previous, y, z, scale, ds, "absolute"), absolute
    )
    torch.testing.assert_close(
        pose_objective(previous, current, z, y, scale, ds, "motion"), expected
    )


def metadata_fixture():
    n = 90
    metadata = dict(
        room=np.repeat(["E01", "E03", "E01", "E02"], n),
        person=np.repeat(["S01", "S21", "S04", "S11"], n),
        record=np.repeat(
            ["E01/S01/A01", "E03/S21/A01", "E01/S04/A01", "E02/S11/A01"], n
        ),
        end=np.tile(np.arange(n) * 8 + 39, 4),
    )
    return metadata, np.arange(4 * n) < 2 * n


def test_schedule_is_paired_same_record_causal_source_train_only_and_reproducible(
    monkeypatch,
):
    metadata, train = metadata_fixture()
    monkeypatch.setitem(probe.CONFIG, "steps", 8)
    first, second = probe.schedule(metadata, train), probe.schedule(metadata, train)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (8, 32, 2)
    assert train[first].all()
    np.testing.assert_array_equal(
        metadata["record"][first[:, :, 0]], metadata["record"][first[:, :, 1]]
    )
    np.testing.assert_array_equal(
        metadata["end"][first[:, :, 0]] - metadata["end"][first[:, :, 1]],
        np.full((8, 32), 8),
    )
    for batch in first:
        assert len(set(metadata["room"][batch[:, 0]])) == 1
        assert len(set(batch[:, 0])) == 32


def test_train_statistics_never_read_validation_or_held_values():
    metadata, train = metadata_fixture()
    rng = np.random.default_rng(11)
    x, y = rng.normal(size=(len(train), 95, 48)), rng.normal(size=(len(train), 17, 2))

    class Tracked:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, ids):
            assert train[ids].all()
            return self.value[ids]

    before = probe.training_statistics(Tracked(x), y, metadata, train)
    x[~train], y[~train] = np.nan, np.nan
    after = probe.training_statistics(Tracked(x), y, metadata, train)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    wrong = train.copy()
    wrong[-1] = True
    with pytest.raises(ValueError, match="source TRAIN"):
        probe.training_statistics(Tracked(x), y, metadata, wrong)
    wrong = train.copy()
    wrong[-100] = True
    with pytest.raises(ValueError, match="source TRAIN"):
        probe.training_statistics(Tracked(x), y, metadata, wrong)


def test_readout_can_learn_synthetic_local_token_signal_with_finite_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    x = torch.randn(48, 95, 48) * 0.1
    # Signal sits in a subset of patches; numerical fixture, not radio data.
    signal = torch.linspace(-1, 1, 48)
    x[:, :8, 0] += signal[:, None] * 4
    y = torch.zeros(48, 17, 2)
    y[:, 0, 0] = signal
    network = model()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    first = (network(x) - y).square().mean().item()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = (network(x) - y).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
    assert (network(x) - y).square().mean().item() < first * 0.35


def test_invalid_head_schema_and_existing_output_refuse(tmp_path):
    with pytest.raises(ValueError):
        SpatialPose(torch.zeros(17, 2), torch.zeros(17, 2))
    with pytest.raises(ValueError):
        model()(torch.zeros(3, 1, 48))
    with pytest.raises(FileExistsError):
        probe.run(tmp_path / "absent", tmp_path / "absent", tmp_path, "cpu")
