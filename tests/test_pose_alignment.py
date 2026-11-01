import pytest
import torch

from ghostmesh.pose_alignment import PoseAlignedStudent, cross_person_alignment


def test_inference_needs_only_rf_and_preserves_the_source_pose_prior():
    torch.manual_seed(1)
    prototypes = torch.randn(128, 17, 2) * 0.1
    prototypes -= prototypes[:, [11, 12]].mean(1, keepdim=True)
    prior = torch.arange(1, 129, dtype=torch.float32)
    prior /= prior.sum()
    model = PoseAlignedStudent(
        torch.zeros(12, 114), torch.ones(12, 114), prototypes, prior
    )
    x = torch.randn(2, 32, 12, 114)
    out = model(x)
    expected = torch.einsum("k,kjd->jd", prior, prototypes)
    torch.testing.assert_close(
        out["pose"], expected.expand(2, -1, -1), atol=2e-7, rtol=2e-6
    )
    torch.testing.assert_close(out["prototype_weights"].sum(1), torch.ones(2))
    assert out["code"].shape == (2, 128)
    with pytest.raises(ValueError):
        model(x[:, :8])


def test_continuous_cross_person_loss_is_finite_and_updates_radio_codes_only():
    torch.manual_seed(3)
    code = torch.randn(6, 128, requires_grad=True)
    pose = torch.randn(6, 17, 2, requires_grad=True)
    people = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = cross_person_alignment(code, pose, people, torch.ones(17, 2))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(code.grad).all()
    assert code.grad.abs().sum() > 0 and pose.grad is None
    shuffled = cross_person_alignment(
        code,
        pose,
        people,
        torch.ones(17, 2),
        permutation=torch.tensor([1, 0, 3, 2, 5, 4]),
    )
    assert abs(float(shuffled.detach() - loss.detach())) > 1e-5
    with pytest.raises(ValueError, match="different"):
        cross_person_alignment(code, pose, torch.zeros(6), torch.ones(17, 2))
    with pytest.raises(ValueError, match="Within-person"):
        cross_person_alignment(
            code, pose, people, torch.ones(17, 2), permutation=torch.arange(5, -1, -1)
        )
