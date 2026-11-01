import pytest
import torch

from ghostmesh.cross_room_pose import cross_room_alignment


def test_cross_room_teacher_uses_only_other_source_room_and_detaches_poses():
    torch.manual_seed(7)
    code = torch.randn(8, 128, requires_grad=True)
    pose = torch.randn(8, 17, 2, requires_grad=True)
    room = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss = cross_room_alignment(code, pose, room, torch.ones(17, 2))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(code.grad).all()
    assert code.grad.abs().sum() > 0 and pose.grad is None
    permuted = cross_room_alignment(
        code,
        pose,
        room,
        torch.ones(17, 2),
        permutation=torch.tensor([2, 3, 0, 1, 6, 7, 4, 5]),
    )
    assert abs(float(loss.detach() - permuted.detach())) > 1e-5
    with pytest.raises(ValueError, match="another SOURCE room"):
        cross_room_alignment(code, pose, torch.zeros(8), torch.ones(17, 2))
    with pytest.raises(ValueError, match="Within-SOURCE-room"):
        cross_room_alignment(
            code, pose, room, torch.ones(17, 2), permutation=torch.arange(7, -1, -1)
        )
