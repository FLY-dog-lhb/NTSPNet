from itertools import permutations
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import HeteroData

from model import PNet
from transforms import HorizontalFlip
from transforms import TeacherTrajectoryTransform


def make_sample():
    data = HeteroData()
    data['scenario_name'] = 'scene'
    data['case_id'] = 7
    data['agent']['id'] = torch.tensor([2, 5, 8], dtype=torch.uint8)
    data['agent']['category'] = torch.tensor([1, 0, 1], dtype=torch.uint8)
    data['agent']['position'] = torch.zeros(3, 4, 2)
    data['agent']['heading'] = torch.zeros(3, 4)
    return data


def write_sidecar(cache_dir, *, agent_id=None):
    split_dir = cache_dir / 'train'
    split_dir.mkdir(parents=True)
    trajectory = torch.arange(2 * 3 * 4 * 2, dtype=torch.float).reshape(
        2, 3, 4, 2,
    )
    torch.save({
        'schema_version': 1,
        'sample_id': 'scene_7.pt',
        'teacher_checkpoint': '/teacher.ckpt',
        'agent_id': (
            torch.tensor([2, 5, 8], dtype=torch.uint8)
            if agent_id is None else agent_id
        ),
        'predicted_agent_mask': torch.tensor([True, False, True]),
        'trajectory': trajectory,
    }, split_dir / 'scene_7.pt')
    return trajectory


def test_teacher_transform_attaches_trajectories_in_agent_order(tmp_path):
    expected = write_sidecar(tmp_path)
    data = TeacherTrajectoryTransform(tmp_path)(make_sample())

    assert data['agent']['teacher_trajectory'].shape == (3, 3, 4, 2)
    assert torch.equal(data['agent']['teacher_trajectory'][0], expected[0])
    assert torch.equal(
        data['agent']['teacher_trajectory'][1],
        torch.zeros_like(expected[0]),
    )
    assert torch.equal(data['agent']['teacher_trajectory'][2], expected[1])
    assert torch.equal(
        data['agent']['teacher_trajectory_valid_mask'],
        torch.tensor([True, False, True]),
    )


def test_teacher_transform_rejects_agent_order_mismatch(tmp_path):
    write_sidecar(
        tmp_path,
        agent_id=torch.tensor([5, 2, 8], dtype=torch.uint8),
    )
    with pytest.raises(ValueError, match='agent order'):
        TeacherTrajectoryTransform(tmp_path)(make_sample())


def test_horizontal_flip_keeps_cached_teacher_in_same_coordinate_frame():
    data = make_sample()
    teacher = torch.tensor([[[[1.0, 2.0]]]]).expand(3, 1, 1, 2).clone()
    data['agent']['teacher_trajectory'] = teacher
    data['lane']['position'] = torch.zeros(1, 2)
    data['lane']['heading'] = torch.zeros(1)
    data['polyline']['position'] = torch.zeros(1, 2)
    data['polyline']['heading'] = torch.zeros(1)
    data['polyline']['side'] = torch.ones(1, dtype=torch.long)
    data['lane', 'lane']['left_neighbor_edge_index'] = torch.empty(
        2, 0, dtype=torch.long,
    )
    data['lane', 'lane']['right_neighbor_edge_index'] = torch.empty(
        2, 0, dtype=torch.long,
    )

    flipped = HorizontalFlip(flip_p=1.0)(data)

    assert torch.equal(
        flipped['agent']['teacher_trajectory'][..., 0],
        torch.full((3, 1, 1), -1.0),
    )
    assert torch.equal(
        flipped['agent']['teacher_trajectory'][..., 1],
        torch.full((3, 1, 1), 2.0),
    )


def distillation_model(num_modes):
    return SimpleNamespace(
        num_modes=num_modes,
        teacher_advantage_margin=0.05,
        _teacher_mode_permutations=torch.tensor(
            list(permutations(range(num_modes))), dtype=torch.long,
        ),
    )


def distillation_data(teacher):
    data = HeteroData()
    data['agent']['teacher_trajectory'] = teacher
    data['agent']['teacher_trajectory_valid_mask'] = torch.ones(
        teacher.size(0), dtype=torch.bool,
    )
    return data


def test_distillation_is_zero_for_a_permutation_of_teacher_modes():
    teacher = torch.arange(3 * 3 * 2 * 2, dtype=torch.float).reshape(
        3, 3, 2, 2,
    )
    student = teacher[:, [2, 0, 1]].unsqueeze(1).clone().requires_grad_()
    loss = PNet._compute_teacher_trajectory_distillation(
        distillation_model(3),
        data=distillation_data(teacher),
        traj_output=student,
        agent_mask=torch.ones(3, dtype=torch.bool),
        agent_batch=torch.tensor([0, 1, 1]),
    )

    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_distillation_uses_equal_scene_weighting():
    teacher = torch.zeros(4, 2, 1, 2)
    student = torch.zeros(4, 1, 2, 1, 2)
    student[0, ..., 0] = 1.0
    student[1:, ..., 0] = 2.0
    student.requires_grad_()
    loss = PNet._compute_teacher_trajectory_distillation(
        distillation_model(2),
        data=distillation_data(teacher),
        traj_output=student,
        agent_mask=torch.ones(4, dtype=torch.bool),
        agent_batch=torch.tensor([0, 1, 1, 1]),
    )

    # SmoothL1(1)=0.5 and SmoothL1(2)=1.5; scenes contribute equally.
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum().item() > 0


def test_advantage_distillation_updates_only_one_mode_of_eligible_agent():
    teacher = torch.zeros(2, 2, 2, 2)
    teacher[1, :, :, 0] = 0.2
    student = torch.zeros(2, 1, 2, 2, 2)
    student[0, 0, 0, :, 0] = 0.2
    student[0, 0, 1, :, 0] = 5.0
    student[1, 0, 0, :, 0] = 0.0
    student[1, 0, 1, :, 0] = 5.0
    student.requires_grad_()
    target = torch.zeros(2, 1, 2, 2)
    target_mask = torch.ones(2, 1, 2, dtype=torch.bool)
    predict_mask = torch.ones(2, 1, dtype=torch.bool)

    loss, eligible_rate = PNet._compute_teacher_advantage_distillation(
        distillation_model(2),
        data=distillation_data(teacher),
        traj_output=student,
        agent_mask=torch.ones(2, dtype=torch.bool),
        agent_batch=torch.tensor([0, 1]),
        target_traj=target,
        target_mask=target_mask,
        predict_mask=predict_mask,
    )

    assert loss.item() == pytest.approx(0.04)
    assert eligible_rate.item() == pytest.approx(0.5)
    loss.backward()
    assert student.grad[0, 0, 0].abs().sum().item() > 0
    assert student.grad[0, 0, 1].abs().sum().item() == 0
    assert student.grad[1].abs().sum().item() == 0


def test_advantage_distillation_returns_graph_zero_without_margin():
    teacher = torch.zeros(1, 2, 2, 2)
    student = torch.zeros(1, 1, 2, 2, 2)
    student[..., 0] = 0.04
    student.requires_grad_()

    loss, eligible_rate = PNet._compute_teacher_advantage_distillation(
        distillation_model(2),
        data=distillation_data(teacher),
        traj_output=student,
        agent_mask=torch.ones(1, dtype=torch.bool),
        agent_batch=torch.zeros(1, dtype=torch.long),
        target_traj=torch.zeros(1, 1, 2, 2),
        target_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        predict_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    assert loss.item() == 0.0
    assert eligible_rate.item() == 0.0
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_scene_mode_distillation_uses_one_shared_mode_for_all_agents():
    teacher = torch.zeros(2, 2, 2, 2)
    teacher[:, 1, :, 0] = 4.0
    student = torch.zeros(2, 1, 2, 2, 2)
    student[0, 0, 0, :, 0] = 0.2
    student[0, 0, 1, :, 0] = 3.0
    student[1, 0, 0, :, 0] = 3.0
    student[1, 0, 1, :, 0] = 0.4
    student.requires_grad_()

    loss, eligible_rate = PNet._compute_teacher_scene_mode_distillation(
        distillation_model(2),
        data=distillation_data(teacher),
        traj_output=student,
        agent_mask=torch.ones(2, dtype=torch.bool),
        agent_batch=torch.zeros(2, dtype=torch.long),
        target_traj=torch.zeros(2, 1, 2, 2),
        target_mask=torch.ones(2, 1, 2, dtype=torch.bool),
        predict_mask=torch.ones(2, 1, dtype=torch.bool),
    )

    assert loss.item() == pytest.approx(2.52)
    assert eligible_rate.item() == 1.0
    loss.backward()
    assert student.grad[:, 0, 0].abs().sum(dim=(1, 2)).gt(0).all()
    assert student.grad[:, 0, 1].abs().sum().item() == 0


def test_scene_mode_distillation_returns_graph_zero_without_scene_margin():
    teacher = torch.zeros(2, 2, 2, 2)
    student = torch.zeros(2, 1, 2, 2, 2)
    student[..., 0] = 0.04
    student.requires_grad_()

    loss, eligible_rate = PNet._compute_teacher_scene_mode_distillation(
        distillation_model(2),
        data=distillation_data(teacher),
        traj_output=student,
        agent_mask=torch.ones(2, dtype=torch.bool),
        agent_batch=torch.zeros(2, dtype=torch.long),
        target_traj=torch.zeros(2, 1, 2, 2),
        target_mask=torch.ones(2, 1, 2, dtype=torch.bool),
        predict_mask=torch.ones(2, 1, dtype=torch.bool),
    )

    assert loss.item() == 0.0
    assert eligible_rate.item() == 0.0
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))
