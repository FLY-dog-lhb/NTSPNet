from types import SimpleNamespace

import pytest
import torch

from losses import HuberTrajLoss
from model import PNet


def test_late_horizon_auxiliary_selects_whole_segment_mode():
    model = SimpleNamespace(
        reg_loss_traj=HuberTrajLoss(),
        late_horizon_steps=3,
        late_horizon_reduction='agent_mean',
    )
    target = torch.tensor([[[
        [0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [10.0, 0.0],
    ]]])
    target_mask = torch.ones(1, 1, 4, dtype=torch.bool)
    predict_mask = torch.ones(1, 1, dtype=torch.bool)
    proposal = torch.tensor([
        0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 10.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 6.0, 0.0, 11.0, 0.0,
    ]).reshape(1, 1, 2, 4, 2).requires_grad_()
    output = torch.tensor([
        0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 10.0, 0.0,
        0.0, 0.0, 2.0, 0.0, 7.0, 0.0, 12.0, 0.0,
    ]).reshape(1, 1, 2, 4, 2).requires_grad_()
    zero_loss = (proposal.sum() + output.sum()) * 0.0

    loss = PNet._compute_late_horizon_auxiliary(
        model,
        traj_propose=proposal,
        traj_output=output,
        target_traj=target,
        target_mask=target_mask,
        predict_mask=predict_mask,
        agent_batch=torch.zeros(1, dtype=torch.long),
        zero_loss=zero_loss,
    )

    assert loss.item() == pytest.approx(1.75)
    loss.backward()
    assert proposal.grad[0, 0, 0].abs().sum().item() == 0.0
    assert output.grad[0, 0, 0].abs().sum().item() == 0.0
    assert proposal.grad[0, 0, 1, 0].abs().sum().item() == 0.0
    assert output.grad[0, 0, 1, 0].abs().sum().item() == 0.0
    assert proposal.grad[0, 0, 1, 1:].abs().sum().item() > 0.0
    assert output.grad[0, 0, 1, 1:].abs().sum().item() > 0.0


def test_late_horizon_auxiliary_requires_complete_segment():
    model = SimpleNamespace(
        reg_loss_traj=HuberTrajLoss(),
        late_horizon_steps=3,
        late_horizon_reduction='agent_mean',
    )
    proposal = torch.randn(2, 1, 3, 4, 2, requires_grad=True)
    output = torch.randn(2, 1, 3, 4, 2, requires_grad=True)
    zero_loss = (proposal.sum() + output.sum()) * 0.0
    target_mask = torch.ones(2, 1, 4, dtype=torch.bool)
    target_mask[:, :, -2] = False

    loss = PNet._compute_late_horizon_auxiliary(
        model,
        traj_propose=proposal,
        traj_output=output,
        target_traj=torch.zeros(2, 1, 4, 2),
        target_mask=target_mask,
        predict_mask=torch.ones(2, 1, dtype=torch.bool),
        agent_batch=torch.tensor([0, 1]),
        zero_loss=zero_loss,
    )

    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(proposal.grad, torch.zeros_like(proposal))
    assert torch.equal(output.grad, torch.zeros_like(output))


def test_scene_mean_gives_each_scene_equal_weight():
    target = torch.zeros(4, 1, 2, 2)
    target_mask = torch.ones(4, 1, 2, dtype=torch.bool)
    predict_mask = torch.ones(4, 1, dtype=torch.bool)
    agent_batch = torch.tensor([0, 1, 1, 1])
    values = torch.tensor([1.0, 3.0, 3.0, 3.0])
    prediction = torch.zeros(4, 1, 1, 2, 2)
    prediction[:, :, :, :, 0] = values[:, None, None, None]

    losses = {}
    gradients = {}
    for reduction in ('agent_mean', 'scene_mean'):
        proposal = prediction.clone().requires_grad_()
        output = prediction.clone().requires_grad_()
        model = SimpleNamespace(
            reg_loss_traj=HuberTrajLoss(),
            late_horizon_steps=2,
            late_horizon_reduction=reduction,
        )
        loss = PNet._compute_late_horizon_auxiliary(
            model,
            traj_propose=proposal,
            traj_output=output,
            target_traj=target,
            target_mask=target_mask,
            predict_mask=predict_mask,
            agent_batch=agent_batch,
            zero_loss=(proposal.sum() + output.sum()) * 0.0,
        )
        loss.backward()
        losses[reduction] = loss.item()
        gradients[reduction] = output.grad.detach().clone()

    assert losses['agent_mean'] == pytest.approx(3.0)
    assert losses['scene_mean'] == pytest.approx(2.25)
    scene_mean_gradient = gradients['scene_mean'].abs()
    assert scene_mean_gradient[0].sum().item() == pytest.approx(
        scene_mean_gradient[1:].sum().item()
    )
