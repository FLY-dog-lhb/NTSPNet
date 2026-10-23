from types import SimpleNamespace

import pytest
import torch

from losses import HuberTrajLoss
from model import PNet


def test_endpoint_auxiliary_selects_endpoint_mode_independently():
    model = SimpleNamespace(reg_loss_traj=HuberTrajLoss())
    target = torch.tensor([[[[0.0, 0.0], [10.0, 0.0]]]])
    target_mask = torch.ones(1, 1, 2, dtype=torch.bool)
    predict_mask = torch.ones(1, 1, dtype=torch.bool)
    proposal = torch.tensor([
        0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 9.0, 0.0,
    ]).reshape(1, 1, 2, 2, 2).requires_grad_()
    output = torch.tensor([
        0.0, 0.0, 10.0, 0.0,
        100.0, 0.0, 8.0, 0.0,
    ]).reshape(1, 1, 2, 2, 2).requires_grad_()
    zero_loss = (proposal.sum() + output.sum()) * 0.0

    loss = PNet._compute_endpoint_auxiliary(
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
    assert proposal.grad[0, 0, 1, -1].abs().sum().item() > 0.0
    assert output.grad[0, 0, 1, -1].abs().sum().item() > 0.0


def test_endpoint_auxiliary_returns_graph_zero_without_valid_endpoint():
    model = SimpleNamespace(reg_loss_traj=HuberTrajLoss())
    proposal = torch.randn(2, 1, 3, 2, 2, requires_grad=True)
    output = torch.randn(2, 1, 3, 2, 2, requires_grad=True)
    zero_loss = (proposal.sum() + output.sum()) * 0.0
    loss = PNet._compute_endpoint_auxiliary(
        model,
        traj_propose=proposal,
        traj_output=output,
        target_traj=torch.zeros(2, 1, 2, 2),
        target_mask=torch.zeros(2, 1, 2, dtype=torch.bool),
        predict_mask=torch.ones(2, 1, dtype=torch.bool),
        agent_batch=torch.tensor([0, 1]),
        zero_loss=zero_loss,
    )
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(proposal.grad, torch.zeros_like(proposal))
    assert torch.equal(output.grad, torch.zeros_like(output))
