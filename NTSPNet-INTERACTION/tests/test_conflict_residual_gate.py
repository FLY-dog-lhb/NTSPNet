import pytest
import torch

from modules.backbone import Backbone


def make_backbone(gate_scale=0.0):
    return Backbone(
        hidden_dim=8,
        num_historical_steps=3,
        num_future_steps=4,
        duration=1,
        a2a_radius=50.0,
        l2a_radius=50.0,
        num_attn_layers=1,
        num_modes=2,
        num_heads=2,
        dropout=0.0,
        proposal_conflict_residual_gate_scale=gate_scale,
    )


def conflict_nodes():
    # Flattened [H=3,K=2,N=3] indices at the final historical step.
    return torch.tensor([[12, 15], [13, 16]])


def test_zero_scale_returns_original_conflict_output():
    backbone = make_backbone(0.0)
    conflict_input = torch.randn(18, backbone.hidden_dim)
    conflict_output = torch.randn_like(conflict_input)

    output = backbone._gate_proposal_conflict_residual(
        conflict_input,
        conflict_output,
        conflict_nodes(),
        num_agents=3,
    )

    assert output is conflict_output


def test_zero_initialized_gate_is_strict_identity_with_gradient():
    backbone = make_backbone(0.1)
    conflict_input = torch.randn(18, backbone.hidden_dim)
    conflict_output = conflict_input + torch.randn_like(conflict_input)

    output = backbone._gate_proposal_conflict_residual(
        conflict_input,
        conflict_output,
        conflict_nodes(),
        num_agents=3,
    )
    output.sum().backward()

    gate = backbone.proposal_conflict_residual_gate_tokens.weight
    assert torch.equal(output, conflict_output)
    assert torch.count_nonzero(gate).item() == 0
    assert gate.grad is not None
    assert torch.any(gate.grad != 0)


def test_gate_uses_mode_order_and_changes_participants_only():
    backbone = make_backbone(0.1)
    with torch.no_grad():
        backbone.proposal_conflict_residual_gate_tokens.weight[0].zero_()
        backbone.proposal_conflict_residual_gate_tokens.weight[1].fill_(2.0)
    conflict_input = torch.zeros(18, backbone.hidden_dim)
    conflict_output = torch.ones_like(conflict_input)

    output = backbone._gate_proposal_conflict_residual(
        conflict_input,
        conflict_output,
        conflict_nodes(),
        num_agents=3,
    )

    assert torch.equal(output[12:15], conflict_output[12:15])
    assert torch.all(output[[15, 16]] > conflict_output[[15, 16]])
    assert torch.all(output[[15, 16]] <= 1.1)
    assert torch.equal(output[17], conflict_output[17])
    assert torch.equal(output[:12], conflict_output[:12])


def test_gate_rejects_non_hkn_layout():
    backbone = make_backbone(0.1)

    with pytest.raises(ValueError, match=r'\[H,K,N,D\]'):
        backbone._gate_proposal_conflict_residual(
            torch.zeros(17, 8),
            torch.zeros(17, 8),
            conflict_nodes(),
            num_agents=3,
        )


def test_pre_gate_state_dict_strictly_loads_with_zero_migration():
    source = make_backbone()
    pre_gate_state = source.state_dict()
    pre_gate_state.pop('proposal_conflict_residual_gate_tokens.weight')
    target = make_backbone(0.1)
    with torch.no_grad():
        target.proposal_conflict_residual_gate_tokens.weight.fill_(3.0)

    target.load_state_dict(pre_gate_state, strict=True)

    assert torch.count_nonzero(
        target.proposal_conflict_residual_gate_tokens.weight
    ).item() == 0


@pytest.mark.parametrize('scale', [-0.1, 1.1])
def test_residual_gate_scale_is_bounded(scale):
    with pytest.raises(ValueError, match='between 0 and 1'):
        make_backbone(scale)
