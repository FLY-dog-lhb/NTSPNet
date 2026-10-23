import pytest
import torch

from modules.backbone import Backbone


def make_backbone(scale=0.0, adapter_scale=0.0):
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
        proposal_conflict_mode_scale=scale,
        proposal_conflict_mode_adapter_scale=adapter_scale,
    )


def test_zero_scale_is_strict_identity():
    backbone = make_backbone(0.0)
    edge_embedding = torch.randn(6, backbone.hidden_dim, requires_grad=True)

    output = backbone._condition_proposal_conflict_edges(edge_embedding)

    assert output is edge_embedding


def test_mode_conditioning_is_bounded_and_uses_edge_major_mode_order():
    backbone = make_backbone(0.1)
    with torch.no_grad():
        backbone.mode_tokens.weight[0].zero_()
        backbone.mode_tokens.weight[1].fill_(2.0)
    edge_embedding = torch.ones(4, backbone.hidden_dim)

    output = backbone._condition_proposal_conflict_edges(edge_embedding)

    assert torch.equal(output[[0, 2]], edge_embedding[[0, 2]])
    assert torch.all(output[[1, 3]] > 1.0)
    assert torch.all(output[[1, 3]] <= 1.1)
    assert torch.isfinite(output).all()


def test_mode_conditioning_rejects_unexpanded_edge_count():
    backbone = make_backbone(0.1)

    with pytest.raises(ValueError, match='divisible by num_modes'):
        backbone._condition_proposal_conflict_edges(torch.ones(3, 8))


def test_mode_conditioning_adds_no_checkpoint_state():
    legacy = make_backbone(0.0)
    candidate = make_backbone(0.1)

    assert legacy.state_dict().keys() == candidate.state_dict().keys()
    candidate.load_state_dict(legacy.state_dict(), strict=True)


def test_adapter_is_zero_initialized_and_starts_as_strict_identity():
    backbone = make_backbone(adapter_scale=0.1)
    edge_embedding = torch.randn(4, backbone.hidden_dim)

    output = backbone._condition_proposal_conflict_edges(edge_embedding)

    assert torch.count_nonzero(
        backbone.proposal_conflict_mode_tokens.weight
    ).item() == 0
    assert torch.equal(output, edge_embedding)


def test_adapter_uses_dedicated_tokens_without_global_token_gradients():
    backbone = make_backbone(adapter_scale=0.1)
    with torch.no_grad():
        backbone.mode_tokens.weight.fill_(5.0)
        backbone.proposal_conflict_mode_tokens.weight[0].zero_()
        backbone.proposal_conflict_mode_tokens.weight[1].fill_(2.0)
    edge_embedding = torch.ones(4, backbone.hidden_dim)

    output = backbone._condition_proposal_conflict_edges(edge_embedding)
    output.sum().backward()

    assert torch.equal(output[[0, 2]], edge_embedding[[0, 2]])
    assert torch.all(output[[1, 3]] > 1.0)
    assert backbone.mode_tokens.weight.grad is None
    assert torch.any(
        backbone.proposal_conflict_mode_tokens.weight.grad != 0
    )


def test_pre_adapter_state_dict_strictly_loads_with_zero_migration():
    source = make_backbone()
    pre_adapter_state = source.state_dict()
    pre_adapter_state.pop('proposal_conflict_mode_tokens.weight')
    target = make_backbone(adapter_scale=0.1)
    with torch.no_grad():
        target.proposal_conflict_mode_tokens.weight.fill_(3.0)

    target.load_state_dict(pre_adapter_state, strict=True)

    assert torch.count_nonzero(
        target.proposal_conflict_mode_tokens.weight
    ).item() == 0


@pytest.mark.parametrize('scale', [-0.1, 1.1])
@pytest.mark.parametrize('scale_name', ['legacy', 'adapter'])
def test_mode_conditioning_scale_is_bounded(scale, scale_name):
    kwargs = (
        {'scale': scale}
        if scale_name == 'legacy'
        else {'adapter_scale': scale}
    )
    with pytest.raises(ValueError, match='between 0 and 1'):
        make_backbone(**kwargs)
