import pytest
import torch

from model import PNet


def make_model(**overrides):
    configuration = dict(
        hidden_dim=8,
        num_historical_steps=3,
        num_future_steps=4,
        duration=1,
        a2a_radius=50.0,
        l2a_radius=50.0,
        num_visible_steps=1,
        num_modes=2,
        num_attn_layers=1,
        num_hops=1,
        num_heads=2,
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-4,
        warmup_epochs=1,
        T_max=2,
        proposal_conflict_residual_gate_scale=0.1,
        proposal_conflict_training_scope='residual_gate_only',
    )
    configuration.update(overrides)
    return PNet(**configuration)


def test_residual_gate_only_scope_has_exactly_gate_parameters():
    model = make_model()
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert set(trainable) == {
        'Backbone.proposal_conflict_residual_gate_tokens.weight'
    }
    assert sum(parameter.numel() for parameter in trainable.values()) == 16


def test_residual_gate_only_optimizer_contains_no_frozen_parameters():
    model = make_model()

    optimizers, _ = model.configure_optimizers()
    optimized = {
        id(parameter)
        for group in optimizers[0].param_groups
        for parameter in group['params']
    }

    assert optimized == {
        id(model.Backbone.proposal_conflict_residual_gate_tokens.weight)
    }
    assert optimizers[0].param_groups[0]['weight_decay'] == 0.0


def test_zero_conflict_anchor_targets_residual_gate():
    model = make_model()

    anchor = model._proposal_conflict_adapter_anchor()
    anchor.backward()

    gradient = (
        model.Backbone.proposal_conflict_residual_gate_tokens.weight.grad
    )
    assert anchor.requires_grad
    assert torch.equal(gradient, torch.zeros_like(gradient))
    assert model.Backbone.proposal_conflict_mode_tokens.weight.grad is None


@pytest.mark.parametrize(
    'overrides, message',
    [
        (
            {'proposal_conflict_residual_gate_scale': 0.0},
            'requires a non-zero',
        ),
        (
            {'proposal_conflict_mode_scale': 0.1},
            'legacy proposal_conflict_mode_scale',
        ),
        (
            {'proposal_conflict_mode_adapter_scale': 0.1},
            'mode adapter to remain disabled',
        ),
        (
            {'trunk_variant': 'original'},
            'requires the extended trunk',
        ),
    ],
)
def test_residual_gate_only_rejects_confounded_configurations(
        overrides, message):
    with pytest.raises(ValueError, match=message):
        make_model(**overrides)
