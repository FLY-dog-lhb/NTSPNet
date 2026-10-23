import pytest

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
        motion_source='csv',
        trunk_variant='original_residual',
        baseline_residual_training_scope='residual_branch',
        baseline_residual_provider_lr_scale=0.1,
    )
    configuration.update(overrides)
    return PNet(**configuration)


def test_residual_branch_optimizer_uses_smaller_provider_learning_rate():
    model = make_model()
    optimizer = model.configure_optimizers()[0][0]
    parameter_names = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    learning_rates_by_name = {
        parameter_names[id(parameter)]: group['lr']
        for group in optimizer.param_groups
        for parameter in group['params']
    }

    assert learning_rates_by_name
    assert all(
        learning_rate == pytest.approx(1e-4)
        for name, learning_rate in learning_rates_by_name.items()
        if name.startswith('Backbone.residual.')
    )
    assert all(
        learning_rate == pytest.approx(1e-3)
        for name, learning_rate in learning_rates_by_name.items()
        if not name.startswith('Backbone.residual.')
    )
    assert not any(
        name.startswith('Backbone.base.') or name.startswith('MapEncoder.')
        for name in learning_rates_by_name
    )


@pytest.mark.parametrize('scale', [0.0, -0.1, 1.1])
def test_provider_learning_rate_scale_must_be_bounded(scale):
    with pytest.raises(ValueError, match='provider_lr_scale'):
        make_model(baseline_residual_provider_lr_scale=scale)
