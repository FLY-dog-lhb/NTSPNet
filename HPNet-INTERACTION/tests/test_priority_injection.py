import torch

from modules.backbone import Backbone


def make_backbone(priority_injection_mode):
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
        priority_injection_mode=priority_injection_mode,
    )


def test_auxiliary_only_replaces_priority_trajectory_inputs_with_constants():
    backbone = make_backbone('auxiliary_only')
    probability = torch.randn((4, 3), requires_grad=True)
    head_bias = torch.randn((4, 2), requires_grad=True)

    trajectory_probability, trajectory_head_bias = (
        backbone._priority_trajectory_inputs(probability, head_bias)
    )

    assert torch.equal(trajectory_probability, torch.zeros_like(probability))
    assert torch.equal(trajectory_head_bias, torch.zeros_like(head_bias))
    assert not trajectory_probability.requires_grad
    assert not trajectory_head_bias.requires_grad

    auxiliary_loss = probability.square().mean() + head_bias.square().mean()
    auxiliary_loss.backward()
    assert torch.any(probability.grad != 0)
    assert torch.any(head_bias.grad != 0)


def test_legacy_priority_injection_preserves_outputs_and_gradients():
    backbone = make_backbone('legacy')
    probability = torch.randn((4, 3), requires_grad=True)
    head_bias = torch.randn((4, 2), requires_grad=True)

    trajectory_probability, trajectory_head_bias = (
        backbone._priority_trajectory_inputs(probability, head_bias)
    )
    trajectory_loss = (
        trajectory_probability.square().mean()
        + trajectory_head_bias.square().mean()
    )
    trajectory_loss.backward()

    assert trajectory_probability is probability
    assert trajectory_head_bias is head_bias
    assert torch.any(probability.grad != 0)
    assert torch.any(head_bias.grad != 0)
