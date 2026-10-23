import pytest
import torch

from model import PNet
from modules.backbone import Backbone


def make_backbone(scale=0.1):
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
        scene_mode_coordination_scale=scale,
    )


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
        weight_decay=0.0,
        warmup_epochs=1,
        T_max=2,
        scene_mode_coordination_scale=0.1,
        proposal_conflict_training_scope='scene_coordination_only',
    )
    configuration.update(overrides)
    return PNet(**configuration)


def scene_inputs(backbone):
    num_agents = 3
    node_embedding = torch.randn(
        num_agents
        * backbone.num_historical_steps
        * backbone.num_modes,
        backbone.hidden_dim,
    )
    agent_batch = torch.tensor([0, 0, 1], dtype=torch.long)
    valid_mask = torch.ones(
        num_agents,
        backbone.num_historical_steps,
        backbone.num_modes,
        dtype=torch.bool,
    )
    return node_embedding, agent_batch, valid_mask


def test_zero_initialized_coordination_is_strict_identity_with_gradient():
    backbone = make_backbone()
    node_embedding, agent_batch, valid_mask = scene_inputs(backbone)

    output = backbone._coordinate_scene_modes(
        node_embedding,
        agent_batch,
        valid_mask,
        num_scenes=2,
    )
    output.sum().backward()

    assert torch.equal(output, node_embedding)
    assert torch.count_nonzero(backbone.scene_mode_coordination.weight) == 0
    assert torch.count_nonzero(backbone.scene_mode_coordination.bias) == 0
    assert backbone.scene_mode_coordination.weight.grad is not None
    assert torch.any(backbone.scene_mode_coordination.weight.grad != 0)


def test_coordination_is_bounded_and_keeps_invisible_nodes_unchanged():
    backbone = make_backbone(scale=0.1)
    with torch.no_grad():
        backbone.scene_mode_coordination.weight.fill_(2.0)
        backbone.scene_mode_coordination.bias.fill_(2.0)
    node_embedding, agent_batch, valid_mask = scene_inputs(backbone)
    valid_mask[0, 0, 0] = False

    output = backbone._coordinate_scene_modes(
        node_embedding,
        agent_batch,
        valid_mask,
        num_scenes=2,
    )
    delta = (output - node_embedding).reshape(
        3, backbone.num_historical_steps, backbone.num_modes, -1,
    )

    assert torch.equal(delta[0, 0, 0], torch.zeros(backbone.hidden_dim))
    assert delta.abs().max() <= 0.100001
    assert torch.any(delta[valid_mask] != 0)


def test_pre_coordination_state_dict_strictly_loads_as_zero():
    source = make_backbone(scale=0.0)
    old_state = source.state_dict()
    old_state.pop('scene_mode_coordination.weight')
    old_state.pop('scene_mode_coordination.bias')
    target = make_backbone(scale=0.1)
    with torch.no_grad():
        target.scene_mode_coordination.weight.fill_(3.0)
        target.scene_mode_coordination.bias.fill_(3.0)

    target.load_state_dict(old_state, strict=True)

    assert torch.count_nonzero(target.scene_mode_coordination.weight) == 0
    assert torch.count_nonzero(target.scene_mode_coordination.bias) == 0


def test_scene_coordination_only_scope_trains_only_new_projection():
    model = make_model()
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert set(trainable) == {
        'Backbone.scene_mode_coordination.weight',
        'Backbone.scene_mode_coordination.bias',
    }
    assert sum(parameter.numel() for parameter in trainable.values()) == 72


@pytest.mark.parametrize(
    'overrides, message',
    [
        (
            {'scene_mode_coordination_scale': 0.0},
            'requires a non-zero',
        ),
        (
            {'proposal_conflict_mode_adapter_scale': 0.1},
            'experimental branches to remain disabled',
        ),
        (
            {'trunk_variant': 'original'},
            'requires the extended trunk',
        ),
    ],
)
def test_scene_coordination_only_rejects_confounded_configuration(
        overrides, message):
    with pytest.raises(ValueError, match=message):
        make_model(**overrides)


@pytest.mark.parametrize('scale', [-0.1, 1.1])
def test_scene_coordination_scale_is_bounded(scale):
    with pytest.raises(ValueError, match='between 0 and 1'):
        make_backbone(scale)
