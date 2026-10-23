import pytest
import torch

from model import PNet
from modules.backbone import Backbone


def make_backbone(
        scale=1.0,
        assignment_mode='independent',
        max_exception_agents=2):
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
        joint_mode_composition_scale=scale,
        joint_mode_assignment_mode=assignment_mode,
        joint_mode_max_exception_agents=max_exception_agents,
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
        joint_mode_composition_scale=1.0,
        proposal_conflict_training_scope='joint_composition_only',
    )
    configuration.update(overrides)
    return PNet(**configuration)


def composition_inputs(backbone):
    num_agents = 3
    node_embedding = torch.randn(
        num_agents
        * backbone.num_historical_steps
        * backbone.num_modes,
        backbone.hidden_dim,
    )
    shape = (
        num_agents,
        backbone.num_historical_steps,
        backbone.num_modes,
        backbone.num_future_steps,
        2,
    )
    proposal = torch.randn(shape)
    refinement = torch.randn(shape)
    agent_batch = torch.tensor([0, 0, 1], dtype=torch.long)
    valid_mask = torch.ones(shape[:3], dtype=torch.bool)
    return node_embedding, proposal, refinement, agent_batch, valid_mask


def test_zero_initialized_hard_composer_is_exact_identity_with_gradient():
    backbone = make_backbone()
    inputs = composition_inputs(backbone)
    proposal, refinement = backbone._compose_joint_modes(
        *inputs, num_scenes=2,
    )
    refinement[0, 0, 0].sum().backward()

    assert torch.equal(proposal, inputs[1])
    assert torch.equal(refinement, inputs[2])
    assert torch.count_nonzero(backbone.joint_mode_composer.weight) == 0
    assert torch.count_nonzero(backbone.joint_mode_composer.bias) == 0
    assert backbone.joint_mode_composer.weight.grad is not None
    assert torch.any(backbone.joint_mode_composer.weight.grad != 0)
    assert backbone.joint_mode_composer.bias.grad is not None
    assert torch.any(backbone.joint_mode_composer.bias.grad != 0)


def test_permutation_composer_is_exact_identity_with_finite_gradient():
    backbone = make_backbone(assignment_mode='permutation')
    inputs = composition_inputs(backbone)
    proposal, refinement = backbone._compose_joint_modes(
        *inputs, num_scenes=2,
    )
    refinement[0, 0, 0].sum().backward()

    assert torch.equal(proposal, inputs[1])
    assert torch.equal(refinement, inputs[2])
    assert backbone.joint_mode_composer.weight.grad is not None
    assert torch.isfinite(backbone.joint_mode_composer.weight.grad).all()
    assert torch.any(backbone.joint_mode_composer.weight.grad != 0)
    assert backbone.joint_mode_composer.bias.grad is not None
    assert torch.isfinite(backbone.joint_mode_composer.bias.grad).all()
    assert torch.allclose(
        backbone.joint_mode_composer.bias.grad,
        torch.zeros_like(backbone.joint_mode_composer.bias.grad),
        atol=1e-6,
    )


def test_limited_permutation_is_exact_identity_with_finite_gradient():
    backbone = make_backbone(assignment_mode='limited_permutation')
    inputs = composition_inputs(backbone)
    proposal, refinement = backbone._compose_joint_modes(
        *inputs, num_scenes=2,
    )
    refinement[0, 0, 0].sum().backward()

    assert torch.equal(proposal, inputs[1])
    assert torch.equal(refinement, inputs[2])
    gradient = backbone.joint_mode_composer.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.any(gradient != 0)


def test_sinkhorn_assignment_is_doubly_stochastic():
    logits = torch.randn(3, 4, 4)
    assignment = Backbone._sinkhorn_joint_mode_assignment(
        logits, num_iterations=20,
    )

    assert torch.allclose(
        assignment.sum(dim=-1), torch.ones(3, 4), atol=1e-4,
    )
    assert torch.allclose(
        assignment.sum(dim=-2), torch.ones(3, 4), atol=1e-4,
    )


def test_permutation_composer_reorders_without_duplicate_local_modes():
    backbone = make_backbone(assignment_mode='permutation')
    inputs = list(composition_inputs(backbone))
    node_view = inputs[0].reshape(3, 3, 2, 8)
    node_view.zero_()
    node_view[:, :, 0, 0] = 1.0
    node_view[:, :, 1, 1] = 1.0
    for trajectory in inputs[1:3]:
        trajectory[:, :, 0].fill_(10.0)
        trajectory[:, :, 1].fill_(20.0)
    with torch.no_grad():
        backbone.joint_mode_composer.weight.zero_()
        backbone.joint_mode_composer.bias.zero_()
        backbone.joint_mode_composer.weight[1, 0] = 2.0
        backbone.joint_mode_composer.weight[0, 1] = 2.0

    proposal, refinement = backbone._compose_joint_modes(
        *inputs, num_scenes=2,
    )

    assert torch.equal(proposal[:, :, 0], inputs[1][:, :, 1])
    assert torch.equal(proposal[:, :, 1], inputs[1][:, :, 0])
    assert torch.equal(refinement[:, :, 0], inputs[2][:, :, 1])
    assert torch.equal(refinement[:, :, 1], inputs[2][:, :, 0])


def test_permutation_composer_preserves_partly_invalid_anchor():
    backbone = make_backbone(assignment_mode='permutation')
    inputs = list(composition_inputs(backbone))
    inputs[4][0, 0, 0] = False
    with torch.no_grad():
        backbone.joint_mode_composer.weight.normal_()
        backbone.joint_mode_composer.bias.normal_()

    proposal, refinement = backbone._compose_joint_modes(
        *inputs, num_scenes=2,
    )

    assert torch.equal(proposal[0, 0], inputs[1][0, 0])
    assert torch.equal(refinement[0, 0], inputs[2][0, 0])


def test_limited_permutation_caps_exceptions_and_respects_agent_mask():
    backbone = make_backbone(
        assignment_mode='limited_permutation',
        max_exception_agents=1,
    )
    inputs = list(composition_inputs(backbone))
    node_view = inputs[0].reshape(3, 3, 2, 8)
    node_view.zero_()
    node_view[:, :, 0, 0] = 1.0
    node_view[:, :, 1, 1] = 1.0
    for trajectory in inputs[1:3]:
        trajectory[:, :, 0].fill_(10.0)
        trajectory[:, :, 1].fill_(20.0)
    with torch.no_grad():
        backbone.joint_mode_composer.weight.zero_()
        backbone.joint_mode_composer.bias.zero_()
        backbone.joint_mode_composer.weight[1, 0] = 2.0
        backbone.joint_mode_composer.weight[0, 1] = 2.0

    proposal, refinement = backbone._compose_joint_modes(
        *inputs,
        num_scenes=2,
        exception_agent_mask=torch.tensor([True, False, False]),
    )

    assert torch.equal(proposal[0, :, 0], inputs[1][0, :, 1])
    assert torch.equal(proposal[0, :, 1], inputs[1][0, :, 0])
    assert torch.equal(refinement[0, :, 0], inputs[2][0, :, 1])
    assert torch.equal(refinement[0, :, 1], inputs[2][0, :, 0])
    assert torch.equal(proposal[1:], inputs[1][1:])
    assert torch.equal(refinement[1:], inputs[2][1:])


def test_limited_permutation_selects_highest_gain_per_scene_and_anchor():
    backbone = make_backbone(
        assignment_mode='limited_permutation',
        max_exception_agents=1,
    )
    hard_index = torch.tensor([
        [[1, 0], [1, 0], [1, 0]],
        [[1, 0], [1, 0], [1, 0]],
        [[1, 0], [1, 0], [1, 0]],
        [[1, 0], [1, 0], [1, 0]],
    ])
    gain = torch.tensor([
        [1.0, 4.0, 1.0],
        [3.0, 2.0, 5.0],
        [7.0, 1.0, 2.0],
        [2.0, 6.0, 3.0],
    ])
    limited = backbone._limit_joint_mode_permutation_exceptions(
        hard_index=hard_index,
        permutation_gain=gain,
        agent_batch=torch.tensor([0, 0, 1, 1]),
        anchor_valid=torch.ones(4, 3, dtype=torch.bool),
        exception_agent_mask=torch.tensor([True, True, False, True]),
        num_scenes=2,
    )
    nonidentity = (limited != torch.tensor([0, 1])).any(dim=-1)

    assert torch.equal(nonidentity[:, 0], torch.tensor([False, True, False, True]))
    assert torch.equal(nonidentity[:, 1], torch.tensor([True, False, False, True]))
    assert torch.equal(nonidentity[:, 2], torch.tensor([False, True, False, True]))


def test_composer_uses_hard_existing_trajectory_and_keeps_invalid_rows():
    backbone = make_backbone()
    inputs = list(composition_inputs(backbone))
    inputs[4][0, 0, 0] = False
    with torch.no_grad():
        backbone.joint_mode_composer.bias[1] = 1.0

    proposal, refinement = backbone._compose_joint_modes(
        *inputs, num_scenes=2,
    )

    assert torch.equal(proposal[0, 0, 0], inputs[1][0, 0, 0])
    assert torch.equal(refinement[0, 0, 0], inputs[2][0, 0, 0])
    assert torch.equal(proposal[0, 0, 1], inputs[1][0, 0, 1])
    assert torch.equal(refinement[0, 0, 1], inputs[2][0, 0, 1])
    assert torch.equal(proposal[1:, :, 0], inputs[1][1:, :, 1])
    assert torch.equal(refinement[1:, :, 0], inputs[2][1:, :, 1])


def test_pre_composer_state_dict_strictly_loads_as_zero():
    source = make_backbone(scale=0.0)
    old_state = source.state_dict()
    old_state.pop('joint_mode_composer.weight')
    old_state.pop('joint_mode_composer.bias')
    target = make_backbone(scale=1.0)
    with torch.no_grad():
        target.joint_mode_composer.weight.fill_(3.0)
        target.joint_mode_composer.bias.fill_(3.0)

    target.load_state_dict(old_state, strict=True)

    assert torch.count_nonzero(target.joint_mode_composer.weight) == 0
    assert torch.count_nonzero(target.joint_mode_composer.bias) == 0


def test_joint_composition_scope_trains_only_new_composer():
    model = make_model()
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert set(trainable) == {
        'Backbone.joint_mode_composer.weight',
        'Backbone.joint_mode_composer.bias',
    }
    assert sum(parameter.numel() for parameter in trainable.values()) == 34


def test_permutation_scope_excludes_unidentifiable_composer_bias():
    model = make_model(joint_mode_assignment_mode='permutation')
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert set(trainable) == {'Backbone.joint_mode_composer.weight'}
    assert sum(parameter.numel() for parameter in trainable.values()) == 32


def test_limited_permutation_scope_excludes_composer_bias():
    model = make_model(joint_mode_assignment_mode='limited_permutation')
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert set(trainable) == {'Backbone.joint_mode_composer.weight'}
    assert sum(parameter.numel() for parameter in trainable.values()) == 32


@pytest.mark.parametrize(
    'overrides, message',
    [
        (
            {'joint_mode_composition_scale': 0.0},
            'requires a non-zero',
        ),
        (
            {'scene_mode_coordination_scale': 0.1},
            'earlier experimental branches',
        ),
        (
            {'trunk_variant': 'original'},
            'requires the extended trunk',
        ),
    ],
)
def test_joint_composition_scope_rejects_confounded_configuration(
        overrides, message):
    with pytest.raises(ValueError, match=message):
        make_model(**overrides)


@pytest.mark.parametrize('scale', [-0.1, 1.1])
def test_joint_mode_composition_scale_is_bounded(scale):
    with pytest.raises(ValueError, match='between 0 and 1'):
        make_backbone(scale)


def test_joint_mode_assignment_mode_is_validated():
    with pytest.raises(ValueError, match='independent.*permutation'):
        make_backbone(assignment_mode='invalid')


@pytest.mark.parametrize('maximum', [0, -1])
def test_joint_mode_max_exception_agents_is_positive(maximum):
    with pytest.raises(ValueError, match='must be positive'):
        make_backbone(max_exception_agents=maximum)
