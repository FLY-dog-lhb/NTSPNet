import pytest
import torch
import torch.nn as nn

from modules.original_residual_backbone import OriginalResidualBackbone


class RecordingBase(nn.Module):
    def __init__(self, hidden_dim=4, num_modes=2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.last_residuals = None

    def forward(self, data, l_embs, representation_residuals=None):
        self.last_residuals = representation_residuals
        proposal = torch.arange(8, dtype=torch.float).reshape(
            1, 1, 2, 2, 2,
        )
        refinement = proposal + 1.0
        probability = torch.tensor([[[0.75, 0.25]]])
        if representation_residuals is not None:
            proposal = proposal + representation_residuals['proposal'].sum()
            refinement = (
                refinement + representation_residuals['refinement'].sum()
            )
        return proposal, refinement, probability


class ContextProvider(nn.Module):
    def __init__(self, hidden_dim=4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.hidden_dim = hidden_dim

    def forward(self, data, l_embs, return_module_contexts=False):
        assert return_module_contexts
        output = (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )
        contexts = {
            stage: {
                module_name: (
                    torch.arange(
                        2 * self.hidden_dim,
                        dtype=torch.float,
                    ).reshape(1, 2, self.hidden_dim)
                    + module_index
                    + self.weight * 0.0
                )
                for module_index, module_name in enumerate(
                    ('style', 'reachability', 'priority')
                )
            }
            for stage in ('proposal', 'refinement')
        }
        return output, contexts

    def require_style_statistics(self):
        return None


def make_backbone(training_scope='fusion_only', active_modules='all'):
    return OriginalResidualBackbone(
        base=RecordingBase(),
        residual=ContextProvider(),
        hidden_dim=4,
        num_modes=2,
        max_scale=0.1,
        training_scope=training_scope,
        active_modules=active_modules,
    )


def test_zero_module_gates_strictly_preserve_original_outputs():
    backbone = make_backbone()
    expected = backbone.base(data=None, l_embs=None)

    actual = backbone(data=None, l_embs=None)

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)
    for residual in backbone.base.last_residuals.values():
        assert torch.count_nonzero(residual).item() == 0


def test_fusion_only_trains_adapters_and_gates_but_freezes_branches():
    backbone = make_backbone()

    assert not any(
        parameter.requires_grad for parameter in backbone.base.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in backbone.residual.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in backbone.adapters.parameters()
    )
    assert all(gate.requires_grad for gate in backbone.gates.values())

    outputs = backbone(data=None, l_embs=None)
    outputs[0].sum().backward()
    assert backbone.gates['style'].grad is not None
    assert backbone.gates['style'].grad.abs().item() > 0


def test_context_residual_is_broadcast_to_every_original_mode():
    backbone = make_backbone()
    backbone.gates['style'].data.fill_(0.05)

    backbone(data=None, l_embs=None)

    proposal_residual = backbone.base.last_residuals['proposal'].reshape(
        1, 2, 2, 4,
    )
    assert torch.count_nonzero(proposal_residual).item() > 0
    assert torch.equal(
        proposal_residual[:, :, 0],
        proposal_residual[:, :, 1],
    )


def test_residual_branch_scope_keeps_only_original_branch_frozen():
    backbone = make_backbone(training_scope='residual_branch')

    assert not any(
        parameter.requires_grad for parameter in backbone.base.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in backbone.residual.parameters()
    )


def test_module_scales_are_zero_centered_and_bounded():
    backbone = make_backbone()
    backbone.gates['style'].data.fill_(-1.0)
    backbone.gates['reachability'].data.fill_(0.05)
    backbone.gates['priority'].data.fill_(1.0)

    scales = backbone.fusion_scales()

    assert scales['style'].item() == pytest.approx(-0.1, abs=1e-6)
    assert scales['reachability'].item() == pytest.approx(
        0.1 * torch.tanh(torch.tensor(0.5)).item(),
    )
    assert scales['priority'].item() == pytest.approx(0.1, abs=1e-6)


def test_style_only_trains_and_injects_only_style_parameters():
    backbone = make_backbone(active_modules='style')

    trainable_names = {
        name for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_names == {
        'gates.style',
        'adapters.proposal.style.0.weight',
        'adapters.proposal.style.0.bias',
        'adapters.proposal.style.1.weight',
        'adapters.proposal.style.1.bias',
        'adapters.refinement.style.0.weight',
        'adapters.refinement.style.0.bias',
        'adapters.refinement.style.1.weight',
        'adapters.refinement.style.1.bias',
    }

    backbone.gates['reachability'].data.fill_(1.0)
    backbone.gates['priority'].data.fill_(-1.0)
    scales = backbone.fusion_scales()
    assert scales['style'].item() == 0.0
    assert scales['reachability'].item() == 0.0
    assert scales['priority'].item() == 0.0

    expected = backbone.base(data=None, l_embs=None)
    actual = backbone(data=None, l_embs=None)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)

    backbone.gates['style'].data.fill_(0.05)
    backbone(data=None, l_embs=None)
    assert any(
        torch.count_nonzero(residual).item() > 0
        for residual in backbone.base.last_residuals.values()
    )


def test_style_only_does_not_require_inactive_contexts():
    backbone = make_backbone(active_modules='style')
    contexts = {
        stage: {'style': torch.zeros(1, 2, 4)}
        for stage in ('proposal', 'refinement')
    }

    residuals = backbone._adapt_contexts(contexts)

    assert set(residuals) == {'proposal', 'refinement'}


@pytest.mark.parametrize(
    ('selection', 'expected_modules', 'inactive_module'),
    [
        (
            'style_reachability',
            ('style', 'reachability'),
            'priority',
        ),
        ('style_priority', ('style', 'priority'), 'reachability'),
        (
            'reachability_priority',
            ('reachability', 'priority'),
            'style',
        ),
    ],
)
def test_two_module_selection_trains_and_injects_only_selected_modules(
        selection, expected_modules, inactive_module):
    backbone = make_backbone(active_modules=selection)

    assert backbone.active_modules == expected_modules
    trainable_names = {
        name for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    }
    for module_name in expected_modules:
        assert f'gates.{module_name}' in trainable_names
    assert not any(inactive_module in name for name in trainable_names)
    assert len(trainable_names) == 18

    backbone.gates[expected_modules[0]].data.fill_(0.05)
    backbone.gates[expected_modules[1]].data.fill_(-0.05)
    backbone.gates[inactive_module].data.fill_(1.0)
    scales = backbone.fusion_scales()
    assert scales[expected_modules[0]].item() > 0
    assert scales[expected_modules[1]].item() < 0
    assert scales[inactive_module].item() == 0.0

    contexts = {
        stage: {
            module_name: torch.zeros(1, 2, 4)
            for module_name in expected_modules
        }
        for stage in ('proposal', 'refinement')
    }
    residuals = backbone._adapt_contexts(contexts)
    assert set(residuals) == {'proposal', 'refinement'}


@pytest.mark.parametrize('active_modules', [(), ('style', 'style'), 'unknown'])
def test_invalid_active_modules_fail_closed(active_modules):
    with pytest.raises(ValueError):
        make_backbone(active_modules=active_modules)


def test_module_subset_rejects_residual_branch_training():
    with pytest.raises(ValueError, match='fusion_only'):
        make_backbone(
            training_scope='residual_branch',
            active_modules='style',
        )


def test_missing_module_context_fails_closed():
    backbone = make_backbone()
    contexts = {
        stage: {
            name: torch.zeros(1, 2, 4)
            for name in ('style', 'reachability', 'priority')
        }
        for stage in ('proposal', 'refinement')
    }
    del contexts['proposal']['priority']

    with pytest.raises(ValueError, match='proposal/priority'):
        backbone._adapt_contexts(contexts)
