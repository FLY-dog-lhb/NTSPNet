import math

import pytest
import torch

from model.PNet import PNet


def make_model():
    return PNet(
        hidden_dim=8,
        num_historical_steps=10,
        num_future_steps=4,
        duration=1,
        a2a_radius=50.0,
        l2a_radius=50.0,
        num_visible_steps=3,
        num_modes=2,
        num_attn_layers=1,
        num_hops=1,
        num_heads=2,
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-4,
        warmup_epochs=1,
        T_max=2,
    )


def test_style_losses_use_hard_pseudo_labels_and_uniform_marginal_kl():
    model = make_model()
    model.Backbone.style_statistics_ready.fill_(True)
    logits = torch.zeros((3, 3), requires_grad=True)
    probability = torch.softmax(logits, dim=-1)
    data = {
        'agent': {
            'style_reliable_mask': torch.tensor([True, True, True]),
            'style_logits': logits,
            'style_probability': probability,
            'style_pseudo_label': torch.tensor([0, 1, 2]),
        },
    }

    cls_loss, balance_loss = model._compute_style_losses(data)

    assert torch.allclose(cls_loss, torch.tensor(math.log(3.0)))
    assert torch.allclose(balance_loss, torch.tensor(0.0), atol=1e-7)
    (
        model.lambda_style
        * (cls_loss + model.style_balance_weight * balance_loss)
    ).backward()
    assert logits.grad is not None


def test_style_loss_requires_precomputed_training_statistics():
    model = make_model()
    data = {
        'agent': {
            'style_reliable_mask': torch.tensor([False]),
            'style_logits': torch.zeros((1, 3)),
            'style_probability': torch.full((1, 3), 1.0 / 3.0),
            'style_pseudo_label': torch.tensor([-1]),
        },
    }

    with pytest.raises(RuntimeError, match='compute_style_statistics.py'):
        model._compute_style_losses(data)


def test_nonuniform_batch_style_distribution_has_positive_balance_loss():
    model = make_model()
    model.Backbone.style_statistics_ready.fill_(True)
    logits = torch.tensor([[3.0, 0.0, 0.0]]).repeat(4, 1)
    data = {
        'agent': {
            'style_reliable_mask': torch.ones(4, dtype=torch.bool),
            'style_logits': logits,
            'style_probability': torch.softmax(logits, dim=-1),
            'style_pseudo_label': torch.zeros(4, dtype=torch.long),
        },
    }

    _, balance_loss = model._compute_style_losses(data)

    assert balance_loss > 0
    assert model.lambda_style == 0.05
    assert model.style_balance_weight == 0.05


def test_style_loss_uses_only_reliable_prefix_anchors():
    model = make_model()
    model.Backbone.style_statistics_ready.fill_(True)
    logits = torch.zeros((2, 4, 3), requires_grad=True)
    probability = torch.softmax(logits, dim=-1)
    reliable = torch.tensor([
        [False, False, False, False],
        [False, False, True, True],
    ])
    pseudo_label = torch.full((2, 4), -1, dtype=torch.long)
    pseudo_label[1, 2:] = torch.tensor([0, 2])
    data = {
        'agent': {
            'style_prefix_reliable_mask': reliable,
            'style_prefix_logits': logits,
            'style_prefix_probability': probability,
            'style_prefix_pseudo_label': pseudo_label,
        },
    }

    cls_loss, balance_loss = model._compute_style_losses(data)

    assert torch.allclose(cls_loss, torch.tensor(math.log(3.0)))
    assert torch.allclose(balance_loss, torch.tensor(0.0), atol=1e-7)
    cls_loss.backward()
    assert torch.equal(logits.grad[~reliable], torch.zeros_like(logits.grad[~reliable]))
    assert torch.any(logits.grad[reliable] != 0)
