import torch
import torch.nn as nn

from modules.backbone import Backbone


class AddOneAttention(nn.Module):
    def forward(self, x, edge_index, edge_attr, attention_bias):
        return x + 1.0


def make_backbone(conflict_update_scope):
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
        conflict_update_scope=conflict_update_scope,
    )


def final_step_bidirectional_edges():
    # Flattened [historical_step, mode, agent] nodes for agents 1 and 3.
    return torch.tensor([
        [17, 19, 21, 23],
        [19, 17, 23, 21],
    ])


def apply_conflict_attention(backbone, edge_index):
    num_agents = 4
    node_embedding = torch.zeros(
        backbone.num_historical_steps
        * backbone.num_modes
        * num_agents,
        backbone.hidden_dim,
    )
    return backbone._apply_conflict_attention(
        node_embedding=node_embedding,
        conflict_edge_index=edge_index,
        conflict_edge_embedding=torch.zeros((edge_index.size(1), 8)),
        priority_head_bias=torch.zeros((edge_index.size(1), 2)),
        attention_layer=AddOneAttention(),
        num_agents=num_agents,
    )


def test_participants_only_updates_exact_conflict_edge_endpoints():
    backbone = make_backbone('participants_only')
    edge_index = final_step_bidirectional_edges()

    output = apply_conflict_attention(backbone, edge_index)

    expected_mask = torch.zeros(output.size(0), dtype=torch.bool)
    expected_mask[torch.tensor([17, 19, 21, 23])] = True
    assert torch.equal(output[expected_mask], torch.ones_like(output[expected_mask]))
    assert torch.equal(output[~expected_mask], torch.zeros_like(output[~expected_mask]))


def test_legacy_scope_updates_every_agent_at_the_final_step():
    backbone = make_backbone('legacy')

    output = apply_conflict_attention(
        backbone,
        final_step_bidirectional_edges(),
    )

    expected_mask = torch.zeros(output.size(0), dtype=torch.bool)
    expected_mask[-backbone.num_modes * 4:] = True
    assert torch.equal(output[expected_mask], torch.ones_like(output[expected_mask]))
    assert torch.equal(output[~expected_mask], torch.zeros_like(output[~expected_mask]))


def test_empty_conflict_graph_is_strict_identity_for_both_scopes():
    empty_edge_index = torch.empty((2, 0), dtype=torch.long)
    for scope in ('legacy', 'participants_only'):
        backbone = make_backbone(scope)
        output = apply_conflict_attention(backbone, empty_edge_index)
        assert torch.equal(output, torch.zeros_like(output))
