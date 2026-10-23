import pytest
import torch

from modules.backbone import Backbone


def make_backbone(current_radius=8.0):
    return Backbone(
        hidden_dim=8,
        num_historical_steps=2,
        num_future_steps=4,
        duration=2,
        a2a_radius=50.0,
        l2a_radius=50.0,
        num_attn_layers=1,
        num_modes=2,
        num_heads=2,
        dropout=0.0,
        traffic_current_radius=current_radius,
    )


def build_topology_edges(backbone, agent_position):
    empty_edge_index = torch.empty((2, 0), dtype=torch.long)
    return backbone._build_lane_to_agent_topology_edges(
        lane_position=torch.tensor([[0.0, 0.0]]),
        lane_heading=torch.zeros(1),
        lane_batch=torch.zeros(1, dtype=torch.long),
        lane_visible_mask=torch.ones(1, dtype=torch.bool),
        agent_position=agent_position,
        agent_heading=torch.zeros(agent_position.size(0)),
        agent_batch=torch.zeros(agent_position.size(0), dtype=torch.long),
        agent_visible_mask=torch.ones(agent_position.size(0), dtype=torch.bool),
        agent_speed=torch.zeros(agent_position.size(0)),
        agent_lateral_velocity=torch.zeros(agent_position.size(0)),
        successor_edge_index=empty_edge_index,
        left_neighbor_edge_index=empty_edge_index,
        right_neighbor_edge_index=empty_edge_index,
        conflict_edge_index=empty_edge_index,
        potential_conflict_mask=torch.zeros(1),
        lane_turn_direction=torch.zeros(1, dtype=torch.uint8),
    )


def test_current_lane_seed_respects_total_distance_limit():
    backbone = make_backbone(current_radius=8.0)

    near_edge_index, near_weight, _ = build_topology_edges(
        backbone,
        agent_position=torch.tensor([[7.0, 0.0]]),
    )
    far_edge_index, far_weight, _ = build_topology_edges(
        backbone,
        agent_position=torch.tensor([[9.0, 0.0]]),
    )

    assert near_edge_index.tolist() == [[0], [0]]
    assert near_weight.tolist() == [1.0]
    assert far_edge_index.numel() == 0
    assert far_weight.numel() == 0


def test_style_seed_uses_topology_then_stops_inheritance_across_visibility_gap():
    backbone = make_backbone(current_radius=8.0)
    lane_position = torch.tensor([[0.0, 0.0], [10.0, 1.0]])
    lane_heading = torch.zeros(2)
    lane_width = torch.tensor([3.0, 4.0])
    visible_mask = torch.tensor([[True, True, False, True]])

    seed_lane, reference_lane, lateral_distance = backbone._select_style_reference_lanes(
        lane_position=lane_position,
        lane_heading=lane_heading,
        lane_width=lane_width,
        lane_batch=torch.zeros(2, dtype=torch.long),
        lane_visible_mask=torch.ones(2, dtype=torch.bool),
        agent_position=torch.tensor([[
            [0.0, 1.0],
            [19.0, 1.0],
            [0.0, 0.0],
            [19.0, 1.0],
        ]]),
        agent_heading=torch.zeros((1, 4)),
        agent_speed=torch.tensor([[0.0, 20.0, 0.0, 20.0]]),
        agent_batch=torch.zeros(1, dtype=torch.long),
        agent_visible_mask=visible_mask,
        successor_edge_index=torch.tensor([[1], [0]], dtype=torch.long),
        predecessor_edge_index=torch.empty((2, 0), dtype=torch.long),
    )

    assert seed_lane.tolist() == [[0, 1, -1, -1]]
    assert reference_lane.tolist() == [[0, 1, -1, 1]]
    assert torch.allclose(lateral_distance, torch.tensor([[1.0, 0.0, 0.0, 2.0]]))


def test_style_reference_lane_rejects_an_empty_lane_graph():
    backbone = make_backbone()
    empty_edge_index = torch.empty((2, 0), dtype=torch.long)

    with pytest.raises(ValueError, match='at least one lane'):
        backbone._select_style_reference_lanes(
            lane_position=torch.empty((0, 2)),
            lane_heading=torch.empty(0),
            lane_width=torch.empty(0),
            lane_batch=torch.empty(0, dtype=torch.long),
            lane_visible_mask=torch.empty(0, dtype=torch.bool),
            agent_position=torch.zeros((1, 2, 2)),
            agent_heading=torch.zeros((1, 2)),
            agent_speed=torch.zeros((1, 2)),
            agent_batch=torch.zeros(1, dtype=torch.long),
            agent_visible_mask=torch.ones((1, 2), dtype=torch.bool),
            successor_edge_index=empty_edge_index,
            predecessor_edge_index=empty_edge_index,
        )
