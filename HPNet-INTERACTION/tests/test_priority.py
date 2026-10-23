import torch

from modules.lane_reachability import ConflictInstances
from modules.priority import PriorityModule


def test_priority_rule_feature_order_is_fixed():
    assert PriorityModule.RULE_FEATURE_NAMES == (
        'r_i_priority',
        'r_j_priority',
        'r_i_yield',
        'r_j_yield',
        'r_i_controlled',
        'r_j_controlled',
    )


def make_conflict_instances(num_instances=2):
    pair_index = torch.tensor([[0] * num_instances, [1] * num_instances])
    scalar = torch.arange(1, num_instances + 1, dtype=torch.float)
    canonical_attr = torch.tensor([[1.0, 2.0, 1.0, 3.0, 4.0, 0.2, float(num_instances)]])
    return ConflictInstances(
        pair_index=pair_index,
        conflict_lane=torch.zeros(num_instances, dtype=torch.long),
        distance_i=scalar + 2.0,
        distance_j=scalar + 3.0,
        tau_raw_i=scalar,
        tau_raw_j=scalar + 1.0,
        tau_clip_i=scalar.clamp(max=5.0),
        tau_clip_j=(scalar + 1.0).clamp(max=5.0),
        urgency_cost=torch.tensor([0.1, 0.2])[:num_instances],
        valid_mask=torch.ones(num_instances, dtype=torch.bool),
        pair_group=torch.zeros(num_instances, dtype=torch.long),
        canonical_pair_index=torch.tensor([[0], [1]]),
        canonical_edge_attr=canonical_attr,
        E_VV_conf=torch.tensor([[0, 1], [1, 0]]),
        directed_edge_attr=torch.cat([
            canonical_attr,
            canonical_attr[:, [1, 0, 2, 4, 3, 5, 6]],
        ]),
    )


def test_priority_aggregates_instances_and_forces_reverse_probability_permutation():
    torch.manual_seed(5)
    module = PriorityModule(hidden_dim=4, num_heads=2).eval()
    conflict = make_conflict_instances()
    polyline_position = torch.tensor([[5.0, 1.0], [5.0, -1.0]])
    polyline_heading = torch.zeros(2)
    polyline_length = torch.full((2,), 10.0)
    polyline_side = torch.tensor([0, 2], dtype=torch.uint8)
    p2l = torch.tensor([[0, 1], [0, 0]])

    with torch.no_grad():
        output = module(
            conflict_instances=conflict,
            style_embedding=torch.randn((2, 4)),
            longitudinal_speed=torch.tensor([2.0, 2.0]),
            longitudinal_acceleration=torch.zeros(2),
            conflict_type_feature=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            rule_feature=torch.zeros((2, 6)),
            historical_position=torch.tensor([[[-1.0, 0.0]], [[-3.0, 0.0]]]),
            future_position=torch.tensor([
                [[1.0, 0.0], [2.0, 0.0]],
                [[-2.0, 0.0], [1.0, 0.0]],
            ]),
            historical_visible_mask=torch.ones((2, 1), dtype=torch.bool),
            future_visible_mask=torch.ones((2, 2), dtype=torch.bool),
            polyline_position=polyline_position,
            polyline_heading=polyline_heading,
            polyline_length=polyline_length,
            polyline_side=polyline_side,
            polyline_to_lane_edge_index=p2l,
            step_time=1.0,
        )

    assert output.instance_logits.shape == (2, 3)
    assert output.pair_probability.shape == (1, 3)
    assert torch.allclose(output.pair_probability.sum(-1), torch.ones(1))
    assert torch.allclose(
        output.directed_pair_probability[1],
        output.directed_pair_probability[0, [1, 0, 2]],
    )
    assert output.directed_head_bias.shape == (2, 2)
    assert output.pseudo_label.tolist() == [0, 0]
    assert output.pseudo_label_valid_mask.tolist() == [True, True]
    assert torch.all(output.order_target > 0)


def test_priority_ignores_instance_when_one_vehicle_never_enters_lanelet():
    module = PriorityModule(hidden_dim=4, num_heads=2).eval()
    conflict = make_conflict_instances(num_instances=1)
    with torch.no_grad():
        output = module(
            conflict_instances=conflict,
            style_embedding=torch.zeros((2, 4)),
            longitudinal_speed=torch.ones(2),
            longitudinal_acceleration=torch.zeros(2),
            conflict_type_feature=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            rule_feature=torch.zeros((1, 6)),
            historical_position=torch.tensor([[[-1.0, 0.0]], [[-3.0, 0.0]]]),
            future_position=torch.tensor([
                [[1.0, 0.0]],
                [[-2.0, 0.0]],
            ]),
            historical_visible_mask=torch.ones((2, 1), dtype=torch.bool),
            future_visible_mask=torch.ones((2, 1), dtype=torch.bool),
            polyline_position=torch.tensor([[5.0, 1.0], [5.0, -1.0]]),
            polyline_heading=torch.zeros(2),
            polyline_length=torch.full((2,), 10.0),
            polyline_side=torch.tensor([0, 2], dtype=torch.uint8),
            polyline_to_lane_edge_index=torch.tensor([[0, 1], [0, 0]]),
            step_time=1.0,
        )

    assert output.pseudo_label.tolist() == [-1]
    assert output.pseudo_label_valid_mask.tolist() == [False]
