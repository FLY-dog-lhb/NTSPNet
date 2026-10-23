import torch

from modules.lane_reachability import LaneAssociation
from modules.lane_reachability import LaneReachability
from modules.lane_reachability import LaneReachabilityBuilder


def empty_edge():
    return torch.empty((2, 0), dtype=torch.long)


def centerline_segments(y_values):
    positions = []
    lane_index = []
    for lane_idx, y_value in enumerate(y_values):
        for x_value in (1.0, 3.0, 5.0, 7.0, 9.0):
            positions.append((x_value, y_value))
            lane_index.append(lane_idx)
    num_segments = len(positions)
    return {
        'position': torch.tensor(positions),
        'heading': torch.zeros(num_segments),
        'length': torch.full((num_segments,), 2.0),
        'side': torch.ones(num_segments, dtype=torch.uint8),
        'edge_index': torch.stack([
            torch.arange(num_segments),
            torch.tensor(lane_index),
        ]),
    }


def test_unique_main_lane_and_auxiliary_seed_use_hard_filtered_candidates():
    builder = LaneReachabilityBuilder()
    segment = centerline_segments([0.0, 2.0])
    association = builder.associate(
        agent_position=torch.tensor([[[2.0, 0.9], [4.0, 1.1]]]),
        agent_heading=torch.zeros((1, 2)),
        agent_speed=torch.ones((1, 2)),
        agent_heading_valid_mask=torch.ones((1, 2), dtype=torch.bool),
        agent_visible_mask=torch.ones((1, 2), dtype=torch.bool),
        agent_batch=torch.zeros(1, dtype=torch.long),
        lane_heading=torch.zeros(2),
        lane_width=torch.full((2,), 4.0),
        lane_drivable=torch.ones(2, dtype=torch.bool),
        lane_visible_mask=torch.ones(2, dtype=torch.bool),
        lane_batch=torch.zeros(2, dtype=torch.long),
        lane_centerline_spacing=torch.full((2,), 2.0),
        predecessor_edge_index=empty_edge(),
        successor_edge_index=empty_edge(),
        left_change_edge_index=torch.tensor([[1], [0]]),
        right_change_edge_index=empty_edge(),
        adjacent_left_edge_index=empty_edge(),
        adjacent_right_edge_index=empty_edge(),
        polyline_position=segment['position'],
        polyline_heading=segment['heading'],
        polyline_length=segment['length'],
        polyline_side=segment['side'],
        polyline_to_lane_edge_index=segment['edge_index'],
    )

    # At t=1 lane 1 is geometrically closer, but causal continuity keeps lane 0.
    assert association.main_lane.tolist() == [[0, 0]]
    assert association.auxiliary_seed_edge_index.tolist() == [[0, 1], [1, 1]]
    assert torch.allclose(association.projection_s, torch.tensor([[2.0, 4.0]]))


def test_stationary_vehicle_does_not_use_heading_as_a_hard_constraint():
    builder = LaneReachabilityBuilder()
    segment = centerline_segments([0.0])
    association = builder.associate(
        agent_position=torch.tensor([[[5.0, 0.0]]]),
        agent_heading=torch.tensor([[torch.pi]]),
        agent_speed=torch.tensor([[0.5]]),
        agent_heading_valid_mask=torch.ones((1, 1), dtype=torch.bool),
        agent_visible_mask=torch.ones((1, 1), dtype=torch.bool),
        agent_batch=torch.zeros(1, dtype=torch.long),
        lane_heading=torch.zeros(1),
        lane_width=torch.tensor([3.5]),
        lane_drivable=torch.ones(1, dtype=torch.bool),
        lane_visible_mask=torch.ones(1, dtype=torch.bool),
        lane_batch=torch.zeros(1, dtype=torch.long),
        lane_centerline_spacing=torch.tensor([2.0]),
        predecessor_edge_index=empty_edge(),
        successor_edge_index=empty_edge(),
        left_change_edge_index=empty_edge(),
        right_change_edge_index=empty_edge(),
        adjacent_left_edge_index=empty_edge(),
        adjacent_right_edge_index=empty_edge(),
        polyline_position=segment['position'],
        polyline_heading=segment['heading'],
        polyline_length=segment['length'],
        polyline_side=segment['side'],
        polyline_to_lane_edge_index=segment['edge_index'],
    )

    assert association.main_lane.item() == 0


def test_reachability_uses_remaining_start_length_successors_and_one_lane_change():
    builder = LaneReachabilityBuilder()
    association = LaneAssociation(
        main_lane=torch.tensor([[0]]),
        projection_s=torch.tensor([[4.0]]),
        lateral_distance=torch.zeros((1, 1)),
        heading_difference=torch.zeros((1, 1)),
        auxiliary_seed_edge_index=empty_edge(),
    )
    reachability = builder.build_reachability(
        association=association,
        agent_velocity=torch.tensor([[[2.0, 0.0]]]),
        agent_acceleration=torch.zeros((1, 1, 2)),
        agent_visible_mask=torch.ones((1, 1), dtype=torch.bool),
        lane_heading=torch.zeros(3),
        lane_length=torch.tensor([10.0, 10.0, 10.0]),
        lane_start_position=torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 3.0]]),
        lane_end_position=torch.tensor([[10.0, 0.0], [20.0, 0.0], [10.0, 3.0]]),
        lane_drivable=torch.ones(3, dtype=torch.bool),
        lane_visible_mask=torch.ones(3, dtype=torch.bool),
        successor_edge_index=torch.tensor([[1], [0]]),
        lane_change_pair_index=torch.tensor([[0], [2]]),
    )

    distance_by_lane = {
        int(lane): float(distance)
        for lane, distance in zip(
            reachability.edge_index[0], reachability.path_distance,
        )
    }
    assert reachability.distance_budget.item() == 15.0
    assert distance_by_lane == {0: 0.0, 1: 6.0, 2: 12.0}
    assert reachability.lane_change_count.tolist() == [0, 0, 1]


def test_conflict_instances_use_raw_tau_and_expand_canonical_pair_bidirectionally():
    builder = LaneReachabilityBuilder()
    reachability = LaneReachability(
        edge_index=torch.tensor([[2, 2], [0, 1]], dtype=torch.long),
        path_distance=torch.tensor([10.0, 12.0]),
        lane_change_count=torch.zeros(2, dtype=torch.long),
        distance_budget=torch.full((2, 1), 20.0),
        longitudinal_speed=torch.tensor([[2.0], [3.0]]),
    )
    conflict = builder.build_conflict_instances(
        reachability=reachability,
        potential_conflict_mask=torch.tensor([False, False, True]),
        agent_batch=torch.zeros(2, dtype=torch.long),
        num_historical_steps=1,
    )

    assert conflict.pair_index.tolist() == [[0], [1]]
    assert conflict.conflict_lane.tolist() == [2]
    assert torch.allclose(conflict.tau_raw_i, torch.tensor([5.0]))
    assert torch.allclose(conflict.tau_raw_j, torch.tensor([4.0]))
    assert conflict.valid_mask.tolist() == [True]
    assert conflict.canonical_pair_index.tolist() == [[0], [1]]
    assert conflict.E_VV_conf.tolist() == [[0, 1], [1, 0]]
    assert torch.allclose(
        conflict.directed_edge_attr[:, :2],
        torch.tensor([[5.0, 4.0], [4.0, 5.0]]),
    )
