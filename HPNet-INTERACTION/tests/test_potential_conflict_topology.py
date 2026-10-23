import torch

from modules.lane_topology import build_potential_conflict_topology


def edge(pairs):
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def test_builds_merge_split_crossing_and_directed_lane_change_regions():
    # Merge: 0,1 -> 2. Split: 3 -> 4,5. Changeable: 8 -> 9.
    lane_heading = torch.tensor([
        0.0, 0.1, 0.05,
        0.0, -0.2, 0.2,
        0.0, 0.0,
        0.0, 0.0,
    ])
    result = build_potential_conflict_topology(
        lane_heading=lane_heading,
        lane_drivable=torch.ones(10, dtype=torch.bool),
        predecessor_edge_index=edge([(0, 2), (1, 2)]),
        successor_edge_index=edge([(4, 3), (5, 3)]),
        left_change_edge_index=edge([(9, 8)]),
        right_change_edge_index=edge([]),
        adjacent_left_edge_index=edge([]),
        adjacent_right_edge_index=edge([]),
        crossing_edge_index=edge([(6, 7), (7, 6)]),
    )

    assert result['merge_core_mask'].tolist() == [
        False, False, True, False, False, False, False, False, False, False,
    ]
    assert result['merge_region_mask'][:3].tolist() == [True, True, True]
    assert result['split_core_mask'][3]
    assert result['split_region_mask'][3:6].tolist() == [True, True, True]
    assert result['crossing_conflict_mask'][6:8].tolist() == [True, True]
    assert result['lane_change_pair_index'].tolist() == [[8], [9]]
    assert result['lane_change_critical_mask'][8:10].tolist() == [True, True]
    assert result['potential_conflict_mask'].all()


def test_nonchangeable_adjacency_can_trigger_boundary_but_never_enters_qlc():
    # The physical adjacency around predecessor 0/1 makes adjacency-start false,
    # but only the changeable relation 2 -> 3 is eligible for Qlc.
    result = build_potential_conflict_topology(
        lane_heading=torch.zeros(4),
        lane_drivable=torch.ones(4, dtype=torch.bool),
        predecessor_edge_index=edge([(0, 2), (1, 3)]),
        successor_edge_index=edge([]),
        left_change_edge_index=edge([(3, 2)]),
        right_change_edge_index=edge([]),
        adjacent_left_edge_index=edge([(1, 0)]),
        adjacent_right_edge_index=edge([]),
        crossing_edge_index=edge([]),
    )

    # No successors means the adjacency ends, so the directed changeable pair
    # remains critical. The nonchangeable 0 -> 1 pair is never emitted.
    assert result['lane_change_pair_index'].tolist() == [[2], [3]]
    assert not torch.any(
        (result['lane_change_pair_index'][0] == 0)
        & (result['lane_change_pair_index'][1] == 1)
    )


def test_non_drivable_lanes_are_excluded_from_all_conflict_regions():
    result = build_potential_conflict_topology(
        lane_heading=torch.zeros(2),
        lane_drivable=torch.tensor([True, False]),
        predecessor_edge_index=edge([]),
        successor_edge_index=edge([]),
        left_change_edge_index=edge([(1, 0)]),
        right_change_edge_index=edge([]),
        adjacent_left_edge_index=edge([]),
        adjacent_right_edge_index=edge([]),
        crossing_edge_index=edge([(0, 1)]),
    )

    assert result['crossing_conflict_mask'].tolist() == [True, False]
    assert result['lane_change_pair_index'].numel() == 0
    assert result['potential_conflict_mask'].tolist() == [True, False]
