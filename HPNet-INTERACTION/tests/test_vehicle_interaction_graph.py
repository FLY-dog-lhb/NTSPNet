import torch

from modules.vehicle_interaction_graph import VehicleInteractionGraphBuilder


def make_builder(**kwargs):
    config = dict(
        distance_margin=2.0,
        observation_window=0.0,
        ttc_threshold=3.0,
        collision_distance=2.0,
        max_distance=50.0,
    )
    config.update(kwargs)
    return VehicleInteractionGraphBuilder(**config)


def empty_map():
    return (
        torch.empty((0, 2)),
        torch.empty(0, dtype=torch.long),
        torch.empty((2, 0), dtype=torch.long),
    )


def test_distance_builds_bidirectional_edges_at_each_visible_step():
    builder = make_builder(distance_margin=5.0)
    position = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0]],
        [[4.0, 0.0], [5.0, 0.0]],
    ])
    velocity = torch.zeros_like(position)
    lane_position, lane_batch, conflict = empty_map()

    graph = builder(
        position,
        velocity,
        torch.ones((2, 2), dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
    )

    assert graph.E_VV_msg.tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]
    assert graph.edge_index.data_ptr() == graph.E_VV_msg.data_ptr()
    assert torch.all(graph.edge_attr[:, 2] == 1)
    assert torch.all(graph.edge_attr[:, 3:] == 0)


def test_ttc_builds_an_edge_outside_distance_threshold():
    builder = make_builder()
    position = torch.tensor([[[0.0, 0.0]], [[10.0, 0.0]]])
    velocity = torch.tensor([[[2.0, 0.0]], [[-2.0, 0.0]]])
    lane_position, lane_batch, conflict = empty_map()

    graph = builder(
        position,
        velocity,
        torch.ones((2, 1), dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
    )

    assert graph.E_VV_msg.size(1) == 2
    assert torch.allclose(graph.edge_attr[:, 1], torch.full((2,), 2.0))
    assert torch.all(graph.edge_attr[:, 2] == 0)
    assert torch.all(graph.edge_attr[:, 3] == 1)


def test_parallel_motion_does_not_create_a_false_ttc_edge():
    builder = make_builder()
    position = torch.tensor([[[0.0, 0.0]], [[10.0, 0.0]]])
    velocity = torch.tensor([[[2.0, 0.0]], [[2.0, 0.0]]])
    lane_position, lane_batch, conflict = empty_map()

    graph = builder(
        position,
        velocity,
        torch.ones((2, 1), dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
    )

    assert graph.E_VV_msg.numel() == 0
    assert graph.D_VV_msg.numel() == 0
    assert graph.A_VV_msg.shape == (2, 2)
    assert not graph.A_VV_msg.any()
    assert all(neighbors.numel() == 0 for neighbors in graph.N_VV_msg)
    assert torch.equal(graph.degree_centrality, torch.zeros(2))
    assert torch.equal(graph.closeness_centrality, torch.zeros(2))


def test_distance_threshold_uses_target_vehicle_speed():
    builder = make_builder(distance_margin=1.0, observation_window=1.0)
    position = torch.tensor([[[0.0, 0.0]], [[5.0, 0.0]]])
    velocity = torch.tensor([[[0.0, 10.0]], [[0.0, 0.0]]])
    lane_position, lane_batch, conflict = empty_map()

    graph = builder(
        position,
        velocity,
        torch.ones((2, 1), dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
    )

    assert graph.E_VV_msg.tolist() == [[1], [0]]
    assert graph.edge_attr[0, 2] == 1


def test_conflict_distance_adjacency_and_incoming_neighborhood():
    builder = make_builder(distance_margin=1.0, observation_window=1.0)
    position = torch.tensor([[[0.0, 0.0]], [[5.0, 0.0]]])
    velocity = torch.tensor([[[0.0, 10.0]], [[0.0, 0.0]]])
    lane_position, lane_batch, conflict = empty_map()

    graph = builder(
        position,
        velocity,
        torch.ones((2, 1), dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
    )

    assert torch.allclose(graph.D_VV_msg, torch.tensor([5.0]))
    assert graph.A_VV_msg.tolist() == [[False, True], [False, False]]
    assert graph.N_VV_msg[0].tolist() == [1]
    assert graph.N_VV_msg[1].numel() == 0
    assert torch.allclose(graph.edge_attr[:, 0], graph.D_VV_msg)
    assert torch.allclose(graph.degree_centrality, torch.tensor([1.0, 0.0]))
    assert torch.allclose(graph.closeness_centrality, torch.tensor([0.2, 0.0]))


def test_centrality_uses_weighted_multi_hop_paths():
    builder = make_builder(distance_margin=2.1, collision_distance=0.0)
    position = torch.tensor([[[0.0, 0.0]], [[2.0, 0.0]], [[4.0, 0.0]]])
    velocity = torch.zeros_like(position)
    lane_position, lane_batch, conflict = empty_map()

    graph = builder(
        position,
        velocity,
        torch.ones((3, 1), dtype=torch.bool),
        torch.zeros(3, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
    )

    assert torch.allclose(graph.degree_centrality, torch.tensor([0.5, 1.0, 0.5]))
    assert torch.allclose(
        graph.closeness_centrality,
        torch.tensor([1.0 / 3.0, 0.5, 1.0 / 3.0]),
    )


def test_centrality_changes_use_causal_backward_differences():
    builder = make_builder(distance_margin=1.1, collision_distance=0.0)
    position = torch.tensor([
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[10.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[20.0, 0.0], [20.0, 0.0], [2.0, 0.0]],
    ])
    velocity = torch.zeros_like(position)
    lane_position, lane_batch, conflict = empty_map()

    graph = builder(
        position,
        velocity,
        torch.ones((3, 3), dtype=torch.bool),
        torch.zeros(3, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
    )

    degree_change = graph.degree_centrality_change.reshape(3, 3)
    degree_second_change = graph.degree_centrality_second_change.reshape(3, 3)
    assert torch.allclose(
        degree_change,
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]),
    )
    assert torch.allclose(
        degree_second_change,
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.5, 0.0, 0.5]]),
    )

    closeness_change = graph.closeness_centrality_change.reshape(3, 3)
    closeness_second_change = graph.closeness_centrality_second_change.reshape(3, 3)
    assert torch.allclose(
        closeness_change,
        torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.0 / 6.0, 0.5, 2.0 / 3.0],
        ]),
    )
    assert torch.allclose(
        closeness_second_change,
        torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0 / 3.0, 0.0, 2.0 / 3.0],
        ]),
    )
    assert graph.centrality_change_valid_mask.reshape(3, 3).tolist() == [
        [False, False, False],
        [True, True, True],
        [True, True, True],
    ]
    assert graph.centrality_second_change_valid_mask.reshape(3, 3).tolist() == [
        [False, False, False],
        [False, False, False],
        [True, True, True],
    ]


def test_same_lane_front_vehicle_safety_features_are_edge_aligned():
    builder = make_builder(distance_margin=25.0)
    position = torch.tensor([[[0.0, 0.0]], [[20.0, 0.0]]])
    velocity = torch.tensor([[[10.0, 0.0]], [[5.0, 0.0]]])
    lane_position = torch.tensor([[10.0, 0.0]])
    lane_heading = torch.zeros(1)
    lane_batch = torch.zeros(1, dtype=torch.long)
    conflict = torch.empty((2, 0), dtype=torch.long)

    graph = builder(
        position,
        velocity,
        torch.ones((2, 1), dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
        lane_heading=lane_heading,
    )

    assert graph.E_VV_msg.tolist() == [[0, 1], [1, 0]]
    assert graph.safety_feature_valid_mask.tolist() == [False, True]
    assert torch.allclose(graph.THW_ij, torch.tensor([0.0, 2.0]), atol=1e-5)
    assert torch.allclose(graph.TTC_ij, torch.tensor([0.0, 4.0]), atol=1e-5)
    assert torch.allclose(
        graph.same_lane_arrival_time_difference,
        torch.tensor([1.0, 1.0]),
        atol=1e-5,
    )
    assert VehicleInteractionGraphBuilder.EDGE_ATTR_NAMES[1] == 'interaction_ttc'
    assert graph.edge_attr[1, 1] != graph.TTC_ij[1]


def test_adjacent_lane_front_vehicle_is_valid_for_safety_features():
    builder = make_builder(distance_margin=25.0)
    position = torch.tensor([[[0.0, 0.0]], [[20.0, 3.0]]])
    velocity = torch.tensor([[[10.0, 0.0]], [[5.0, 0.0]]])
    lane_position = torch.tensor([[0.0, 0.0], [0.0, 3.0]])
    lane_heading = torch.zeros(2)
    lane_batch = torch.zeros(2, dtype=torch.long)
    reference_lane = torch.tensor([[0], [1]], dtype=torch.long)
    left_neighbor_edge_index = torch.tensor([[1], [0]], dtype=torch.long)

    graph = builder(
        position,
        velocity,
        torch.ones((2, 1), dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        lane_position,
        lane_batch,
        torch.empty((2, 0), dtype=torch.long),
        lane_heading=lane_heading,
        reference_lane=reference_lane,
        left_neighbor_edge_index=left_neighbor_edge_index,
    )

    valid_edge = torch.nonzero(graph.safety_feature_valid_mask).flatten()
    assert valid_edge.numel() == 1
    assert graph.E_VV_msg[:, valid_edge].flatten().tolist() == [1, 0]
    assert torch.allclose(graph.THW_ij[valid_edge], torch.tensor([2.0]), atol=1e-5)
    assert torch.allclose(graph.TTC_ij[valid_edge], torch.tensor([4.0]), atol=1e-5)


def test_lane_speed_sets_a_dynamic_ttc_threshold():
    builder = make_builder(
        distance_margin=0.0,
        ttc_threshold=1.0,
        collision_distance=2.0,
    )
    position = torch.tensor([[[0.0, 0.0]], [[20.0, 0.0]]])
    velocity = torch.tensor([[[4.0, 0.0]], [[-4.0, 0.0]]])
    lane_position = torch.tensor([[0.0, 0.0], [20.0, 0.0]])
    lane_batch = torch.zeros(2, dtype=torch.long)
    conflict = torch.empty((2, 0), dtype=torch.long)

    graph = builder(
        position,
        velocity,
        torch.ones((2, 1), dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
    )

    assert graph.E_VV_msg.size(1) == 2
    assert torch.all(graph.edge_attr[:, 2] == 0)
    assert torch.all(graph.edge_attr[:, 3] == 1)


def test_dynamic_ttc_uses_vehicle_count_weighted_speed_across_both_lanes():
    builder = make_builder(
        distance_margin=0.0,
        observation_window=0.0,
        ttc_threshold=1.0,
        collision_distance=2.0,
    )
    position = torch.tensor([
        [[0.0, 0.0]],
        [[0.1, 0.0]],
        [[-0.1, 0.0]],
        [[20.0, 0.0]],
    ])
    velocity = torch.tensor([
        [[6.0, 0.0]],
        [[12.0, 0.0]],
        [[12.0, 0.0]],
        [[0.0, 0.0]],
    ])
    lane_position = torch.tensor([[0.0, 0.0], [20.0, 0.0]])
    lane_batch = torch.zeros(2, dtype=torch.long)
    conflict = torch.empty((2, 0), dtype=torch.long)

    graph = builder(
        position,
        velocity,
        torch.ones((4, 1), dtype=torch.bool),
        torch.zeros(4, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
    )

    pair_key = graph.E_VV_msg[0] * 4 + graph.E_VV_msg[1]
    assert not torch.isin(torch.tensor(3), pair_key)
    assert not torch.isin(torch.tensor(12), pair_key)


def test_similar_arrival_times_at_the_same_lane_node_create_edges():
    builder = make_builder(distance_margin=2.0, lane_arrival_time_threshold=0.1)
    position = torch.tensor([[[0.0, 0.0]], [[10.0, 0.0]]])
    velocity = torch.tensor([[[5.0, 0.0]], [[5.0, 0.0]]])
    lane_position = torch.tensor([[5.0, 0.0]])
    lane_batch = torch.zeros(1, dtype=torch.long)
    conflict = torch.empty((2, 0), dtype=torch.long)

    graph = builder(
        position,
        velocity,
        torch.ones((2, 1), dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
    )

    assert graph.E_VV_msg.size(1) == 2
    assert torch.all(graph.edge_attr[:, 2:5] == 0)
    assert torch.all(graph.edge_attr[:, 5] == 1)


def test_conflicting_lanes_create_shared_conflict_edges():
    builder = make_builder()
    position = torch.tensor([[[0.0, 0.0]], [[20.0, 0.0]]])
    velocity = torch.zeros_like(position)
    lane_position = torch.tensor([[0.0, 0.0], [20.0, 0.0]])
    lane_batch = torch.zeros(2, dtype=torch.long)
    conflict = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    graph = builder(
        position,
        velocity,
        torch.ones((2, 1), dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        lane_position,
        lane_batch,
        conflict,
    )

    assert graph.assigned_lane.tolist() == [[0], [1]]
    assert graph.E_VV_msg.size(1) == 2
    assert torch.all(graph.edge_attr[:, 2:4] == 0)
    assert torch.all(graph.edge_attr[:, 4] == 1)


def test_edges_never_cross_batch_samples_and_expand_per_mode():
    builder = make_builder(distance_margin=5.0)
    position = torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]])
    velocity = torch.zeros_like(position)
    lane_position, lane_batch, conflict = empty_map()
    graph = builder(
        position,
        velocity,
        torch.ones((3, 1), dtype=torch.bool),
        torch.tensor([0, 0, 1]),
        lane_position,
        lane_batch,
        conflict,
    )

    assert graph.E_VV_msg.tolist() == [[0, 1], [1, 0]]
    expanded_edge_index, expanded_edge_attr = builder.expand_across_modes(
        graph,
        num_agents=3,
        num_modes=2,
    )
    assert expanded_edge_index.tolist() == [[0, 3, 1, 4], [1, 4, 0, 3]]
    assert expanded_edge_attr.size(0) == 4
