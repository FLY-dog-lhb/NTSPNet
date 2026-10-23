import torch
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from modules.lane_reachability import load_lane_association_from_data
from modules.lane_reachability import LaneReachabilityBuilder
from transforms import LaneAssociationTransform
from utils import compute_historical_kinematics


LANE_EDGE_NAMES = (
    'predecessor_edge_index',
    'successor_edge_index',
    'left_neighbor_edge_index',
    'right_neighbor_edge_index',
    'adjacent_left_edge_index',
    'adjacent_right_edge_index',
)


def empty_edge():
    return torch.empty((2, 0), dtype=torch.long)


def make_sample(lane_y, agent_y, lane_visible=None, agent_visible=None):
    data = HeteroData()
    num_lanes = len(lane_y)
    num_agents = len(agent_y)
    num_steps = 2

    position = torch.zeros((num_agents, num_steps, 2))
    position[:, 0, 0] = 2.0
    position[:, 1, 0] = 4.0
    position[:, :, 1] = torch.tensor(agent_y).unsqueeze(-1)
    data['agent']['position'] = position
    if agent_visible is None:
        agent_visible = torch.ones((num_agents, num_steps), dtype=torch.bool)
    data['agent']['visible_mask'] = agent_visible
    data['agent'].num_nodes = num_agents

    data['lane'].num_nodes = num_lanes
    data['lane']['position'] = torch.stack([
        torch.full((num_lanes,), 5.0),
        torch.tensor(lane_y),
    ], dim=-1)
    data['lane']['heading'] = torch.zeros(num_lanes)
    data['lane']['width'] = torch.full((num_lanes,), 4.0)
    data['lane']['drivable'] = torch.ones(num_lanes, dtype=torch.bool)
    data['lane']['centerline_spacing'] = torch.full((num_lanes,), 2.0)
    if lane_visible is None:
        lane_visible = torch.ones(num_lanes, dtype=torch.bool)
    data['lane']['visible_mask'] = lane_visible
    for edge_name in LANE_EDGE_NAMES:
        data['lane', 'lane'][edge_name] = empty_edge()

    polyline_position = []
    polyline_lane = []
    for lane_idx, y_value in enumerate(lane_y):
        for x_value in (1.0, 3.0, 5.0, 7.0, 9.0):
            polyline_position.append((x_value, y_value))
            polyline_lane.append(lane_idx)
    num_polylines = len(polyline_position)
    data['polyline']['position'] = torch.tensor(polyline_position)
    data['polyline']['heading'] = torch.zeros(num_polylines)
    data['polyline']['length'] = torch.full((num_polylines,), 2.0)
    data['polyline']['side'] = torch.ones(num_polylines, dtype=torch.uint8)
    data['polyline'].num_nodes = num_polylines
    data['polyline', 'lane']['polyline_to_lane_edge_index'] = torch.stack([
        torch.arange(num_polylines),
        torch.tensor(polyline_lane),
    ])
    return data


def online_association(data):
    kinematics = compute_historical_kinematics(
        data['agent']['position'],
        data['agent']['visible_mask'],
        0.1,
    )
    edge_store = data['lane', 'lane']
    return LaneReachabilityBuilder().associate(
        agent_position=data['agent']['position'],
        agent_heading=kinematics.heading,
        agent_speed=kinematics.speed,
        agent_heading_valid_mask=kinematics.heading_valid_mask,
        agent_visible_mask=data['agent']['visible_mask'],
        agent_batch=data['agent']['batch'],
        lane_heading=data['lane']['heading'],
        lane_width=data['lane']['width'],
        lane_drivable=data['lane']['drivable'],
        lane_visible_mask=data['lane']['visible_mask'],
        lane_batch=data['lane']['batch'],
        lane_centerline_spacing=data['lane']['centerline_spacing'],
        predecessor_edge_index=edge_store['predecessor_edge_index'],
        successor_edge_index=edge_store['successor_edge_index'],
        left_change_edge_index=edge_store['left_neighbor_edge_index'],
        right_change_edge_index=edge_store['right_neighbor_edge_index'],
        adjacent_left_edge_index=edge_store['adjacent_left_edge_index'],
        adjacent_right_edge_index=edge_store['adjacent_right_edge_index'],
        polyline_position=data['polyline']['position'],
        polyline_heading=data['polyline']['heading'],
        polyline_length=data['polyline']['length'],
        polyline_side=data['polyline']['side'],
        polyline_to_lane_edge_index=data[
            'polyline', 'lane'
        ]['polyline_to_lane_edge_index'],
    )


def test_transform_uses_post_occlusion_lane_visibility():
    data = make_sample(
        lane_y=[0.0, 2.0],
        agent_y=[0.1],
        lane_visible=torch.tensor([False, True]),
    )
    transformed = LaneAssociationTransform(2, 0.1)(data)
    association = load_lane_association_from_data(transformed, 2)
    assert association.main_lane.tolist() == [[1, 1]]


def test_worker_association_matches_batched_online_path_and_offsets():
    transform = LaneAssociationTransform(2, 0.1)
    first = transform(make_sample([0.0, 2.0], [0.9]))
    second = transform(make_sample(
        [10.0, 12.0, 14.0],
        [12.9, 13.0],
        agent_visible=torch.tensor([[True, True], [False, False]]),
    ))
    batch = Batch.from_data_list([first, second])

    worker = load_lane_association_from_data(batch, 2)
    online = online_association(batch)
    assert torch.equal(worker.main_lane, online.main_lane)
    assert torch.equal(worker.main_lane[-1], torch.tensor([-1, -1]))
    assert torch.allclose(worker.projection_s, online.projection_s)
    assert torch.allclose(worker.lateral_distance, online.lateral_distance)
    assert torch.allclose(worker.heading_difference, online.heading_difference)
    assert torch.equal(
        worker.auxiliary_seed_edge_index,
        online.auxiliary_seed_edge_index,
    )

    valid = worker.main_lane >= 0
    agent_batch = batch['agent']['batch'].unsqueeze(-1).expand_as(valid)
    assert torch.equal(
        batch['lane']['batch'][worker.main_lane[valid]],
        agent_batch[valid],
    )

    second_load = load_lane_association_from_data(batch, 2)
    assert torch.equal(second_load.main_lane, worker.main_lane)
    assert torch.equal(
        second_load.auxiliary_seed_edge_index,
        worker.auxiliary_seed_edge_index,
    )


def test_missing_worker_fields_use_compatible_fallback_signal():
    data = make_sample([0.0], [0.0])
    assert load_lane_association_from_data(data, 2) is None
