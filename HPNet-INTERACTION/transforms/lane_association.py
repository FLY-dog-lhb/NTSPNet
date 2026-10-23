import torch
from torch_geometric.transforms import BaseTransform

from modules.lane_reachability import LANE_ASSOCIATION_AUXILIARY_EDGE_TYPE
from modules.lane_reachability import LANE_ASSOCIATION_HEADING_DIFFERENCE
from modules.lane_reachability import LANE_ASSOCIATION_LATERAL_DISTANCE
from modules.lane_reachability import LANE_ASSOCIATION_MAIN_LANE_LOCAL
from modules.lane_reachability import LANE_ASSOCIATION_PROJECTION_S
from modules.lane_reachability import LaneReachabilityBuilder
from utils import compute_historical_kinematics


class LaneAssociationTransform(BaseTransform):
    """Compute lane association on each CPU sample after data augmentation."""

    def __init__(self,
                 num_historical_steps: int = 10,
                 observation_step_time: float = 0.1,
                 centerline_side_index: int = 1,
                 default_lateral_distance: float = 2.5) -> None:
        super(LaneAssociationTransform, self).__init__()
        if num_historical_steps <= 0:
            raise ValueError('num_historical_steps must be positive')
        if observation_step_time <= 0:
            raise ValueError('observation_step_time must be positive')
        self.num_historical_steps = num_historical_steps
        self.observation_step_time = observation_step_time
        self.builder = LaneReachabilityBuilder(
            centerline_side_index=centerline_side_index,
            default_lateral_distance=default_lateral_distance,
        )

    @staticmethod
    def _lane_edge_index(data, edge_name: str) -> torch.Tensor:
        lane_edge_store = data['lane', 'lane']
        if edge_name in lane_edge_store:
            return lane_edge_store[edge_name]
        return torch.empty((2, 0), dtype=torch.long)

    def __call__(self, data):
        historical_position = data['agent']['position'][
            :, :self.num_historical_steps
        ]
        historical_visible_mask = data['agent']['visible_mask'][
            :, :self.num_historical_steps
        ]
        kinematics = compute_historical_kinematics(
            position=historical_position,
            visible_mask=historical_visible_mask,
            step_time=self.observation_step_time,
        )

        num_agents = historical_position.size(0)
        num_lanes = data['lane'].num_nodes
        device = historical_position.device
        if 'visible_mask' in data['lane']:
            lane_visible_mask = data['lane']['visible_mask']
        else:
            lane_visible_mask = torch.ones(
                num_lanes, dtype=torch.bool, device=device,
            )
            data['lane']['visible_mask'] = lane_visible_mask

        association = self.builder.associate(
            agent_position=historical_position,
            agent_heading=kinematics.heading,
            agent_speed=kinematics.speed,
            agent_heading_valid_mask=kinematics.heading_valid_mask,
            agent_visible_mask=historical_visible_mask,
            agent_batch=torch.zeros(num_agents, dtype=torch.long, device=device),
            lane_heading=data['lane']['heading'],
            lane_width=data['lane']['width'],
            lane_drivable=data['lane']['drivable'],
            lane_visible_mask=lane_visible_mask,
            lane_batch=torch.zeros(num_lanes, dtype=torch.long, device=device),
            lane_centerline_spacing=data['lane']['centerline_spacing'],
            predecessor_edge_index=self._lane_edge_index(
                data, 'predecessor_edge_index',
            ),
            successor_edge_index=self._lane_edge_index(
                data, 'successor_edge_index',
            ),
            left_change_edge_index=self._lane_edge_index(
                data, 'left_neighbor_edge_index',
            ),
            right_change_edge_index=self._lane_edge_index(
                data, 'right_neighbor_edge_index',
            ),
            adjacent_left_edge_index=self._lane_edge_index(
                data, 'adjacent_left_edge_index',
            ),
            adjacent_right_edge_index=self._lane_edge_index(
                data, 'adjacent_right_edge_index',
            ),
            polyline_position=data['polyline']['position'],
            polyline_heading=data['polyline']['heading'],
            polyline_length=data['polyline']['length'],
            polyline_side=data['polyline']['side'],
            polyline_to_lane_edge_index=data[
                'polyline', 'lane'
            ]['polyline_to_lane_edge_index'],
        )

        agent_store = data['agent']
        agent_store[LANE_ASSOCIATION_MAIN_LANE_LOCAL] = association.main_lane
        agent_store[LANE_ASSOCIATION_PROJECTION_S] = association.projection_s
        agent_store[LANE_ASSOCIATION_LATERAL_DISTANCE] = (
            association.lateral_distance
        )
        agent_store[LANE_ASSOCIATION_HEADING_DIFFERENCE] = (
            association.heading_difference
        )

        auxiliary_source = association.auxiliary_seed_edge_index[0]
        if num_agents > 0:
            auxiliary_step = torch.div(
                auxiliary_source,
                num_agents,
                rounding_mode='floor',
            )
            auxiliary_agent = auxiliary_source.remainder(num_agents)
        else:
            auxiliary_step = torch.empty(
                0, dtype=torch.long, device=device,
            )
            auxiliary_agent = torch.empty(
                0, dtype=torch.long, device=device,
            )
        auxiliary_store = data[LANE_ASSOCIATION_AUXILIARY_EDGE_TYPE]
        auxiliary_store['edge_index'] = torch.stack([
            auxiliary_agent,
            association.auxiliary_seed_edge_index[1],
        ], dim=0)
        auxiliary_store['historical_step'] = auxiliary_step
        return data
