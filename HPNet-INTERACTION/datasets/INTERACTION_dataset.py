import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from itertools import permutations
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

import lanelet2
from lanelet2.projection import UtmProjector

from modules.lane_topology import build_potential_conflict_topology
from utils import compute_angles_lengths_2D
from utils import transform_point_to_local_coordinate
from utils import get_index_of_A_in_B
from utils import update_agent_kinematics
from utils import wrap_angle


class INTERACTIONDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 num_historical_steps: int = 10,
                 num_future_steps:int = 30,
                 observation_step_time: float = 0.1,
                 motion_source: str = 'causal_difference',
                 csv_motion_cache_dir: Optional[str] = None) -> None:
        self.root = root

        if motion_source not in ('causal_difference', 'csv'):
            raise ValueError(
                "motion_source must be 'causal_difference' or 'csv'"
            )
        self.motion_source = motion_source
        self.csv_motion_cache_dir = (
            csv_motion_cache_dir
            if csv_motion_cache_dir is not None
            else os.path.join(root, 'csv_motion_v1')
        )

        if split == 'train':
            self.split = 'train'
            self._directory = 'train'
        elif split == 'val':
            self.split = 'val'
            self._directory = 'val'
        elif split == 'test':
            self.split = 'test'
            self._directory = 'test_multi-agent'
        else:
            raise ValueError(split + ' is not valid')
        
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = []
        for raw_path in tqdm(self.raw_paths):
            raw_dir, raw_file_name = os.path.split(raw_path)
            scenario_name = os.path.splitext(raw_file_name)[0]
            scenario_name = '_'.join(scenario_name.split('_')[:-1])
            df = pd.read_csv(raw_path)
            for case_id in tqdm(df['case_id'].unique()):
                self._processed_file_names.append(scenario_name + '_' + str(int(case_id)) + '.pt')
        self._processed_paths = [os.path.join(self.processed_dir, name) for name in self.processed_file_names]
        
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.observation_step_time = observation_step_time

        self.projector = UtmProjector(lanelet2.io.Origin(0, 0))
        self.traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                           lanelet2.traffic_rules.Participants.Vehicle)

        self._agent_type = ['car', 'pedestrian/bicycle']
        self._polyline_side = ['left','center','right']
        super(INTERACTIONDataset, self).__init__(root=root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory+'_processed')
    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        for raw_path in tqdm(self.raw_paths):
            #map
            raw_dir, raw_file_name = os.path.split(raw_path)
            scenario_name = os.path.splitext(raw_file_name)[0]
            scenario_name = '_'.join(scenario_name.split('_')[:-1])
            base_dir = os.path.dirname(raw_dir)
            map_path = os.path.join(base_dir,'maps',scenario_name+'.osm')
            map_api = lanelet2.io.load(map_path, self.projector)
            routing_graph = lanelet2.routing.RoutingGraph(map_api, self.traffic_rules)
            
            #agent
            df = pd.read_csv(raw_path)
            for case_id, case_df in tqdm(df.groupby('case_id')):
                data = dict()      
                data['scenario_name'] = scenario_name
                data['case_id'] = int(case_id)
                data.update(self.get_features(case_df, map_api, routing_graph))
                torch.save(data, os.path.join(self.processed_dir, scenario_name + '_' + str(int(case_id)) + '.pt'))

    @staticmethod
    def compute_average_lane_width(centerline: torch.Tensor,
                                   left_boundary: torch.Tensor,
                                   right_boundary: torch.Tensor) -> torch.Tensor:
        """Average the nearest left/right boundary distances along the centerline."""
        if centerline.size(0) == 0 or left_boundary.size(0) == 0 or right_boundary.size(0) == 0:
            raise ValueError('lane centerline and both boundaries must be non-empty')

        def point_to_polyline_distance(points: torch.Tensor,
                                       polyline: torch.Tensor) -> torch.Tensor:
            if polyline.size(0) == 1:
                return torch.linalg.vector_norm(points - polyline[0], dim=-1)
            segment_start = polyline[:-1]
            segment_vector = polyline[1:] - polyline[:-1]
            relative = points.unsqueeze(1) - segment_start.unsqueeze(0)
            segment_length_sq = (segment_vector * segment_vector).sum(dim=-1).clamp_min(1e-12)
            projection = (
                (relative * segment_vector.unsqueeze(0)).sum(dim=-1)
                / segment_length_sq.unsqueeze(0)
            ).clamp(0.0, 1.0)
            closest_point = (
                segment_start.unsqueeze(0)
                + projection.unsqueeze(-1) * segment_vector.unsqueeze(0)
            )
            return torch.linalg.vector_norm(
                points.unsqueeze(1) - closest_point,
                dim=-1,
            ).min(dim=1).values

        left_distance = point_to_polyline_distance(centerline[:, :2], left_boundary[:, :2])
        right_distance = point_to_polyline_distance(centerline[:, :2], right_boundary[:, :2])
        return (left_distance + right_distance).mean()

    @staticmethod
    def resample_centerline(centerline: torch.Tensor,
                            spacing: float = 2.0) -> torch.Tensor:
        """Resample a lanelet centerline at a fixed arc-length spacing."""
        if centerline.ndim != 2 or centerline.size(-1) < 2:
            raise ValueError('centerline must have shape [num_points, >=2]')
        if centerline.size(0) < 2:
            raise ValueError('lanelet centerline must contain at least two points')
        if spacing <= 0:
            raise ValueError('centerline spacing must be positive')

        segment = centerline[1:, :2] - centerline[:-1, :2]
        segment_length = torch.linalg.vector_norm(segment, dim=-1)
        valid_segment = segment_length > 1e-6
        if not valid_segment.any():
            raise ValueError('lanelet centerline must have positive length')

        cumulative = torch.cat([
            centerline.new_zeros(1),
            segment_length.cumsum(dim=0),
        ])
        total_length = cumulative[-1]
        sample_distance = torch.arange(
            0.0,
            float(total_length.item()),
            spacing,
            dtype=centerline.dtype,
            device=centerline.device,
        )
        if sample_distance.numel() == 0 or sample_distance[-1] < total_length - 1e-6:
            sample_distance = torch.cat([sample_distance, total_length.reshape(1)])
        else:
            sample_distance[-1] = total_length

        segment_index = torch.searchsorted(
            cumulative[1:],
            sample_distance,
            right=False,
        ).clamp(max=segment_length.numel() - 1)
        local_distance = sample_distance - cumulative[segment_index]
        ratio = local_distance / segment_length[segment_index].clamp_min(1e-6)
        return (
            centerline[segment_index, :2]
            + ratio.unsqueeze(-1) * segment[segment_index]
        )

    def get_features(self, 
                     df: pd.DataFrame,
                     map_api,
                     routing_graph) -> Dict:
        data = {
            'agent': {},
            'lane': {},
            'polyline': {},
            ('polyline', 'lane'): {},
            ('lane', 'lane'): {}
        }
        #AGENT
        # filter out actors that are unseen during the historical time steps
        timestep_ids = list(np.sort(df['timestamp_ms'].unique()))
        historical_timestamps = timestep_ids[:self.num_historical_steps]
        historical_df = df[df['timestamp_ms'].isin(historical_timestamps)]
        agent_ids = list(historical_df['track_id'].unique())
        num_agents = len(agent_ids)
        df = df[df['track_id'].isin(agent_ids)]

        # initialization
        agent_id = torch.zeros(num_agents, dtype=torch.uint8)
        visible_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_position = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        agent_csv_velocity = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_csv_heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        agent_csv_motion_valid_mask = torch.zeros(
            num_agents, self.num_steps, dtype=torch.bool,
        )
        agent_length = torch.zeros(num_agents, dtype=torch.float)
        agent_width = torch.zeros(num_agents, dtype=torch.float)
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        agent_interset = torch.zeros(num_agents, dtype=torch.uint8)
        
        for track_id, track_df in df.groupby('track_id'):
            agent_idx = agent_ids.index(track_id)
            agent_id[agent_idx] = track_id
            agent_steps = [timestep_ids.index(timestamp) for timestamp in track_df['timestamp_ms']]

            visible_mask[agent_idx, agent_steps] = True

            agent_type_name = track_df['agent_type'].values[0]
            agent_type[agent_idx] = torch.tensor(self._agent_type.index(agent_type_name), dtype=torch.uint8)

            if agent_type_name == 'car':
                agent_length[agent_idx] = track_df['length'].values[0]
                agent_width[agent_idx] = track_df['width'].values[0]

            if 'track_to_predict' in track_df.columns:
                agent_category[agent_idx] = torch.tensor(track_df['track_to_predict'].values[0], dtype=torch.uint8)
                agent_interset[agent_idx] = torch.tensor(track_df['interesting_agent'].values[0], dtype=torch.uint8)
            elif agent_type_name == 'car':
                agent_category[agent_idx] = torch.tensor(1, dtype=torch.uint8)


            agent_position[agent_idx, agent_steps] = torch.from_numpy(np.stack([track_df['x'].values, track_df['y'].values], axis=-1)).float()
            csv_velocity = torch.from_numpy(
                np.stack([track_df['vx'].values, track_df['vy'].values], axis=-1)
            ).float()
            velocity_heading = torch.atan2(
                csv_velocity[:, 1], csv_velocity[:, 0],
            )
            if agent_type_name == 'car':
                csv_heading = torch.from_numpy(
                    track_df['psi_rad'].values,
                ).float()
            else:
                csv_heading = velocity_heading
            agent_csv_velocity[agent_idx, agent_steps] = csv_velocity
            agent_csv_heading[agent_idx, agent_steps] = csv_heading
            agent_csv_motion_valid_mask[agent_idx, agent_steps] = True

        data['agent']['id'] = agent_id
        data['agent']['num_nodes'] = num_agents
        data['agent']['visible_mask'] = visible_mask
        data['agent']['position'] = agent_position
        data['agent']['heading'] = agent_heading
        data['agent']['csv_velocity'] = agent_csv_velocity
        data['agent']['csv_heading'] = agent_csv_heading
        data['agent']['csv_motion_valid_mask'] = agent_csv_motion_valid_mask
        data['agent']['length'] = agent_length
        data['agent']['width'] = agent_width
        data['agent']['type'] = agent_type
        data['agent']['category'] = agent_category
        data['agent']['interest'] = agent_interset
        update_agent_kinematics(
            data['agent'],
            num_historical_steps=self.num_historical_steps,
            step_time=self.observation_step_time,
        )
        
        #MAP
        lane_ids = []
        for lane in map_api.laneletLayer: 
            lane_ids.append(lane.id)
        
        num_lanes = len(lane_ids)
        if num_lanes == 0:
            raise ValueError('INTERACTION map contains no lanelets; vehicle style features require at least one lane')
        lane_id = torch.zeros(num_lanes, dtype=torch.float)
        lane_position = torch.zeros(num_lanes, 2, dtype=torch.float)
        lane_heading = torch.zeros(num_lanes, dtype=torch.float)
        lane_length = torch.zeros(num_lanes, dtype=torch.float)
        lane_curvature = torch.zeros(num_lanes, dtype=torch.float)
        lane_turn_direction = torch.zeros(num_lanes, dtype=torch.uint8)
        lane_has_traffic_sign = torch.zeros(num_lanes, dtype=torch.uint8)
        lane_speed_limit = torch.zeros(num_lanes, dtype=torch.float)
        lane_width = torch.zeros(num_lanes, dtype=torch.float)
        lane_drivable = torch.zeros(num_lanes, dtype=torch.bool)
        lane_start_position = torch.zeros(num_lanes, 2, dtype=torch.float)
        lane_end_position = torch.zeros(num_lanes, 2, dtype=torch.float)
        lane_centerline_spacing = torch.full((num_lanes,), 2.0, dtype=torch.float)

        num_polylines = torch.zeros(num_lanes, dtype=torch.long)
        polyline_position: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_heading: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_length: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_side: List[Optional[torch.Tensor]] = [None] * num_lanes

        lane_left_neighbor_edge_index = []
        lane_right_neighbor_edge_index = []
        lane_predecessor_edge_index = []
        lane_successor_edge_index = []
        lane_conflict_edge_index = []
        lane_adjacent_left_edge_index = []
        lane_adjacent_right_edge_index = []

        for lane in map_api.laneletLayer: 
            lane_idx = lane_ids.index(lane.id)
            lane_id[lane_idx] = lane.id

            points = [np.array([pt.x, pt.y]) for pt in lane.centerline]
            centerline = self.resample_centerline(
                torch.from_numpy(np.array(points)).float(),
                spacing=2.0,
            )

            center_index = int((centerline.size(0)-1)/2)
            lane_position[lane_idx] = centerline[center_index, :2]
            lane_heading[lane_idx] = torch.atan2(centerline[center_index+1, 1] - centerline[center_index, 1], 
                                                 centerline[center_index+1, 0] - centerline[center_index, 0])
            lane_length[lane_idx] = torch.norm(centerline[1:] - centerline[:-1], p=2, dim=-1).sum()
            lane_start_position[lane_idx] = centerline[0, :2]
            lane_end_position[lane_idx] = centerline[-1, :2]

            try:
                lane_drivable[lane_idx] = bool(self.traffic_rules.canPass(lane))
            except Exception:
                lane_drivable[lane_idx] = False

            cl_vectors = centerline[1:] - centerline[:-1]
            if cl_vectors.size(0) >= 2:
                cl_angles = torch.atan2(cl_vectors[1:, 1], cl_vectors[1:, 0]) - torch.atan2(cl_vectors[:-1, 1], cl_vectors[:-1, 0])
                lane_curvature[lane_idx] = wrap_angle(cl_angles).abs().mean()
            start_heading = torch.atan2(cl_vectors[0, 1], cl_vectors[0, 0])
            end_heading = torch.atan2(cl_vectors[-1, 1], cl_vectors[-1, 0])
            heading_change = wrap_angle(end_heading - start_heading)
            if heading_change.abs() > 2.5:
                lane_turn_direction[lane_idx] = 3
            elif heading_change > 0.3:
                lane_turn_direction[lane_idx] = 1
            elif heading_change < -0.3:
                lane_turn_direction[lane_idx] = 2
            else:
                lane_turn_direction[lane_idx] = 0

            try:
                has_row = any(type(re).__name__ == 'RightOfWay' for re in lane.regulatoryElements)
                lane_has_traffic_sign[lane_idx] = 1 if has_row else 0
            except Exception:
                lane_has_traffic_sign[lane_idx] = 0

            try:
                speed_lim_info = self.traffic_rules.speedLimit(lane)
                lane_speed_limit[lane_idx] = float(speed_lim_info.speedLimit) / 130.0
            except Exception:
                lane_speed_limit[lane_idx] = 0.0

            points = [np.array([pt.x, pt.y]) for pt in lane.leftBound]
            left_boundary = torch.from_numpy(np.array(points)).float()
            points = [np.array([pt.x, pt.y]) for pt in lane.rightBound]
            right_boundary = torch.from_numpy(np.array(points)).float()
            lane_width[lane_idx] = self.compute_average_lane_width(
                centerline=centerline,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
            )
            left_vector = left_boundary[1:] - left_boundary[:-1]      
            right_vector = right_boundary[1:] - right_boundary[:-1]
            centerline_vector = centerline[1:] - centerline[:-1]
            polyline_position[lane_idx] = torch.cat([(left_boundary[1:] + left_boundary[:-1])/2, (right_boundary[1:] + right_boundary[:-1])/2, (centerline[1:] + centerline[:-1])/2], dim=0)
            polyline_length[lane_idx], polyline_heading[lane_idx] = compute_angles_lengths_2D(torch.cat([left_vector, right_vector, centerline_vector],dim=0))
            num_left_polyline = len(left_vector)
            num_right_polyline = len(right_vector)
            num_centerline_polyline = len(centerline_vector)
            polyline_side[lane_idx] = torch.cat(
                [torch.full((num_left_polyline,), self._polyline_side.index('left'), dtype=torch.uint8),
                 torch.full((num_right_polyline,), self._polyline_side.index('right'), dtype=torch.uint8),
                 torch.full((num_centerline_polyline,), self._polyline_side.index('center'), dtype=torch.uint8)], dim=0)

            num_polylines[lane_idx] = num_left_polyline + num_right_polyline + num_centerline_polyline

            lane_left_neighbor_lane = routing_graph.left(lane)
            lane_left_neighbor_id = [lane_left_neighbor_lane.id] if lane_left_neighbor_lane else []
            lane_left_neighbor_idx = get_index_of_A_in_B(lane_left_neighbor_id, lane_ids)
            if len(lane_left_neighbor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_left_neighbor_idx, dtype=torch.long), torch.full((len(lane_left_neighbor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_left_neighbor_edge_index.append(edge_index)
            lane_right_neighbor_lane = routing_graph.right(lane)
            lane_right_neighbor_id = [lane_right_neighbor_lane.id] if lane_right_neighbor_lane else []
            lane_right_neighbor_idx = get_index_of_A_in_B(lane_right_neighbor_id, lane_ids)
            if len(lane_right_neighbor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_right_neighbor_idx, dtype=torch.long), torch.full((len(lane_right_neighbor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_right_neighbor_edge_index.append(edge_index)
            lane_adjacent_left = routing_graph.adjacentLeft(lane)
            lane_adjacent_left_id = [lane_adjacent_left.id] if lane_adjacent_left else []
            lane_adjacent_left_idx = get_index_of_A_in_B(lane_adjacent_left_id, lane_ids)
            if len(lane_adjacent_left_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_adjacent_left_idx, dtype=torch.long), torch.full((len(lane_adjacent_left_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_adjacent_left_edge_index.append(edge_index)
            lane_adjacent_right = routing_graph.adjacentRight(lane)
            lane_adjacent_right_id = [lane_adjacent_right.id] if lane_adjacent_right else []
            lane_adjacent_right_idx = get_index_of_A_in_B(lane_adjacent_right_id, lane_ids)
            if len(lane_adjacent_right_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_adjacent_right_idx, dtype=torch.long), torch.full((len(lane_adjacent_right_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_adjacent_right_edge_index.append(edge_index)
            lane_predecessor_lanes = routing_graph.previous(lane)
            lane_predecessor_ids = [ll.id for ll in lane_predecessor_lanes] if lane_predecessor_lanes else []
            lane_predecessor_idx = get_index_of_A_in_B(lane_predecessor_ids, lane_ids)
            if len(lane_predecessor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_predecessor_idx, dtype=torch.long), torch.full((len(lane_predecessor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_predecessor_edge_index.append(edge_index)
            lane_successor_lanes = routing_graph.following(lane)
            lane_successor_ids = [ll.id for ll in lane_successor_lanes] if lane_successor_lanes else []
            lane_successor_idx = get_index_of_A_in_B(lane_successor_ids, lane_ids)
            if len(lane_successor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_successor_idx, dtype=torch.long), torch.full((len(lane_successor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_successor_edge_index.append(edge_index)
            try:
                lane_conflict_lanes = routing_graph.conflicting(lane)
                lane_conflict_ids = [ll.id for ll in lane_conflict_lanes] if lane_conflict_lanes else []
            except Exception:
                lane_conflict_ids = []
            lane_conflict_idx = get_index_of_A_in_B(lane_conflict_ids, lane_ids)
            lane_conflict_idx = [idx for idx in lane_conflict_idx if idx != lane_idx]
            if len(lane_conflict_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_conflict_idx, dtype=torch.long), torch.full((len(lane_conflict_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_conflict_edge_index.append(edge_index)
                lane_conflict_edge_index.append(edge_index.flip(0))

        data['lane']['id'] = lane_id
        data['lane']['num_nodes'] = num_lanes
        data['lane']['position'] = lane_position
        data['lane']['length'] = lane_length
        data['lane']['heading'] = lane_heading
        data['lane']['curvature'] = lane_curvature
        data['lane']['turn_direction'] = lane_turn_direction
        data['lane']['has_traffic_sign'] = lane_has_traffic_sign
        data['lane']['speed_limit'] = lane_speed_limit
        data['lane']['width'] = lane_width
        data['lane']['drivable'] = lane_drivable
        data['lane']['start_position'] = lane_start_position
        data['lane']['end_position'] = lane_end_position
        data['lane']['centerline_spacing'] = lane_centerline_spacing

        data['polyline']['num_nodes'] = num_polylines.sum().item()
        data['polyline']['position'] = torch.cat(polyline_position, dim=0)
        data['polyline']['heading'] = torch.cat(polyline_heading, dim=0)
        data['polyline']['length'] = torch.cat(polyline_length, dim=0)
        data['polyline']['side'] = torch.cat(polyline_side, dim=0)

        polyline_to_lane_edge_index = torch.stack([torch.arange(num_polylines.sum(), dtype=torch.long), torch.arange(num_lanes, dtype=torch.long).repeat_interleave(num_polylines)], dim=0)
        data['polyline', 'lane']['polyline_to_lane_edge_index'] = polyline_to_lane_edge_index

        if len(lane_left_neighbor_edge_index) != 0:
            lane_left_neighbor_edge_index = torch.cat(lane_left_neighbor_edge_index, dim=1)
        else:
            lane_left_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_right_neighbor_edge_index) != 0:
            lane_right_neighbor_edge_index = torch.cat(lane_right_neighbor_edge_index, dim=1)
        else:
            lane_right_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_predecessor_edge_index) != 0:
            lane_predecessor_edge_index = torch.cat(lane_predecessor_edge_index, dim=1)
        else:
            lane_predecessor_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_successor_edge_index) != 0:
            lane_successor_edge_index = torch.cat(lane_successor_edge_index, dim=1)
        else:
            lane_successor_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_conflict_edge_index) != 0:
            lane_conflict_edge_index = torch.cat(lane_conflict_edge_index, dim=1)
            lane_conflict_edge_index = torch.unique(lane_conflict_edge_index, dim=1)
        else:
            lane_conflict_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_adjacent_left_edge_index) != 0:
            lane_adjacent_left_edge_index = torch.cat(lane_adjacent_left_edge_index, dim=1)
        else:
            lane_adjacent_left_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_adjacent_right_edge_index) != 0:
            lane_adjacent_right_edge_index = torch.cat(lane_adjacent_right_edge_index, dim=1)
        else:
            lane_adjacent_right_edge_index = torch.tensor([[], []], dtype=torch.long)
        
        data['lane', 'lane']['left_neighbor_edge_index'] = lane_left_neighbor_edge_index
        data['lane', 'lane']['right_neighbor_edge_index'] = lane_right_neighbor_edge_index
        data['lane', 'lane']['predecessor_edge_index'] = lane_predecessor_edge_index
        data['lane', 'lane']['successor_edge_index'] = lane_successor_edge_index
        data['lane', 'lane']['conflict_edge_index'] = lane_conflict_edge_index
        data['lane', 'lane']['adjacent_left_edge_index'] = lane_adjacent_left_edge_index
        data['lane', 'lane']['adjacent_right_edge_index'] = lane_adjacent_right_edge_index

        conflict_topology = build_potential_conflict_topology(
            lane_heading=lane_heading,
            lane_drivable=lane_drivable,
            predecessor_edge_index=lane_predecessor_edge_index,
            successor_edge_index=lane_successor_edge_index,
            left_change_edge_index=lane_left_neighbor_edge_index,
            right_change_edge_index=lane_right_neighbor_edge_index,
            adjacent_left_edge_index=lane_adjacent_left_edge_index,
            adjacent_right_edge_index=lane_adjacent_right_edge_index,
            crossing_edge_index=lane_conflict_edge_index,
        )
        for field_name, field_value in conflict_topology.items():
            if field_name == 'lane_change_pair_index':
                data['lane', 'lane'][field_name] = field_value
            else:
                data['lane'][field_name] = field_value

        return data

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> HeteroData:
        data = HeteroData(torch.load(self.processed_paths[idx]))
        if self.motion_source != 'csv':
            return data

        agent_store = data['agent']
        required_fields = (
            'csv_velocity', 'csv_heading', 'csv_motion_valid_mask',
        )
        if all(field_name in agent_store for field_name in required_fields):
            return data

        processed_name = self.processed_file_names[idx]
        sidecar_path = os.path.join(
            self.csv_motion_cache_dir,
            self.split,
            processed_name,
        )
        if not os.path.exists(sidecar_path):
            raise FileNotFoundError(
                'CSV motion sidecar is required for motion_source=csv but was '
                f'not found: {sidecar_path}. Run compute_csv_motion_cache.py '
                f'for split={self.split}.'
            )
        sidecar = torch.load(sidecar_path)
        if int(sidecar.get('schema_version', -1)) != 1:
            raise ValueError(
                f'unsupported CSV motion sidecar schema: {sidecar_path}'
            )
        sidecar_agent_id = sidecar['agent_id'].to(torch.long)
        processed_agent_id = agent_store['id'].to(torch.long)
        if not torch.equal(sidecar_agent_id, processed_agent_id):
            raise ValueError(
                'CSV motion sidecar agent order does not match processed data: '
                f'{sidecar_path}'
            )
        for field_name in required_fields:
            value = sidecar[field_name]
            if value.size(0) != agent_store.num_nodes:
                raise ValueError(
                    f'{field_name} has incompatible agent count in {sidecar_path}'
                )
            agent_store[field_name] = value
        expected_vector_shape = agent_store['position'].shape
        expected_scalar_shape = agent_store['position'].shape[:2]
        if agent_store['csv_velocity'].shape != expected_vector_shape:
            raise ValueError(
                'csv_velocity shape does not match processed position in '
                f'{sidecar_path}'
            )
        for field_name in ('csv_heading', 'csv_motion_valid_mask'):
            if agent_store[field_name].shape != expected_scalar_shape:
                raise ValueError(
                    f'{field_name} shape does not match processed position in '
                    f'{sidecar_path}'
                )
        return data
