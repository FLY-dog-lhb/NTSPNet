import math
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle

class HorizontalFlip(BaseTransform):
    def __init__(self,
                 flip_p=0.5):
        super(HorizontalFlip, self).__init__()
        self.flip_p=flip_p
    
    def flip_position(self, position):
        position[...,0] = -position[...,0]
        return position

    def flip_heading(self, heading):
        angle = wrap_angle(math.pi - heading)
        return angle

    def flip_vector(self, vector):
        vector[..., 0] = -vector[..., 0]
        return vector

    def __call__(self, data: HeteroData) -> HeteroData:
        if torch.rand(1).item() < self.flip_p:
            data['agent']['position'] = self.flip_position(data['agent']['position'])
            data['agent']['heading'] = self.flip_heading(data['agent']['heading'])
            if 'csv_velocity' in data['agent']:
                data['agent']['csv_velocity'] = self.flip_vector(
                    data['agent']['csv_velocity']
                )
            if 'csv_heading' in data['agent']:
                data['agent']['csv_heading'] = self.flip_heading(
                    data['agent']['csv_heading']
                )
            for vector_name in ('velocity', 'acceleration'):
                if vector_name in data['agent']:
                    data['agent'][vector_name] = self.flip_vector(data['agent'][vector_name])
            if 'teacher_trajectory' in data['agent']:
                data['agent']['teacher_trajectory'] = self.flip_position(
                    data['agent']['teacher_trajectory']
                )
            if 'jerk' in data['agent']:
                data['agent']['jerk'] = self.flip_vector(data['agent']['jerk'])
                data['agent']['acceleration_change'] = data['agent']['jerk']
            elif 'acceleration_change' in data['agent']:
                data['agent']['acceleration_change'] = self.flip_vector(data['agent']['acceleration_change'])
            for delta_name in ('velocity_theta', 'accel_theta', 'heading_change'):
                if delta_name in data['agent']:
                    data['agent'][delta_name] = wrap_angle(-data['agent'][delta_name])
            data['lane']['position'], data['lane']['heading'] = self.flip_position(data['lane']['position']), self.flip_heading(data['lane']['heading'])
            for position_name in ('start_position', 'end_position'):
                if position_name in data['lane']:
                    data['lane'][position_name] = self.flip_position(data['lane'][position_name])
            if 'turn_direction' in data['lane']:
                turn_direction = data['lane']['turn_direction']
                data['lane']['turn_direction'] = torch.where(
                    turn_direction == 1,
                    torch.full_like(turn_direction, 2),
                    torch.where(
                        turn_direction == 2,
                        torch.full_like(turn_direction, 1),
                        turn_direction,
                    ),
                )
            data['polyline']['position'], data['polyline']['heading'] = self.flip_position(data['polyline']['position']), self.flip_heading(data['polyline']['heading'])
            data['polyline']['side'] = 2 - data['polyline']['side']
            data['lane', 'lane']['left_neighbor_edge_index'], data['lane', 'lane']['right_neighbor_edge_index'] = data['lane', 'lane']['right_neighbor_edge_index'], data['lane', 'lane']['left_neighbor_edge_index']
            if (
                'adjacent_left_edge_index' in data['lane', 'lane']
                and 'adjacent_right_edge_index' in data['lane', 'lane']
            ):
                data['lane', 'lane']['adjacent_left_edge_index'], data['lane', 'lane']['adjacent_right_edge_index'] = data['lane', 'lane']['adjacent_right_edge_index'], data['lane', 'lane']['adjacent_left_edge_index']
        return data
