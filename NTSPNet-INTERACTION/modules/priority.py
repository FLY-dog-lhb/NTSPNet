from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn

from layers import TwoLayerMLP
from modules.lane_reachability import ConflictInstances


class PriorityOutput(NamedTuple):
    instance_logits: torch.Tensor
    instance_probability: torch.Tensor
    pair_probability: torch.Tensor
    directed_pair_probability: torch.Tensor
    directed_head_bias: torch.Tensor
    pseudo_label: torch.Tensor
    pseudo_label_valid_mask: torch.Tensor
    order_target: torch.Tensor
    future_entry_time_i: torch.Tensor
    future_entry_time_j: torch.Tensor


class PriorityModule(nn.Module):
    """Instance Priority -> pair aggregation -> per-head attention bias."""

    CLASS_NAMES = ('pass', 'yield', 'simultaneous')
    CONFLICT_TYPE_NAMES = ('crossing', 'merge', 'split', 'lane_change')
    RULE_FEATURE_NAMES = (
        'r_i_priority',
        'r_j_priority',
        'r_i_yield',
        'r_j_yield',
        'r_i_controlled',
        'r_j_controlled',
    )

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 softmin_temperature: float = 0.2,
                 simultaneous_threshold: float = 0.5,
                 order_scale: float = 2.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.softmin_temperature = softmin_temperature
        self.simultaneous_threshold = simultaneous_threshold
        self.order_scale = order_scale

        # z_i,z_j; v_i,v_j; a_i,a_j; d_i,d_j; tau_raw_i,tau_raw_j;
        # tau_clip_i,tau_clip_j; six rule fields; four conflict types.
        input_dim = 2 * hidden_dim + 20
        self.instance_classifier = TwoLayerMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=3,
        )
        self.head_bias_projection = nn.Linear(3, num_heads)

    @staticmethod
    def _lane_polygon(
            lane_idx: int,
            polyline_position: torch.Tensor,
            polyline_heading: torch.Tensor,
            polyline_length: torch.Tensor,
            polyline_side: torch.Tensor,
            polyline_to_lane_edge_index: torch.Tensor) -> torch.Tensor:
        source, target = polyline_to_lane_edge_index

        def boundary(side_index: int) -> torch.Tensor:
            boundary_source = source[
                (target == lane_idx)
                & (polyline_side[source].long() == side_index)
            ].sort().values
            if boundary_source.numel() == 0:
                return polyline_position.new_zeros((0, 2))
            position = polyline_position[boundary_source]
            heading = polyline_heading[boundary_source]
            length = polyline_length[boundary_source]
            direction = torch.stack([
                torch.cos(heading),
                torch.sin(heading),
            ], dim=-1)
            start = position - 0.5 * length.unsqueeze(-1) * direction
            end = position + 0.5 * length.unsqueeze(-1) * direction
            return torch.cat([start[:1], end], dim=0)

        left_boundary = boundary(0)
        right_boundary = boundary(2)
        if left_boundary.size(0) < 2 or right_boundary.size(0) < 2:
            return polyline_position.new_zeros((0, 2))
        direct_alignment = torch.linalg.vector_norm(
            left_boundary[0] - right_boundary[0]
        ) + torch.linalg.vector_norm(left_boundary[-1] - right_boundary[-1])
        reverse_alignment = torch.linalg.vector_norm(
            left_boundary[0] - right_boundary[-1]
        ) + torch.linalg.vector_norm(left_boundary[-1] - right_boundary[0])
        if reverse_alignment < direct_alignment:
            right_boundary = right_boundary.flip(0)
        return torch.cat([left_boundary, right_boundary.flip(0)], dim=0)

    @staticmethod
    def _point_in_polygon(point: torch.Tensor, polygon: torch.Tensor) -> bool:
        if polygon.size(0) < 3:
            return False
        x_value = float(point[0].item())
        y_value = float(point[1].item())
        inside = False
        polygon_cpu = polygon.detach().cpu()
        previous_idx = polygon_cpu.size(0) - 1
        for current_idx in range(polygon_cpu.size(0)):
            x_current, y_current = polygon_cpu[current_idx].tolist()
            x_previous, y_previous = polygon_cpu[previous_idx].tolist()
            crosses = (
                (y_current > y_value) != (y_previous > y_value)
                and x_value
                < (x_previous - x_current)
                * (y_value - y_current)
                / (y_previous - y_current + 1e-12)
                + x_current
            )
            if crosses:
                inside = not inside
            previous_idx = current_idx
        return inside

    @staticmethod
    def _segment_intersection_parameter(
            start: torch.Tensor,
            end: torch.Tensor,
            edge_start: torch.Tensor,
            edge_end: torch.Tensor) -> Optional[float]:
        direction = end - start
        edge_direction = edge_end - edge_start
        denominator = (
            direction[0] * edge_direction[1]
            - direction[1] * edge_direction[0]
        )
        if abs(float(denominator.item())) < 1e-8:
            return None
        relative = edge_start - start
        path_parameter = (
            relative[0] * edge_direction[1]
            - relative[1] * edge_direction[0]
        ) / denominator
        edge_parameter = (
            relative[0] * direction[1]
            - relative[1] * direction[0]
        ) / denominator
        path_value = float(path_parameter.item())
        edge_value = float(edge_parameter.item())
        if 0.0 <= path_value <= 1.0 and 0.0 <= edge_value <= 1.0:
            return path_value
        return None

    def _first_entry_time(
            self,
            trajectory: torch.Tensor,
            visible_mask: torch.Tensor,
            polygon: torch.Tensor,
            step_time: float) -> Tuple[float, bool]:
        if polygon.size(0) < 3 or trajectory.size(0) == 0 or not visible_mask[0]:
            return 0.0, False
        if self._point_in_polygon(trajectory[0], polygon):
            return 0.0, True
        for step_idx in range(1, trajectory.size(0)):
            if not visible_mask[step_idx - 1] or not visible_mask[step_idx]:
                continue
            start = trajectory[step_idx - 1]
            end = trajectory[step_idx]
            intersection_parameters: List[float] = []
            for edge_idx in range(polygon.size(0)):
                parameter = self._segment_intersection_parameter(
                    start,
                    end,
                    polygon[edge_idx],
                    polygon[(edge_idx + 1) % polygon.size(0)],
                )
                if parameter is not None:
                    intersection_parameters.append(parameter)
            if intersection_parameters:
                first_parameter = min(intersection_parameters)
                return ((step_idx - 1) + first_parameter) * step_time, True
            if self._point_in_polygon(end, polygon):
                return step_idx * step_time, True
        return 0.0, False

    def _pseudo_labels(
            self,
            conflict_instances: ConflictInstances,
            historical_position: torch.Tensor,
            future_position: torch.Tensor,
            historical_visible_mask: torch.Tensor,
            future_visible_mask: torch.Tensor,
            polyline_position: torch.Tensor,
            polyline_heading: torch.Tensor,
            polyline_length: torch.Tensor,
            polyline_side: torch.Tensor,
            polyline_to_lane_edge_index: torch.Tensor,
            step_time: float) -> Tuple[torch.Tensor, ...]:
        num_instances = conflict_instances.pair_index.size(1)
        output_device = historical_position.device
        output_dtype = historical_position.dtype
        label = torch.full(
            (num_instances,), -1, dtype=torch.long, device=output_device,
        )
        valid_mask = torch.zeros(
            num_instances, dtype=torch.bool, device=output_device,
        )
        order_target = torch.zeros(
            num_instances, dtype=output_dtype, device=output_device,
        )
        entry_time_i = torch.zeros_like(order_target)
        entry_time_j = torch.zeros_like(order_target)
        if num_instances == 0:
            return label, valid_mask, order_target, entry_time_i, entry_time_j

        # These are non-differentiable supervision targets.  Running their
        # branch-heavy polygon traversal on CUDA turns every Python boolean,
        # item() and tolist() into a device synchronization.  Copy the compact
        # inputs once and execute the identical geometry on the host instead.
        # The five result tensors are copied back once for the loss interface.
        if output_device.type != 'cpu':
            pair_index = conflict_instances.pair_index.detach().cpu()
            conflict_lane = conflict_instances.conflict_lane.detach().cpu()
            instance_valid_mask = conflict_instances.valid_mask.detach().cpu()
            historical_position = historical_position.detach().cpu()
            future_position = future_position.detach().cpu()
            historical_visible_mask = historical_visible_mask.detach().cpu()
            future_visible_mask = future_visible_mask.detach().cpu()
            polyline_position = polyline_position.detach().cpu()
            polyline_heading = polyline_heading.detach().cpu()
            polyline_length = polyline_length.detach().cpu()
            polyline_side = polyline_side.detach().cpu()
            polyline_to_lane_edge_index = (
                polyline_to_lane_edge_index.detach().cpu()
            )
            label = label.cpu()
            valid_mask = valid_mask.cpu()
            order_target = order_target.cpu()
            entry_time_i = entry_time_i.cpu()
            entry_time_j = entry_time_j.cpu()
        else:
            pair_index = conflict_instances.pair_index
            conflict_lane = conflict_instances.conflict_lane
            instance_valid_mask = conflict_instances.valid_mask

        polygon_cache = {}
        for instance_idx in range(num_instances):
            if not bool(instance_valid_mask[instance_idx]):
                continue
            lane_idx = int(conflict_lane[instance_idx].item())
            if lane_idx not in polygon_cache:
                polygon_cache[lane_idx] = self._lane_polygon(
                    lane_idx,
                    polyline_position,
                    polyline_heading,
                    polyline_length,
                    polyline_side,
                    polyline_to_lane_edge_index,
                )
            polygon = polygon_cache[lane_idx]
            agent_i = int(pair_index[0, instance_idx].item())
            agent_j = int(pair_index[1, instance_idx].item())
            trajectory_i = torch.cat([
                historical_position[agent_i, -1:], future_position[agent_i],
            ], dim=0)
            trajectory_j = torch.cat([
                historical_position[agent_j, -1:], future_position[agent_j],
            ], dim=0)
            visible_i = torch.cat([
                historical_visible_mask[agent_i, -1:], future_visible_mask[agent_i],
            ], dim=0)
            visible_j = torch.cat([
                historical_visible_mask[agent_j, -1:], future_visible_mask[agent_j],
            ], dim=0)
            time_i, valid_i = self._first_entry_time(
                trajectory_i, visible_i, polygon, step_time,
            )
            time_j, valid_j = self._first_entry_time(
                trajectory_j, visible_j, polygon, step_time,
            )
            if not valid_i or not valid_j:
                continue
            entry_time_i[instance_idx] = time_i
            entry_time_j[instance_idx] = time_j
            difference = time_j - time_i
            if difference > self.simultaneous_threshold:
                label[instance_idx] = 0
            elif difference < -self.simultaneous_threshold:
                label[instance_idx] = 1
            else:
                label[instance_idx] = 2
            order_target[instance_idx] = max(
                -1.0, min(1.0, difference / self.order_scale)
            )
            valid_mask[instance_idx] = True
        if output_device.type != 'cpu':
            return tuple(value.to(output_device) for value in (
                label,
                valid_mask,
                order_target,
                entry_time_i,
                entry_time_j,
            ))
        return label, valid_mask, order_target, entry_time_i, entry_time_j

    def forward(
            self,
            conflict_instances: ConflictInstances,
            style_embedding: torch.Tensor,
            longitudinal_speed: torch.Tensor,
            longitudinal_acceleration: torch.Tensor,
            conflict_type_feature: torch.Tensor,
            rule_feature: torch.Tensor,
            historical_position: torch.Tensor,
            future_position: torch.Tensor,
            historical_visible_mask: torch.Tensor,
            future_visible_mask: torch.Tensor,
            polyline_position: torch.Tensor,
            polyline_heading: torch.Tensor,
            polyline_length: torch.Tensor,
            polyline_side: torch.Tensor,
            polyline_to_lane_edge_index: torch.Tensor,
            step_time: float) -> PriorityOutput:
        num_instances = conflict_instances.pair_index.size(1)
        if conflict_type_feature.shape != (num_instances, 4):
            raise ValueError('conflict_type_feature must have shape [num_instances, 4]')
        if rule_feature.shape != (num_instances, 6):
            raise ValueError('rule_feature must have shape [num_instances, 6]')
        if num_instances == 0:
            empty_probability = style_embedding.new_zeros((0, 3))
            empty_bias = style_embedding.new_zeros((0, self.num_heads))
            empty_label = torch.empty(0, dtype=torch.long, device=style_embedding.device)
            empty_valid = torch.empty(0, dtype=torch.bool, device=style_embedding.device)
            empty_scalar = style_embedding.new_zeros(0)
            return PriorityOutput(
                empty_probability,
                empty_probability,
                empty_probability,
                empty_probability,
                empty_bias,
                empty_label,
                empty_valid,
                empty_scalar,
                empty_scalar,
                empty_scalar,
            )

        agent_i, agent_j = conflict_instances.pair_index
        instance_input = torch.cat([
            style_embedding[agent_i],
            style_embedding[agent_j],
            longitudinal_speed[agent_i].unsqueeze(-1),
            longitudinal_speed[agent_j].unsqueeze(-1),
            longitudinal_acceleration[agent_i].unsqueeze(-1),
            longitudinal_acceleration[agent_j].unsqueeze(-1),
            conflict_instances.distance_i.unsqueeze(-1),
            conflict_instances.distance_j.unsqueeze(-1),
            conflict_instances.tau_raw_i.unsqueeze(-1),
            conflict_instances.tau_raw_j.unsqueeze(-1),
            conflict_instances.tau_clip_i.unsqueeze(-1),
            conflict_instances.tau_clip_j.unsqueeze(-1),
            rule_feature,
            conflict_type_feature,
        ], dim=-1)
        instance_logits = self.instance_classifier(instance_input)
        instance_probability = torch.softmax(instance_logits, dim=-1)

        num_pairs = conflict_instances.canonical_pair_index.size(1)
        pair_probability = style_embedding.new_zeros((num_pairs, 3))
        for pair_idx in range(num_pairs):
            instance_mask = (
                conflict_instances.valid_mask
                & (conflict_instances.pair_group == pair_idx)
            )
            instance_index = torch.nonzero(instance_mask, as_tuple=False).flatten()
            if instance_index.numel() == 0:
                continue
            weight = torch.softmax(
                -conflict_instances.urgency_cost[instance_index]
                / self.softmin_temperature,
                dim=0,
            )
            pair_probability[pair_idx] = (
                weight.unsqueeze(-1) * instance_probability[instance_index]
            ).sum(dim=0)

        reverse_probability = pair_probability[:, [1, 0, 2]]
        directed_pair_probability = torch.cat([
            pair_probability, reverse_probability,
        ], dim=0)
        directed_head_bias = self.head_bias_projection(
            directed_pair_probability,
        )

        (
            pseudo_label,
            pseudo_label_valid_mask,
            order_target,
            future_entry_time_i,
            future_entry_time_j,
        ) = self._pseudo_labels(
            conflict_instances=conflict_instances,
            historical_position=historical_position,
            future_position=future_position,
            historical_visible_mask=historical_visible_mask,
            future_visible_mask=future_visible_mask,
            polyline_position=polyline_position,
            polyline_heading=polyline_heading,
            polyline_length=polyline_length,
            polyline_side=polyline_side,
            polyline_to_lane_edge_index=polyline_to_lane_edge_index,
            step_time=step_time,
        )
        return PriorityOutput(
            instance_logits,
            instance_probability,
            pair_probability,
            directed_pair_probability,
            directed_head_bias,
            pseudo_label,
            pseudo_label_valid_mask,
            order_target,
            future_entry_time_i,
            future_entry_time_j,
        )
