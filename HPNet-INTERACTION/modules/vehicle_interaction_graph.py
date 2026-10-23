from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn


class VehicleInteractionGraph(NamedTuple):
    """A temporal vehicle graph whose node index is ``time * num_agents + agent``.

    ``E_VV_msg`` is the final directed vehicle-conflict edge set produced by
    the union of the distance, TTC, conflicting-lane, and same-lane-arrival
    conditions.

    ``D_VV_msg`` stores one distance per edge. ``A_VV_msg`` uses the message
    passing orientation ``[target, source]``. Consequently,
    ``N_VV_msg[i]`` contains the source nodes that send messages to node ``i``.
    ``degree_centrality`` is normalized incoming degree, while
    ``closeness_centrality`` uses incoming distance-weighted shortest paths.
    Their first- and second-order changes use causal backward differences and
    are zero where visibility is not continuous.

    For an edge ``j -> i``, ``THW_ij`` and ``TTC_ij`` are safety features of
    target vehicle ``i`` relative to front vehicle ``j``. They are valid only
    when both vehicles are assigned to the same lane and ``j`` is ahead along
    that lane's heading.

    ``edge_attr`` columns are, in order: distance in metres, clipped TTC in
    seconds, distance trigger, TTC trigger, shared-conflict trigger, and
    same-lane arrival trigger.
    """

    E_VV_msg: torch.Tensor
    D_VV_msg: torch.Tensor
    A_VV_msg: torch.Tensor
    N_VV_msg: Tuple[torch.Tensor, ...]
    degree_centrality: torch.Tensor
    closeness_centrality: torch.Tensor
    degree_centrality_change: torch.Tensor
    degree_centrality_second_change: torch.Tensor
    closeness_centrality_change: torch.Tensor
    closeness_centrality_second_change: torch.Tensor
    centrality_change_valid_mask: torch.Tensor
    centrality_second_change_valid_mask: torch.Tensor
    THW_ij: torch.Tensor
    TTC_ij: torch.Tensor
    safety_longitudinal_gap: torch.Tensor
    safety_feature_valid_mask: torch.Tensor
    same_lane_arrival_time_difference: torch.Tensor
    edge_attr: torch.Tensor
    assigned_lane: torch.Tensor

    @property
    def edge_index(self) -> torch.Tensor:
        """Compatibility alias for callers using the generic graph name."""
        return self.E_VV_msg


class VehicleInteractionGraphBuilder(nn.Module):
    """Build same-timestep vehicle interactions from historical observations."""

    EDGE_ATTR_NAMES = (
        'distance',
        'interaction_ttc',
        'distance_trigger',
        'ttc_trigger',
        'shared_conflict_node',
        'same_lane_arrival_trigger',
    )

    def __init__(self,
                 distance_margin: float,
                 observation_window: float,
                 ttc_threshold: float,
                 collision_distance: float,
                 max_distance: float,
                 lane_arrival_time_threshold: float = 1.0,
                 thw_max: float = 5.0,
                 safety_ttc_max: float = 10.0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        if distance_margin < 0:
            raise ValueError('distance_margin must be non-negative')
        if observation_window < 0:
            raise ValueError('observation_window must be non-negative')
        if ttc_threshold <= 0:
            raise ValueError('ttc_threshold must be positive')
        if collision_distance < 0:
            raise ValueError('collision_distance must be non-negative')
        if max_distance <= 0:
            raise ValueError('max_distance must be positive')
        if lane_arrival_time_threshold < 0:
            raise ValueError('lane_arrival_time_threshold must be non-negative')
        if thw_max <= 0 or safety_ttc_max <= 0:
            raise ValueError('THW/TTC safety feature maxima must be positive')

        self.distance_margin = distance_margin
        self.observation_window = observation_window
        self.ttc_threshold = ttc_threshold
        self.collision_distance = collision_distance
        self.max_distance = max_distance
        self.lane_arrival_time_threshold = lane_arrival_time_threshold
        self.thw_max = thw_max
        self.safety_ttc_max = safety_ttc_max
        self.eps = eps

    def _assign_lanes(self,
                      position: torch.Tensor,
                      batch: torch.Tensor,
                      lane_position: torch.Tensor,
                      lane_batch: torch.Tensor) -> torch.Tensor:
        num_agents, num_steps = position.shape[:2]
        assigned_lane = torch.full(
            (num_steps * num_agents,),
            -1,
            dtype=torch.long,
            device=position.device,
        )
        if lane_position.numel() == 0 or num_agents == 0:
            return assigned_lane.reshape(num_steps, num_agents).transpose(0, 1)

        flat_position = position.transpose(0, 1).reshape(-1, 2)
        flat_batch = batch.repeat(num_steps)
        for start in range(0, flat_position.size(0), 4096):
            end = min(start + 4096, flat_position.size(0))
            distance = torch.cdist(flat_position[start:end], lane_position)
            same_sample = flat_batch[start:end].unsqueeze(1) == lane_batch.unsqueeze(0)
            distance = distance.masked_fill(~same_sample, float('inf'))
            min_distance, min_index = distance.min(dim=1)
            has_lane = torch.isfinite(min_distance)
            assigned_lane[start:end] = torch.where(
                has_lane,
                min_index,
                assigned_lane[start:end],
            )

        return assigned_lane.reshape(num_steps, num_agents).transpose(0, 1)

    def _compute_lane_speed_statistics(self,
                                       assigned_lane: torch.Tensor,
                                       speed: torch.Tensor,
                                       visible_mask: torch.Tensor,
                                       num_lanes: int) -> tuple:
        num_agents, num_steps = assigned_lane.shape
        empty_statistics = (
            speed.new_zeros((num_steps, num_lanes)),
            speed.new_zeros((num_steps, num_lanes)),
        )
        if num_lanes == 0 or num_agents == 0:
            return empty_statistics

        lane_by_time = assigned_lane.transpose(0, 1).reshape(-1)
        speed_by_time = speed.transpose(0, 1).reshape(-1)
        visible_by_time = visible_mask.transpose(0, 1).reshape(-1)
        time_index = torch.arange(num_steps, device=speed.device).repeat_interleave(num_agents)
        valid = visible_by_time & (lane_by_time >= 0)
        if not valid.any():
            return empty_statistics

        flat_lane_index = time_index[valid] * num_lanes + lane_by_time[valid]
        speed_sum = speed.new_zeros(num_steps * num_lanes)
        lane_count = speed.new_zeros(num_steps * num_lanes)
        speed_sum.scatter_add_(0, flat_lane_index, speed_by_time[valid])
        lane_count.scatter_add_(0, flat_lane_index, torch.ones_like(speed_by_time[valid]))
        return (
            speed_sum.reshape(num_steps, num_lanes),
            lane_count.reshape(num_steps, num_lanes),
        )

    def _compute_dynamic_ttc_threshold(self,
                                       time_index: torch.Tensor,
                                       src_lane: torch.Tensor,
                                       dst_lane: torch.Tensor,
                                       lane_position: torch.Tensor,
                                       lane_speed_sum: torch.Tensor,
                                       lane_vehicle_count: torch.Tensor) -> torch.Tensor:
        threshold = lane_position.new_full(src_lane.shape, self.ttc_threshold)
        valid_lane = (src_lane >= 0) & (dst_lane >= 0)
        if lane_position.numel() == 0 or not valid_lane.any():
            return threshold

        src_lane_valid = src_lane[valid_lane]
        dst_lane_valid = dst_lane[valid_lane]
        lane_center_distance = torch.linalg.vector_norm(
            lane_position[src_lane_valid] - lane_position[dst_lane_valid],
            dim=-1,
        )
        valid_time = time_index[valid_lane]
        src_speed_sum = lane_speed_sum[valid_time, src_lane_valid]
        src_vehicle_count = lane_vehicle_count[valid_time, src_lane_valid]
        different_lane = src_lane_valid != dst_lane_valid
        dst_speed_sum = lane_speed_sum[valid_time, dst_lane_valid]
        dst_vehicle_count = lane_vehicle_count[valid_time, dst_lane_valid]
        combined_speed_sum = src_speed_sum + torch.where(
            different_lane,
            dst_speed_sum,
            torch.zeros_like(dst_speed_sum),
        )
        combined_vehicle_count = src_vehicle_count + torch.where(
            different_lane,
            dst_vehicle_count,
            torch.zeros_like(dst_vehicle_count),
        )
        pair_lane_speed = combined_speed_sum / combined_vehicle_count.clamp_min(1.0)
        threshold[valid_lane] = lane_center_distance / (pair_lane_speed + self.eps)
        return threshold

    def _compute_same_lane_arrival(self,
                                   src_position: torch.Tensor,
                                   dst_position: torch.Tensor,
                                   src_speed: torch.Tensor,
                                   dst_speed: torch.Tensor,
                                   src_lane: torch.Tensor,
                                   dst_lane: torch.Tensor,
                                   lane_position: torch.Tensor) -> tuple:
        arrival_interaction = torch.zeros_like(src_lane, dtype=torch.bool)
        arrival_time_difference = src_speed.new_zeros(src_lane.shape)
        valid = (
            (src_lane >= 0)
            & (src_lane == dst_lane)
            & (src_speed > self.eps)
            & (dst_speed > self.eps)
        )
        if lane_position.numel() == 0 or not valid.any():
            return arrival_interaction, arrival_time_difference

        lane = src_lane[valid]
        src_arrival_time = torch.linalg.vector_norm(
            lane_position[lane] - src_position[valid],
            dim=-1,
        ) / (src_speed[valid] + self.eps)
        dst_arrival_time = torch.linalg.vector_norm(
            lane_position[lane] - dst_position[valid],
            dim=-1,
        ) / (dst_speed[valid] + self.eps)
        arrival_time_difference[valid] = (src_arrival_time - dst_arrival_time).abs()
        arrival_interaction[valid] = (
            arrival_time_difference[valid] <= self.lane_arrival_time_threshold
        )
        return arrival_interaction, arrival_time_difference

    def _compute_interaction_ttc(self,
                                 relative_position: torch.Tensor,
                                 relative_velocity: torch.Tensor) -> torch.Tensor:
        """Constant-velocity TTC using a circular collision envelope."""
        speed_sq = (relative_velocity * relative_velocity).sum(dim=-1)
        linear = 2.0 * (relative_position * relative_velocity).sum(dim=-1)
        constant = (
            (relative_position * relative_position).sum(dim=-1)
            - self.collision_distance ** 2
        )
        discriminant = linear.square() - 4.0 * speed_sq * constant
        first_contact = (
            -linear - torch.sqrt(discriminant.clamp_min(0.0))
        ) / (2.0 * speed_sq.clamp_min(self.eps))

        already_colliding = constant <= 0
        valid_contact = (
            (speed_sq > self.eps)
            & (discriminant >= 0)
            & (first_contact >= 0)
        )
        ttc = torch.full_like(first_contact, float('inf'))
        ttc = torch.where(valid_contact, first_contact, ttc)
        return torch.where(already_colliding, torch.zeros_like(ttc), ttc)

    def _find_shared_conflict(self,
                              src_lane: torch.Tensor,
                              dst_lane: torch.Tensor,
                              conflict_edge_index: torch.Tensor,
                              num_lanes: int) -> torch.Tensor:
        shared_conflict = torch.zeros_like(src_lane, dtype=torch.bool)
        valid_lane = (src_lane >= 0) & (dst_lane >= 0)
        if conflict_edge_index.numel() == 0 or num_lanes == 0 or not valid_lane.any():
            return shared_conflict

        conflict_key = conflict_edge_index[0] * num_lanes + conflict_edge_index[1]
        conflict_key_reverse = conflict_edge_index[1] * num_lanes + conflict_edge_index[0]
        conflict_key = torch.unique(torch.cat([conflict_key, conflict_key_reverse]))
        pair_key = src_lane[valid_lane] * num_lanes + dst_lane[valid_lane]
        shared_conflict[valid_lane] = torch.isin(pair_key, conflict_key)
        return shared_conflict

    @staticmethod
    def _build_conflict_neighborhood(E_VV_msg: torch.Tensor,
                                     D_VV_msg: torch.Tensor,
                                     num_nodes: int) -> tuple:
        """Build incoming adjacency and neighborhoods for vehicle-conflict edges."""
        A_VV_msg = torch.zeros(
            (num_nodes, num_nodes),
            dtype=torch.bool,
            device=E_VV_msg.device,
        )
        if E_VV_msg.numel() != 0:
            source, target = E_VV_msg
            A_VV_msg[target, source] = True

        N_VV_msg = tuple(
            torch.nonzero(A_VV_msg[target], as_tuple=False).flatten()
            for target in range(num_nodes)
        )
        return D_VV_msg, A_VV_msg, N_VV_msg

    @staticmethod
    def _compute_conflict_centrality(E_VV_msg: torch.Tensor,
                                     D_VV_msg: torch.Tensor,
                                     A_VV_msg: torch.Tensor,
                                     node_group: torch.Tensor,
                                     node_visible_mask: torch.Tensor,
                                     eps: float) -> tuple:
        """Compute incoming degree and weighted closeness within each scene-time graph."""
        num_nodes = A_VV_msg.size(0)
        degree_centrality = D_VV_msg.new_zeros(num_nodes)
        closeness_centrality = D_VV_msg.new_zeros(num_nodes)
        if num_nodes == 0 or not node_visible_mask.any():
            return degree_centrality, closeness_centrality

        source, target = E_VV_msg
        visible_group_ids = torch.unique(node_group[node_visible_mask]).detach().cpu().tolist()
        for group_id in visible_group_ids:
            group_nodes = torch.nonzero(
                node_visible_mask & (node_group == group_id),
                as_tuple=False,
            ).flatten()
            group_size = group_nodes.numel()
            if group_size <= 1:
                continue

            group_adjacency = A_VV_msg[group_nodes][:, group_nodes]
            degree_centrality[group_nodes] = (
                group_adjacency.sum(dim=1).to(D_VV_msg.dtype) / (group_size - 1)
            )

            shortest_distance = D_VV_msg.new_full((group_size, group_size), float('inf'))
            diagonal = torch.arange(group_size, device=D_VV_msg.device)
            shortest_distance[diagonal, diagonal] = 0.0

            if E_VV_msg.numel() != 0:
                edge_in_group = (
                    node_visible_mask[source]
                    & node_visible_mask[target]
                    & (node_group[source] == group_id)
                    & (node_group[target] == group_id)
                )
                if edge_in_group.any():
                    global_to_local = torch.full(
                        (num_nodes,),
                        -1,
                        dtype=torch.long,
                        device=E_VV_msg.device,
                    )
                    global_to_local[group_nodes] = diagonal
                    local_source = global_to_local[source[edge_in_group]]
                    local_target = global_to_local[target[edge_in_group]]
                    shortest_distance[local_target, local_source] = D_VV_msg[edge_in_group]

            for intermediate in range(group_size):
                through_intermediate = (
                    shortest_distance[:, intermediate:intermediate + 1]
                    + shortest_distance[intermediate:intermediate + 1, :]
                )
                shortest_distance = torch.minimum(shortest_distance, through_intermediate)

            reachable = torch.isfinite(shortest_distance)
            reachable[diagonal, diagonal] = False
            reachable_count = reachable.sum(dim=1)
            distance_sum = torch.where(
                reachable,
                shortest_distance,
                torch.zeros_like(shortest_distance),
            ).sum(dim=1)
            has_reachable_node = reachable_count > 0
            group_closeness = D_VV_msg.new_zeros(group_size)
            reachable_count_float = reachable_count.to(D_VV_msg.dtype)
            group_closeness[has_reachable_node] = (
                reachable_count_float[has_reachable_node]
                / distance_sum[has_reachable_node].clamp_min(eps)
                * reachable_count_float[has_reachable_node]
                / (group_size - 1)
            )
            closeness_centrality[group_nodes] = group_closeness

        return degree_centrality, closeness_centrality

    @staticmethod
    def _compute_centrality_changes(centrality: torch.Tensor,
                                    node_visible_mask: torch.Tensor,
                                    num_agents: int,
                                    num_steps: int) -> tuple:
        """Compute causal frame-to-frame and second-order centrality changes."""
        centrality_by_time = centrality.reshape(num_steps, num_agents)
        visible_by_time = node_visible_mask.reshape(num_steps, num_agents)
        first_change = torch.zeros_like(centrality_by_time)
        second_change = torch.zeros_like(centrality_by_time)
        first_valid = torch.zeros_like(visible_by_time)
        second_valid = torch.zeros_like(visible_by_time)

        if num_steps > 1:
            first_valid[1:] = visible_by_time[1:] & visible_by_time[:-1]
            first_delta = centrality_by_time[1:] - centrality_by_time[:-1]
            first_change[1:] = torch.where(
                first_valid[1:],
                first_delta,
                torch.zeros_like(first_delta),
            )

        if num_steps > 2:
            second_valid[2:] = first_valid[2:] & first_valid[1:-1]
            second_delta = first_change[2:] - first_change[1:-1]
            second_change[2:] = torch.where(
                second_valid[2:],
                second_delta,
                torch.zeros_like(second_delta),
            )

        return (
            first_change.reshape(-1),
            second_change.reshape(-1),
            first_valid.reshape(-1),
            second_valid.reshape(-1),
        )

    @staticmethod
    def _compute_safety_features(E_VV_msg: torch.Tensor,
                                 position: torch.Tensor,
                                 velocity: torch.Tensor,
                                 assigned_lane: torch.Tensor,
                                 lane_heading: Optional[torch.Tensor],
                                 reference_lane: Optional[torch.Tensor],
                                 left_neighbor_edge_index: Optional[torch.Tensor],
                                 right_neighbor_edge_index: Optional[torch.Tensor],
                                 thw_max: float,
                                 safety_ttc_max: float,
                                 eps: float) -> tuple:
        """Compute same/adjacent-lane front-vehicle THW and TTC for each conflict edge."""
        num_edges = E_VV_msg.size(1)
        THW_ij = position.new_zeros(num_edges)
        TTC_ij = position.new_zeros(num_edges)
        longitudinal_gap = position.new_zeros(num_edges)
        valid_mask = torch.zeros(num_edges, dtype=torch.bool, device=position.device)
        if num_edges == 0 or lane_heading is None or lane_heading.numel() == 0:
            return THW_ij, TTC_ij, longitudinal_gap, valid_mask

        num_agents = position.size(0)
        source_node, target_node = E_VV_msg
        source_time = torch.div(source_node, num_agents, rounding_mode='floor')
        target_time = torch.div(target_node, num_agents, rounding_mode='floor')
        source_agent = source_node % num_agents
        target_agent = target_node % num_agents

        lane_assignment = assigned_lane if reference_lane is None else reference_lane
        source_lane = lane_assignment[source_agent, source_time]
        target_lane = lane_assignment[target_agent, target_time]
        valid_lane = (
            (source_time == target_time)
            & (source_lane >= 0)
            & (target_lane >= 0)
        )
        same_or_adjacent_lane = source_lane == target_lane
        num_lanes = lane_heading.numel()
        adjacent_edges = []
        if left_neighbor_edge_index is not None and left_neighbor_edge_index.numel() != 0:
            adjacent_edges.append(left_neighbor_edge_index)
        if right_neighbor_edge_index is not None and right_neighbor_edge_index.numel() != 0:
            adjacent_edges.append(right_neighbor_edge_index)
        if adjacent_edges:
            adjacent_edge_index = torch.cat(adjacent_edges, dim=1)
            adjacent_key = torch.unique(torch.cat([
                adjacent_edge_index[0] * num_lanes + adjacent_edge_index[1],
                adjacent_edge_index[1] * num_lanes + adjacent_edge_index[0],
            ]))
            safe_source_lane = source_lane.clamp(min=0, max=num_lanes - 1)
            safe_target_lane = target_lane.clamp(min=0, max=num_lanes - 1)
            pair_key = safe_source_lane * num_lanes + safe_target_lane
            same_or_adjacent_lane = same_or_adjacent_lane | torch.isin(pair_key, adjacent_key)
        valid_lane = valid_lane & same_or_adjacent_lane
        if not valid_lane.any():
            return THW_ij, TTC_ij, longitudinal_gap, valid_mask

        safe_target_lane = target_lane.clamp(min=0, max=lane_heading.numel() - 1)
        target_lane_heading = lane_heading[safe_target_lane]
        lane_direction = torch.stack([
            torch.cos(target_lane_heading),
            torch.sin(target_lane_heading),
        ], dim=-1)
        relative_position = (
            position[source_agent, source_time]
            - position[target_agent, target_time]
        )
        longitudinal_gap = (relative_position * lane_direction).sum(dim=-1)
        valid_mask = valid_lane & (longitudinal_gap > 0)
        if not valid_mask.any():
            return THW_ij, TTC_ij, longitudinal_gap, valid_mask

        source_speed = torch.linalg.vector_norm(
            velocity[source_agent, source_time],
            dim=-1,
        )
        target_speed = torch.linalg.vector_norm(
            velocity[target_agent, target_time],
            dim=-1,
        )
        THW_ij[valid_mask] = (
            longitudinal_gap[valid_mask]
            / (target_speed[valid_mask] + eps)
        ).clamp(max=thw_max)
        closing_speed = target_speed - source_speed
        TTC_ij[valid_mask] = (
            longitudinal_gap[valid_mask]
            / closing_speed[valid_mask].clamp_min(eps)
        ).clamp(max=safety_ttc_max)
        return THW_ij, TTC_ij, longitudinal_gap, valid_mask

    def forward(self,
                position: torch.Tensor,
                velocity: torch.Tensor,
                visible_mask: torch.Tensor,
                batch: torch.Tensor,
                lane_position: torch.Tensor,
                lane_batch: torch.Tensor,
                conflict_edge_index: torch.Tensor,
                lane_heading: Optional[torch.Tensor] = None,
                reference_lane: Optional[torch.Tensor] = None,
                assigned_lane: Optional[torch.Tensor] = None,
                left_neighbor_edge_index: Optional[torch.Tensor] = None,
                right_neighbor_edge_index: Optional[torch.Tensor] = None) -> VehicleInteractionGraph:
        """Return directed edges for all historical steps without future leakage."""
        if position.ndim != 3 or position.size(-1) != 2:
            raise ValueError('position must have shape [num_agents, num_steps, 2]')
        if velocity.shape != position.shape:
            raise ValueError('velocity must have the same shape as position')
        if visible_mask.shape != position.shape[:2]:
            raise ValueError('visible_mask must have shape [num_agents, num_steps]')
        if lane_heading is not None and lane_heading.shape != lane_position.shape[:1]:
            raise ValueError('lane_heading must have shape [num_lanes]')
        if reference_lane is not None and reference_lane.shape != position.shape[:2]:
            raise ValueError('reference_lane must have shape [num_agents, num_steps]')
        if assigned_lane is not None and assigned_lane.shape != position.shape[:2]:
            raise ValueError('assigned_lane must have shape [num_agents, num_steps]')

        num_agents, num_steps = position.shape[:2]
        num_nodes = num_steps * num_agents
        node_visible_mask = visible_mask.transpose(0, 1).reshape(-1)
        if num_nodes == 0:
            node_group = torch.empty(0, dtype=torch.long, device=position.device)
        else:
            batch_stride = int(batch.max().item()) + 1
            node_time = torch.arange(num_steps, device=position.device).repeat_interleave(num_agents)
            node_group = node_time * batch_stride + batch.repeat(num_steps)
        if assigned_lane is None:
            assigned_lane = self._assign_lanes(
                position, batch, lane_position, lane_batch,
            )
        if num_agents == 0 or num_steps == 0:
            E_VV_msg = torch.empty((2, 0), dtype=torch.long, device=position.device)
            D_VV_msg, A_VV_msg, N_VV_msg = self._build_conflict_neighborhood(
                E_VV_msg=E_VV_msg,
                D_VV_msg=position.new_zeros(0),
                num_nodes=num_nodes,
            )
            degree_centrality, closeness_centrality = self._compute_conflict_centrality(
                E_VV_msg=E_VV_msg,
                D_VV_msg=D_VV_msg,
                A_VV_msg=A_VV_msg,
                node_group=node_group,
                node_visible_mask=node_visible_mask,
                eps=self.eps,
            )
            (
                degree_centrality_change,
                degree_centrality_second_change,
                centrality_change_valid_mask,
                centrality_second_change_valid_mask,
            ) = self._compute_centrality_changes(
                degree_centrality,
                node_visible_mask,
                num_agents,
                num_steps,
            )
            (
                closeness_centrality_change,
                closeness_centrality_second_change,
                _,
                _,
            ) = self._compute_centrality_changes(
                closeness_centrality,
                node_visible_mask,
                num_agents,
                num_steps,
            )
            return VehicleInteractionGraph(
                E_VV_msg=E_VV_msg,
                D_VV_msg=D_VV_msg,
                A_VV_msg=A_VV_msg,
                N_VV_msg=N_VV_msg,
                degree_centrality=degree_centrality,
                closeness_centrality=closeness_centrality,
                degree_centrality_change=degree_centrality_change,
                degree_centrality_second_change=degree_centrality_second_change,
                closeness_centrality_change=closeness_centrality_change,
                closeness_centrality_second_change=closeness_centrality_second_change,
                centrality_change_valid_mask=centrality_change_valid_mask,
                centrality_second_change_valid_mask=centrality_second_change_valid_mask,
                THW_ij=position.new_zeros(0),
                TTC_ij=position.new_zeros(0),
                safety_longitudinal_gap=position.new_zeros(0),
                safety_feature_valid_mask=torch.zeros(0, dtype=torch.bool, device=position.device),
                same_lane_arrival_time_difference=position.new_zeros(0),
                edge_attr=position.new_zeros((0, len(self.EDGE_ATTR_NAMES))),
                assigned_lane=assigned_lane,
            )

        same_sample = batch.unsqueeze(1) == batch.unsqueeze(0)
        different_agent = ~torch.eye(num_agents, dtype=torch.bool, device=position.device)
        visible_by_time = visible_mask.transpose(0, 1)
        valid_pair = (
            visible_by_time.unsqueeze(2)
            & visible_by_time.unsqueeze(1)
            & same_sample.unsqueeze(0)
            & different_agent.unsqueeze(0)
        )
        candidate = torch.nonzero(valid_pair, as_tuple=False)
        if candidate.numel() == 0:
            E_VV_msg = torch.empty((2, 0), dtype=torch.long, device=position.device)
            D_VV_msg, A_VV_msg, N_VV_msg = self._build_conflict_neighborhood(
                E_VV_msg=E_VV_msg,
                D_VV_msg=position.new_zeros(0),
                num_nodes=num_nodes,
            )
            degree_centrality, closeness_centrality = self._compute_conflict_centrality(
                E_VV_msg=E_VV_msg,
                D_VV_msg=D_VV_msg,
                A_VV_msg=A_VV_msg,
                node_group=node_group,
                node_visible_mask=node_visible_mask,
                eps=self.eps,
            )
            (
                degree_centrality_change,
                degree_centrality_second_change,
                centrality_change_valid_mask,
                centrality_second_change_valid_mask,
            ) = self._compute_centrality_changes(
                degree_centrality,
                node_visible_mask,
                num_agents,
                num_steps,
            )
            (
                closeness_centrality_change,
                closeness_centrality_second_change,
                _,
                _,
            ) = self._compute_centrality_changes(
                closeness_centrality,
                node_visible_mask,
                num_agents,
                num_steps,
            )
            return VehicleInteractionGraph(
                E_VV_msg=E_VV_msg,
                D_VV_msg=D_VV_msg,
                A_VV_msg=A_VV_msg,
                N_VV_msg=N_VV_msg,
                degree_centrality=degree_centrality,
                closeness_centrality=closeness_centrality,
                degree_centrality_change=degree_centrality_change,
                degree_centrality_second_change=degree_centrality_second_change,
                closeness_centrality_change=closeness_centrality_change,
                closeness_centrality_second_change=closeness_centrality_second_change,
                centrality_change_valid_mask=centrality_change_valid_mask,
                centrality_second_change_valid_mask=centrality_second_change_valid_mask,
                THW_ij=position.new_zeros(0),
                TTC_ij=position.new_zeros(0),
                safety_longitudinal_gap=position.new_zeros(0),
                safety_feature_valid_mask=torch.zeros(0, dtype=torch.bool, device=position.device),
                same_lane_arrival_time_difference=position.new_zeros(0),
                edge_attr=position.new_zeros((0, len(self.EDGE_ATTR_NAMES))),
                assigned_lane=assigned_lane,
            )

        time_index, src_agent, dst_agent = candidate.unbind(dim=1)
        src_position = position[src_agent, time_index]
        dst_position = position[dst_agent, time_index]
        src_velocity = velocity[src_agent, time_index]
        dst_velocity = velocity[dst_agent, time_index]

        relative_position = src_position - dst_position
        relative_velocity = src_velocity - dst_velocity
        distance = torch.linalg.vector_norm(relative_position, dim=-1)
        src_speed = torch.linalg.vector_norm(src_velocity, dim=-1)
        dst_speed = torch.linalg.vector_norm(dst_velocity, dim=-1)
        dynamic_distance = (
            self.distance_margin
            + dst_speed * self.observation_window
        ).clamp(max=self.max_distance)
        distance_trigger = distance <= dynamic_distance

        interaction_ttc = self._compute_interaction_ttc(relative_position, relative_velocity)
        within_max_distance = distance <= self.max_distance
        src_lane = assigned_lane[src_agent, time_index]
        dst_lane = assigned_lane[dst_agent, time_index]
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        lane_speed_sum, lane_vehicle_count = self._compute_lane_speed_statistics(
            assigned_lane,
            speed,
            visible_mask,
            lane_position.size(0),
        )
        dynamic_ttc_threshold = self._compute_dynamic_ttc_threshold(
            time_index,
            src_lane,
            dst_lane,
            lane_position,
            lane_speed_sum,
            lane_vehicle_count,
        )
        ttc_trigger = (interaction_ttc <= dynamic_ttc_threshold) & within_max_distance

        shared_conflict = self._find_shared_conflict(
            src_lane,
            dst_lane,
            conflict_edge_index,
            lane_position.size(0),
        ) & within_max_distance
        same_lane_arrival, same_lane_arrival_time_difference = self._compute_same_lane_arrival(
            src_position,
            dst_position,
            src_speed,
            dst_speed,
            src_lane,
            dst_lane,
            lane_position,
        )
        same_lane_arrival = same_lane_arrival & within_max_distance

        vv_conflict_mask = distance_trigger | ttc_trigger | shared_conflict | same_lane_arrival
        time_index = time_index[vv_conflict_mask]
        src_agent = src_agent[vv_conflict_mask]
        dst_agent = dst_agent[vv_conflict_mask]
        E_VV_msg = torch.stack([
            time_index * num_agents + src_agent,
            time_index * num_agents + dst_agent,
        ], dim=0)
        D_VV_msg, A_VV_msg, N_VV_msg = self._build_conflict_neighborhood(
            E_VV_msg=E_VV_msg,
            D_VV_msg=distance[vv_conflict_mask],
            num_nodes=num_nodes,
        )
        degree_centrality, closeness_centrality = self._compute_conflict_centrality(
            E_VV_msg=E_VV_msg,
            D_VV_msg=D_VV_msg,
            A_VV_msg=A_VV_msg,
            node_group=node_group,
            node_visible_mask=node_visible_mask,
            eps=self.eps,
        )
        (
            degree_centrality_change,
            degree_centrality_second_change,
            centrality_change_valid_mask,
            centrality_second_change_valid_mask,
        ) = self._compute_centrality_changes(
            degree_centrality,
            node_visible_mask,
            num_agents,
            num_steps,
        )
        (
            closeness_centrality_change,
            closeness_centrality_second_change,
            _,
            _,
        ) = self._compute_centrality_changes(
            closeness_centrality,
            node_visible_mask,
            num_agents,
            num_steps,
        )
        (
            THW_ij,
            TTC_ij,
            safety_longitudinal_gap,
            safety_feature_valid_mask,
        ) = self._compute_safety_features(
            E_VV_msg=E_VV_msg,
            position=position,
            velocity=velocity,
            assigned_lane=assigned_lane,
            lane_heading=lane_heading,
            reference_lane=reference_lane,
            left_neighbor_edge_index=left_neighbor_edge_index,
            right_neighbor_edge_index=right_neighbor_edge_index,
            thw_max=self.thw_max,
            safety_ttc_max=self.safety_ttc_max,
            eps=self.eps,
        )

        clipped_interaction_ttc = torch.nan_to_num(
            interaction_ttc[vv_conflict_mask],
            nan=self.ttc_threshold,
            posinf=self.ttc_threshold,
            neginf=0.0,
        ).clamp(min=0.0, max=self.ttc_threshold)
        edge_attr = torch.stack([
            D_VV_msg,
            clipped_interaction_ttc,
            distance_trigger[vv_conflict_mask].to(position.dtype),
            ttc_trigger[vv_conflict_mask].to(position.dtype),
            shared_conflict[vv_conflict_mask].to(position.dtype),
            same_lane_arrival[vv_conflict_mask].to(position.dtype),
        ], dim=-1)
        return VehicleInteractionGraph(
            E_VV_msg=E_VV_msg,
            D_VV_msg=D_VV_msg,
            A_VV_msg=A_VV_msg,
            N_VV_msg=N_VV_msg,
            degree_centrality=degree_centrality,
            closeness_centrality=closeness_centrality,
            degree_centrality_change=degree_centrality_change,
            degree_centrality_second_change=degree_centrality_second_change,
            closeness_centrality_change=closeness_centrality_change,
            closeness_centrality_second_change=closeness_centrality_second_change,
            centrality_change_valid_mask=centrality_change_valid_mask,
            centrality_second_change_valid_mask=centrality_second_change_valid_mask,
            THW_ij=THW_ij,
            TTC_ij=TTC_ij,
            safety_longitudinal_gap=safety_longitudinal_gap,
            safety_feature_valid_mask=safety_feature_valid_mask,
            same_lane_arrival_time_difference=same_lane_arrival_time_difference[vv_conflict_mask],
            edge_attr=edge_attr,
            assigned_lane=assigned_lane,
        )

    @staticmethod
    def expand_across_modes(graph: VehicleInteractionGraph,
                            num_agents: int,
                            num_modes: int) -> tuple:
        """Lift ``[time, agent]`` graph nodes to PNet's ``[time, mode, agent]`` layout."""
        if graph.E_VV_msg.numel() == 0:
            return graph.E_VV_msg, graph.edge_attr

        src, dst = graph.E_VV_msg
        src_time, src_agent = torch.div(src, num_agents, rounding_mode='floor'), src % num_agents
        dst_time, dst_agent = torch.div(dst, num_agents, rounding_mode='floor'), dst % num_agents
        mode = torch.arange(num_modes, device=src.device).unsqueeze(0)
        expanded_src = (src_time.unsqueeze(1) * num_modes + mode) * num_agents + src_agent.unsqueeze(1)
        expanded_dst = (dst_time.unsqueeze(1) * num_modes + mode) * num_agents + dst_agent.unsqueeze(1)
        expanded_edge_index = torch.stack([
            expanded_src.reshape(-1),
            expanded_dst.reshape(-1),
        ], dim=0)
        return expanded_edge_index, graph.edge_attr.repeat_interleave(num_modes, dim=0)
