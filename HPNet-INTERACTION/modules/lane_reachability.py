import functools
import heapq
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from utils import wrap_angle


LANE_ASSOCIATION_MAIN_LANE_LOCAL = 'lane_association_main_lane_local'
LANE_ASSOCIATION_PROJECTION_S = 'lane_association_projection_s'
LANE_ASSOCIATION_LATERAL_DISTANCE = 'lane_association_lateral_distance'
LANE_ASSOCIATION_HEADING_DIFFERENCE = 'lane_association_heading_difference'
LANE_ASSOCIATION_AUXILIARY_EDGE_TYPE = (
    'agent', 'lane_association_auxiliary', 'lane',
)


class LaneAssociation(NamedTuple):
    main_lane: torch.Tensor
    projection_s: torch.Tensor
    lateral_distance: torch.Tensor
    heading_difference: torch.Tensor
    auxiliary_seed_edge_index: torch.Tensor


def load_lane_association_from_data(
        data,
        num_historical_steps: int) -> Optional[LaneAssociation]:
    """Load per-sample worker results and convert local lanes to batch indices."""
    agent_store = data['agent']
    field_names = (
        LANE_ASSOCIATION_MAIN_LANE_LOCAL,
        LANE_ASSOCIATION_PROJECTION_S,
        LANE_ASSOCIATION_LATERAL_DISTANCE,
        LANE_ASSOCIATION_HEADING_DIFFERENCE,
    )
    present = [field_name in agent_store for field_name in field_names]
    if not any(present):
        return None
    if not all(present):
        missing = [
            field_name
            for field_name, is_present in zip(field_names, present)
            if not is_present
        ]
        raise ValueError(
            f'incomplete worker lane association fields: {missing}'
        )

    main_lane_local = agent_store[LANE_ASSOCIATION_MAIN_LANE_LOCAL]
    expected_shape = (agent_store.num_nodes, num_historical_steps)
    if main_lane_local.shape != expected_shape:
        raise ValueError(
            'lane_association_main_lane_local must have shape '
            f'{expected_shape}, got {tuple(main_lane_local.shape)}'
        )
    if main_lane_local.dtype != torch.long:
        raise ValueError('lane_association_main_lane_local must use torch.long')

    dense_fields = (
        agent_store[LANE_ASSOCIATION_PROJECTION_S],
        agent_store[LANE_ASSOCIATION_LATERAL_DISTANCE],
        agent_store[LANE_ASSOCIATION_HEADING_DIFFERENCE],
    )
    if any(field.shape != expected_shape for field in dense_fields):
        raise ValueError(
            'worker lane association scalar fields must match main_lane shape'
        )

    if 'batch' in agent_store:
        if 'ptr' not in data['lane']:
            raise ValueError(
                'batched worker lane association requires data["lane"]["ptr"]'
            )
        lane_offset = data['lane']['ptr'][agent_store['batch']]
    else:
        lane_offset = torch.zeros(
            main_lane_local.size(0),
            dtype=torch.long,
            device=main_lane_local.device,
        )
    main_lane = torch.where(
        main_lane_local >= 0,
        main_lane_local + lane_offset.unsqueeze(-1),
        main_lane_local,
    )

    if LANE_ASSOCIATION_AUXILIARY_EDGE_TYPE in data.edge_types:
        auxiliary_store = data[LANE_ASSOCIATION_AUXILIARY_EDGE_TYPE]
        auxiliary_agent_lane = auxiliary_store['edge_index']
        historical_step = auxiliary_store['historical_step']
        auxiliary_source = (
            historical_step * main_lane.size(0)
            + auxiliary_agent_lane[0]
        )
        auxiliary_seed_edge_index = torch.stack([
            auxiliary_source,
            auxiliary_agent_lane[1],
        ], dim=0)
    else:
        auxiliary_seed_edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=main_lane.device,
        )

    return LaneAssociation(
        main_lane=main_lane,
        projection_s=dense_fields[0],
        lateral_distance=dense_fields[1],
        heading_difference=dense_fields[2],
        auxiliary_seed_edge_index=auxiliary_seed_edge_index,
    )


class LaneReachability(NamedTuple):
    edge_index: torch.Tensor
    path_distance: torch.Tensor
    lane_change_count: torch.Tensor
    distance_budget: torch.Tensor
    longitudinal_speed: torch.Tensor


class ConflictInstances(NamedTuple):
    pair_index: torch.Tensor
    conflict_lane: torch.Tensor
    distance_i: torch.Tensor
    distance_j: torch.Tensor
    tau_raw_i: torch.Tensor
    tau_raw_j: torch.Tensor
    tau_clip_i: torch.Tensor
    tau_clip_j: torch.Tensor
    urgency_cost: torch.Tensor
    valid_mask: torch.Tensor
    pair_group: torch.Tensor
    canonical_pair_index: torch.Tensor
    canonical_edge_attr: torch.Tensor
    E_VV_conf: torch.Tensor
    directed_edge_attr: torch.Tensor


def _relation_by_current(edge_index: torch.Tensor,
                         num_lanes: int) -> List[set]:
    relation = [set() for _ in range(num_lanes)]
    if edge_index.numel() == 0:
        return relation
    for related, current in edge_index.detach().cpu().t().tolist():
        if 0 <= related < num_lanes and 0 <= current < num_lanes:
            relation[current].add(related)
    return relation


def _directed_relation(edge_index: torch.Tensor,
                       num_lanes: int) -> List[set]:
    relation = [set() for _ in range(num_lanes)]
    if edge_index.numel() == 0:
        return relation
    for source, target in edge_index.detach().cpu().t().tolist():
        if 0 <= source < num_lanes and 0 <= target < num_lanes:
            relation[source].add(target)
    return relation


class LaneReachabilityBuilder:
    """Causal lanelet association and forward-reachability construction."""

    def __init__(self,
                 centerline_side_index: int = 1,
                 default_lateral_distance: float = 2.5,
                 centerline_spacing: float = 2.0,
                 lane_change_distance: float = 12.0,
                 max_lane_changes: int = 1,
                 tau_max: float = 5.0,
                 conflict_arrival_threshold: float = 2.0,
                 softmin_temperature: float = 0.2) -> None:
        self.centerline_side_index = centerline_side_index
        self.default_lateral_distance = default_lateral_distance
        self.centerline_spacing = centerline_spacing
        self.lane_change_distance = lane_change_distance
        self.max_lane_changes = max_lane_changes
        self.tau_max = tau_max
        self.conflict_arrival_threshold = conflict_arrival_threshold
        self.softmin_temperature = softmin_temperature

    @staticmethod
    def _centerline_segments(
            num_lanes: int,
            polyline_position: torch.Tensor,
            polyline_heading: torch.Tensor,
            polyline_length: torch.Tensor,
            polyline_side: torch.Tensor,
            polyline_to_lane_edge_index: torch.Tensor,
            centerline_side_index: int) -> List[Tuple[torch.Tensor, ...]]:
        result: List[Tuple[torch.Tensor, ...]] = []
        source, target = polyline_to_lane_edge_index
        center_mask = polyline_side[source].long() == centerline_side_index
        for lane_idx in range(num_lanes):
            lane_source = source[center_mask & (target == lane_idx)]
            lane_source = lane_source.sort().values
            result.append((
                polyline_position[lane_source],
                polyline_heading[lane_source],
                polyline_length[lane_source],
            ))
        return result

    @staticmethod
    def _project_to_centerline(
            point: torch.Tensor,
            segment_position: torch.Tensor,
            segment_heading: torch.Tensor,
            segment_length: torch.Tensor,
            endpoint_tolerance: float) -> Tuple[float, float, bool]:
        if segment_position.numel() == 0:
            return 0.0, float('inf'), False
        direction = torch.stack([
            torch.cos(segment_heading),
            torch.sin(segment_heading),
        ], dim=-1)
        segment_start = segment_position - 0.5 * segment_length.unsqueeze(-1) * direction
        relative = point.unsqueeze(0) - segment_start
        raw_projection = (relative * direction).sum(dim=-1)
        clamped_projection = torch.minimum(
            raw_projection.clamp_min(0.0),
            segment_length,
        )
        closest_point = segment_start + clamped_projection.unsqueeze(-1) * direction
        distance = torch.linalg.vector_norm(point.unsqueeze(0) - closest_point, dim=-1)
        segment_idx = int(distance.argmin().item())
        prefix_length = segment_length[:segment_idx].sum()
        projection_s = prefix_length + clamped_projection[segment_idx]

        longitudinal_valid = True
        if segment_idx == 0 and raw_projection[segment_idx] < 0:
            longitudinal_valid = bool(
                -raw_projection[segment_idx] <= endpoint_tolerance
            )
        if (
            segment_idx == segment_length.numel() - 1
            and raw_projection[segment_idx] > segment_length[segment_idx]
        ):
            longitudinal_valid = bool(
                raw_projection[segment_idx] - segment_length[segment_idx]
                <= endpoint_tolerance
            )
        return (
            float(projection_s.item()),
            float(distance[segment_idx].item()),
            longitudinal_valid,
        )

    def associate(
            self,
            agent_position: torch.Tensor,
            agent_heading: torch.Tensor,
            agent_speed: torch.Tensor,
            agent_heading_valid_mask: torch.Tensor,
            agent_visible_mask: torch.Tensor,
            agent_batch: torch.Tensor,
            lane_heading: torch.Tensor,
            lane_width: torch.Tensor,
            lane_drivable: torch.Tensor,
            lane_visible_mask: torch.Tensor,
            lane_batch: torch.Tensor,
            lane_centerline_spacing: torch.Tensor,
            predecessor_edge_index: torch.Tensor,
            successor_edge_index: torch.Tensor,
            left_change_edge_index: torch.Tensor,
            right_change_edge_index: torch.Tensor,
            adjacent_left_edge_index: torch.Tensor,
            adjacent_right_edge_index: torch.Tensor,
            polyline_position: torch.Tensor,
            polyline_heading: torch.Tensor,
            polyline_length: torch.Tensor,
            polyline_side: torch.Tensor,
            polyline_to_lane_edge_index: torch.Tensor) -> LaneAssociation:
        num_agents, num_steps = agent_position.shape[:2]
        num_lanes = lane_heading.numel()
        device = agent_position.device
        main_lane = torch.full(
            (num_agents, num_steps), -1, dtype=torch.long, device=device,
        )
        projection_s = agent_position.new_zeros((num_agents, num_steps))
        lateral_distance = agent_position.new_zeros((num_agents, num_steps))
        heading_difference = agent_position.new_zeros((num_agents, num_steps))
        if num_agents == 0 or num_lanes == 0:
            return LaneAssociation(
                main_lane,
                projection_s,
                lateral_distance,
                heading_difference,
                torch.empty((2, 0), dtype=torch.long, device=device),
            )

        centerline_segments = self._centerline_segments(
            num_lanes,
            polyline_position,
            polyline_heading,
            polyline_length,
            polyline_side,
            polyline_to_lane_edge_index,
            self.centerline_side_index,
        )
        predecessor = _relation_by_current(predecessor_edge_index, num_lanes)
        successor = _relation_by_current(successor_edge_index, num_lanes)
        left = _relation_by_current(left_change_edge_index, num_lanes)
        right = _relation_by_current(right_change_edge_index, num_lanes)
        adjacent_left = _relation_by_current(adjacent_left_edge_index, num_lanes)
        adjacent_right = _relation_by_current(adjacent_right_edge_index, num_lanes)
        adjacent = [
            left[idx] | right[idx] | adjacent_left[idx] | adjacent_right[idx]
            for idx in range(num_lanes)
        ]

        auxiliary_pairs: List[Tuple[int, int]] = []
        lane_heading_cpu = lane_heading.detach().cpu()
        lane_width_cpu = lane_width.detach().cpu()
        lane_drivable_cpu = lane_drivable.detach().cpu().bool().tolist()
        lane_visible_cpu = lane_visible_mask.detach().cpu().bool().tolist()
        lane_batch_cpu = lane_batch.detach().cpu().tolist()
        agent_batch_cpu = agent_batch.detach().cpu().tolist()
        visible_cpu = agent_visible_mask.detach().cpu().bool().tolist()
        speed_cpu = agent_speed.detach().cpu()
        heading_cpu = agent_heading.detach().cpu()
        heading_valid_cpu = agent_heading_valid_mask.detach().cpu().bool()
        position_cpu = agent_position.detach().cpu()
        spacing_cpu = lane_centerline_spacing.detach().cpu()

        for agent_idx in range(num_agents):
            for time_idx in range(num_steps):
                if not visible_cpu[agent_idx][time_idx]:
                    continue
                previous_lane = -1
                if time_idx > 0 and visible_cpu[agent_idx][time_idx - 1]:
                    previous_lane = int(main_lane[agent_idx, time_idx - 1].item())

                candidates = []
                for lane_idx in range(num_lanes):
                    if (
                        not lane_visible_cpu[lane_idx]
                        or not lane_drivable_cpu[lane_idx]
                        or lane_batch_cpu[lane_idx] != agent_batch_cpu[agent_idx]
                    ):
                        continue

                    spacing = float(spacing_cpu[lane_idx].item())
                    endpoint_tolerance = min(4.0, 1.5 * spacing)
                    segment_position, segment_heading, segment_length = (
                        segment.detach().cpu()
                        for segment in centerline_segments[lane_idx]
                    )
                    s_value, lateral_value, longitudinal_valid = self._project_to_centerline(
                        position_cpu[agent_idx, time_idx],
                        segment_position,
                        segment_heading,
                        segment_length,
                        endpoint_tolerance,
                    )
                    if not longitudinal_valid:
                        continue

                    lane_width_value = float(lane_width_cpu[lane_idx].item())
                    if lane_width_value > 0 and torch.isfinite(lane_width_cpu[lane_idx]):
                        lateral_limit = min(2.8, max(2.0, lane_width_value / 2.0 + 0.5))
                    else:
                        lateral_limit = self.default_lateral_distance
                    if lateral_value > lateral_limit:
                        continue

                    heading_diff = float(abs(wrap_angle(
                        heading_cpu[agent_idx, time_idx]
                        - lane_heading_cpu[lane_idx]
                    ).item()))
                    speed_value = float(speed_cpu[agent_idx, time_idx].item())
                    if bool(heading_valid_cpu[agent_idx, time_idx]) and speed_value > 0.5:
                        heading_limit = torch.pi / 6 if speed_value >= 1.0 else torch.pi / 3
                        if heading_diff > heading_limit:
                            continue

                    topology_rank = 4
                    if previous_lane >= 0:
                        if lane_idx == previous_lane:
                            topology_rank = 0
                        elif lane_idx in successor[previous_lane]:
                            topology_rank = 1
                        elif lane_idx in predecessor[previous_lane]:
                            topology_rank = 2
                        elif lane_idx in adjacent[previous_lane]:
                            topology_rank = 3
                    candidates.append({
                        'lane': lane_idx,
                        's': s_value,
                        'lateral': lateral_value,
                        'heading': heading_diff,
                        'rank': topology_rank,
                    })

                if not candidates:
                    continue

                def compare(first: Dict, second: Dict) -> int:
                    if first['rank'] != second['rank']:
                        return -1 if first['rank'] < second['rank'] else 1
                    if abs(first['lateral'] - second['lateral']) < 0.3:
                        if first['heading'] != second['heading']:
                            return -1 if first['heading'] < second['heading'] else 1
                    elif first['lateral'] != second['lateral']:
                        return -1 if first['lateral'] < second['lateral'] else 1
                    return -1 if first['lane'] < second['lane'] else int(first['lane'] > second['lane'])

                candidates.sort(key=functools.cmp_to_key(compare))
                selected = candidates[0]
                main_lane[agent_idx, time_idx] = selected['lane']
                projection_s[agent_idx, time_idx] = selected['s']
                lateral_distance[agent_idx, time_idx] = selected['lateral']
                heading_difference[agent_idx, time_idx] = selected['heading']
                node_idx = time_idx * num_agents + agent_idx
                auxiliary_pairs.extend(
                    (node_idx, candidate['lane']) for candidate in candidates[1:]
                )

        if auxiliary_pairs:
            auxiliary_edge_index = torch.tensor(
                auxiliary_pairs, dtype=torch.long, device=device,
            ).t().contiguous()
        else:
            auxiliary_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        return LaneAssociation(
            main_lane,
            projection_s,
            lateral_distance,
            heading_difference,
            auxiliary_edge_index,
        )

    def build_reachability(
            self,
            association: LaneAssociation,
            agent_velocity: torch.Tensor,
            agent_acceleration: torch.Tensor,
            agent_visible_mask: torch.Tensor,
            lane_heading: torch.Tensor,
            lane_length: torch.Tensor,
            lane_start_position: torch.Tensor,
            lane_end_position: torch.Tensor,
            lane_drivable: torch.Tensor,
            lane_visible_mask: torch.Tensor,
            successor_edge_index: torch.Tensor,
            lane_change_pair_index: torch.Tensor) -> LaneReachability:
        num_agents, num_steps = association.main_lane.shape
        num_lanes = lane_heading.numel()
        device = agent_velocity.device
        successor = _relation_by_current(successor_edge_index, num_lanes)
        lane_change = _directed_relation(lane_change_pair_index, num_lanes)
        drivable = lane_drivable.detach().cpu().bool().tolist()
        visible_lane = lane_visible_mask.detach().cpu().bool().tolist()
        lane_length_cpu = lane_length.detach().cpu().tolist()
        lane_start_cpu = lane_start_position.detach().cpu()
        lane_end_cpu = lane_end_position.detach().cpu()
        main_lane_cpu = association.main_lane.detach().cpu().tolist()
        projection_s_cpu = association.projection_s.detach().cpu().tolist()
        agent_visible_cpu = agent_visible_mask.detach().cpu().bool().tolist()

        distance_budget = agent_velocity.new_zeros((num_agents, num_steps))
        longitudinal_speed = agent_velocity.new_zeros((num_agents, num_steps))
        reachable_pairs: List[Tuple[int, int]] = []
        path_distances: List[float] = []
        lane_change_counts: List[int] = []

        for agent_idx in range(num_agents):
            for time_idx in range(num_steps):
                start_lane = int(main_lane_cpu[agent_idx][time_idx])
                if start_lane < 0 or not agent_visible_cpu[agent_idx][time_idx]:
                    continue
                direction = torch.stack([
                    torch.cos(lane_heading[start_lane]),
                    torch.sin(lane_heading[start_lane]),
                ])
                speed_long = torch.dot(
                    agent_velocity[agent_idx, time_idx], direction,
                ).clamp_min(0.0)
                acceleration_long_positive = torch.dot(
                    agent_acceleration[agent_idx, time_idx], direction,
                ).clamp_min(0.0)
                budget = (
                    3.0 * speed_long
                    + 0.5 * acceleration_long_positive * 9.0
                    + 5.0
                ).clamp(min=15.0, max=80.0)
                distance_budget[agent_idx, time_idx] = budget
                longitudinal_speed[agent_idx, time_idx] = speed_long
                budget_value = float(budget.item())

                best_state: Dict[Tuple[int, int], float] = {(start_lane, 0): 0.0}
                frontier = [(0.0, start_lane, 0)]
                best_lane: Dict[int, Tuple[float, int]] = {start_lane: (0.0, 0)}
                while frontier:
                    distance_now, lane_now, lane_changes_now = heapq.heappop(frontier)
                    if distance_now > best_state.get((lane_now, lane_changes_now), float('inf')):
                        continue
                    if distance_now > budget_value:
                        continue

                    for next_lane in successor[lane_now]:
                        if not drivable[next_lane] or not visible_lane[next_lane]:
                            continue
                        gap = torch.linalg.vector_norm(
                            lane_end_cpu[lane_now] - lane_start_cpu[next_lane]
                        ).item()
                        if lane_now == start_lane and distance_now == 0.0:
                            step_distance = max(
                                lane_length_cpu[lane_now]
                                - projection_s_cpu[agent_idx][time_idx],
                                0.0,
                            ) + gap
                        else:
                            step_distance = lane_length_cpu[lane_now] + gap
                        next_distance = distance_now + step_distance
                        next_state = (next_lane, lane_changes_now)
                        if (
                            next_distance <= budget_value
                            and next_distance < best_state.get(next_state, float('inf'))
                        ):
                            best_state[next_state] = next_distance
                            heapq.heappush(
                                frontier,
                                (next_distance, next_lane, lane_changes_now),
                            )
                            if next_distance < best_lane.get(next_lane, (float('inf'), 0))[0]:
                                best_lane[next_lane] = (next_distance, lane_changes_now)

                    if lane_changes_now >= self.max_lane_changes:
                        continue
                    for next_lane in lane_change[lane_now]:
                        if not drivable[next_lane] or not visible_lane[next_lane]:
                            continue
                        next_distance = distance_now + self.lane_change_distance
                        next_state = (next_lane, lane_changes_now + 1)
                        if (
                            next_distance <= budget_value
                            and next_distance < best_state.get(next_state, float('inf'))
                        ):
                            best_state[next_state] = next_distance
                            heapq.heappush(
                                frontier,
                                (next_distance, next_lane, lane_changes_now + 1),
                            )
                            if next_distance < best_lane.get(next_lane, (float('inf'), 0))[0]:
                                best_lane[next_lane] = (
                                    next_distance,
                                    lane_changes_now + 1,
                                )

                node_idx = time_idx * num_agents + agent_idx
                for lane_idx, (distance_value, lane_change_count) in sorted(best_lane.items()):
                    reachable_pairs.append((lane_idx, node_idx))
                    path_distances.append(distance_value)
                    lane_change_counts.append(lane_change_count)

        if reachable_pairs:
            edge_index = torch.tensor(
                reachable_pairs, dtype=torch.long, device=device,
            ).t().contiguous()
            path_distance = agent_velocity.new_tensor(path_distances)
            lane_change_count = torch.tensor(
                lane_change_counts, dtype=torch.long, device=device,
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            path_distance = agent_velocity.new_zeros(0)
            lane_change_count = torch.empty(0, dtype=torch.long, device=device)
        return LaneReachability(
            edge_index,
            path_distance,
            lane_change_count,
            distance_budget,
            longitudinal_speed,
        )

    def build_conflict_instances(
            self,
            reachability: LaneReachability,
            potential_conflict_mask: torch.Tensor,
            agent_batch: torch.Tensor,
            num_historical_steps: int,
            agent_valid_mask: torch.Tensor = None) -> ConflictInstances:
        num_agents = agent_batch.numel()
        device = agent_batch.device
        dtype = reachability.path_distance.dtype
        final_offset = (num_historical_steps - 1) * num_agents
        reachable_by_agent: List[Dict[int, float]] = [dict() for _ in range(num_agents)]
        for edge_idx in range(reachability.edge_index.size(1)):
            lane_idx = int(reachability.edge_index[0, edge_idx].item())
            node_idx = int(reachability.edge_index[1, edge_idx].item())
            if final_offset <= node_idx < final_offset + num_agents:
                reachable_by_agent[node_idx - final_offset][lane_idx] = float(
                    reachability.path_distance[edge_idx].item()
                )

        pair_list: List[Tuple[int, int]] = []
        conflict_lane_list: List[int] = []
        distance_i_list: List[float] = []
        distance_j_list: List[float] = []
        tau_i_list: List[float] = []
        tau_j_list: List[float] = []
        urgency_list: List[float] = []
        valid_list: List[bool] = []
        batch_cpu = agent_batch.detach().cpu().tolist()
        if agent_valid_mask is None:
            valid_agent_cpu = [True] * num_agents
        else:
            if agent_valid_mask.shape != (num_agents,):
                raise ValueError('agent_valid_mask must have shape [num_agents]')
            valid_agent_cpu = agent_valid_mask.detach().cpu().bool().tolist()
        conflict_cpu = potential_conflict_mask.detach().cpu().bool().tolist()
        longitudinal_speed = reachability.longitudinal_speed[:, -1].detach().cpu().tolist()

        for agent_i in range(num_agents):
            if not valid_agent_cpu[agent_i]:
                continue
            for agent_j in range(agent_i + 1, num_agents):
                if not valid_agent_cpu[agent_j]:
                    continue
                if batch_cpu[agent_i] != batch_cpu[agent_j]:
                    continue
                shared_lane = (
                    set(reachable_by_agent[agent_i])
                    & set(reachable_by_agent[agent_j])
                )
                for conflict_lane in sorted(shared_lane):
                    if not conflict_cpu[conflict_lane]:
                        continue
                    distance_i = reachable_by_agent[agent_i][conflict_lane]
                    distance_j = reachable_by_agent[agent_j][conflict_lane]
                    tau_i = distance_i / max(longitudinal_speed[agent_i], 1.0)
                    tau_j = distance_j / max(longitudinal_speed[agent_j], 1.0)
                    urgency = (
                        abs(tau_i - tau_j) / 2.0
                        + 0.5 * min(tau_i, tau_j) / 5.0
                    )
                    pair_list.append((agent_i, agent_j))
                    conflict_lane_list.append(conflict_lane)
                    distance_i_list.append(distance_i)
                    distance_j_list.append(distance_j)
                    tau_i_list.append(tau_i)
                    tau_j_list.append(tau_j)
                    urgency_list.append(urgency)
                    valid_list.append(
                        abs(tau_i - tau_j) < self.conflict_arrival_threshold
                    )

        def tensor(values, tensor_dtype=None):
            return torch.tensor(
                values,
                dtype=tensor_dtype or dtype,
                device=device,
            )

        if pair_list:
            pair_index = tensor(pair_list, torch.long).t().contiguous()
            conflict_lane = tensor(conflict_lane_list, torch.long)
            distance_i = tensor(distance_i_list)
            distance_j = tensor(distance_j_list)
            tau_raw_i = tensor(tau_i_list)
            tau_raw_j = tensor(tau_j_list)
            urgency_cost = tensor(urgency_list)
            valid_mask = tensor(valid_list, torch.bool)
        else:
            pair_index = torch.empty((2, 0), dtype=torch.long, device=device)
            conflict_lane = torch.empty(0, dtype=torch.long, device=device)
            distance_i = torch.empty(0, dtype=dtype, device=device)
            distance_j = torch.empty(0, dtype=dtype, device=device)
            tau_raw_i = torch.empty(0, dtype=dtype, device=device)
            tau_raw_j = torch.empty(0, dtype=dtype, device=device)
            urgency_cost = torch.empty(0, dtype=dtype, device=device)
            valid_mask = torch.empty(0, dtype=torch.bool, device=device)
        tau_clip_i = tau_raw_i.clamp(max=self.tau_max)
        tau_clip_j = tau_raw_j.clamp(max=self.tau_max)

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        canonical_pairs: List[Tuple[int, int]] = []
        canonical_attrs: List[torch.Tensor] = []
        pair_group = torch.full(
            (pair_index.size(1),), -1, dtype=torch.long, device=device,
        )
        if valid_indices.numel() != 0:
            valid_pair_keys = torch.unique(
                pair_index[0, valid_indices] * max(num_agents, 1)
                + pair_index[1, valid_indices]
            )
            for group_idx, pair_key in enumerate(valid_pair_keys.tolist()):
                agent_i = pair_key // max(num_agents, 1)
                agent_j = pair_key % max(num_agents, 1)
                instance_mask = (
                    valid_mask
                    & (pair_index[0] == agent_i)
                    & (pair_index[1] == agent_j)
                )
                instance_index = torch.nonzero(
                    instance_mask, as_tuple=False,
                ).flatten()
                weight = torch.softmax(
                    -urgency_cost[instance_index] / self.softmin_temperature,
                    dim=0,
                )
                delta_tau = (tau_raw_i[instance_index] - tau_raw_j[instance_index]).abs()
                aggregate = torch.stack([
                    (weight * tau_clip_i[instance_index]).sum(),
                    (weight * tau_clip_j[instance_index]).sum(),
                    (weight * delta_tau.clamp(max=self.tau_max)).sum(),
                    (weight * distance_i[instance_index]).sum(),
                    (weight * distance_j[instance_index]).sum(),
                    (weight * urgency_cost[instance_index]).sum(),
                    urgency_cost.new_tensor(float(instance_index.numel())),
                ])
                canonical_pairs.append((agent_i, agent_j))
                canonical_attrs.append(aggregate)
                pair_group[instance_index] = group_idx

        if canonical_pairs:
            canonical_pair_index = tensor(
                canonical_pairs, torch.long,
            ).t().contiguous()
            canonical_edge_attr = torch.stack(canonical_attrs, dim=0)
            forward_attr = canonical_edge_attr
            reverse_attr = canonical_edge_attr[:, [1, 0, 2, 4, 3, 5, 6]]
            E_VV_conf = torch.cat([
                canonical_pair_index,
                canonical_pair_index.flip(0),
            ], dim=1)
            directed_edge_attr = torch.cat([forward_attr, reverse_attr], dim=0)
        else:
            canonical_pair_index = torch.empty((2, 0), dtype=torch.long, device=device)
            canonical_edge_attr = torch.empty((0, 7), dtype=dtype, device=device)
            E_VV_conf = torch.empty((2, 0), dtype=torch.long, device=device)
            directed_edge_attr = torch.empty((0, 7), dtype=dtype, device=device)

        return ConflictInstances(
            pair_index=pair_index,
            conflict_lane=conflict_lane,
            distance_i=distance_i,
            distance_j=distance_j,
            tau_raw_i=tau_raw_i,
            tau_raw_j=tau_raw_j,
            tau_clip_i=tau_clip_i,
            tau_clip_j=tau_clip_j,
            urgency_cost=urgency_cost,
            valid_mask=valid_mask,
            pair_group=pair_group,
            canonical_pair_index=canonical_pair_index,
            canonical_edge_attr=canonical_edge_attr,
            E_VV_conf=E_VV_conf,
            directed_edge_attr=directed_edge_attr,
        )

    @staticmethod
    def expand_conflict_edges_across_modes(
            conflict_instances: ConflictInstances,
            num_agents: int,
            num_modes: int,
            historical_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if conflict_instances.E_VV_conf.numel() == 0:
            return (
                conflict_instances.E_VV_conf,
                conflict_instances.directed_edge_attr,
            )
        source_agent, target_agent = conflict_instances.E_VV_conf
        mode = torch.arange(num_modes, device=source_agent.device).unsqueeze(0)
        base = historical_step * num_modes
        source = (
            (base + mode) * num_agents + source_agent.unsqueeze(1)
        ).reshape(-1)
        target = (
            (base + mode) * num_agents + target_agent.unsqueeze(1)
        ).reshape(-1)
        edge_index = torch.stack([source, target], dim=0)
        edge_attr = conflict_instances.directed_edge_attr.repeat_interleave(
            num_modes, dim=0,
        )
        return edge_index, edge_attr

    @staticmethod
    def expand_reachability_across_modes(
            reachability: LaneReachability,
            num_agents: int,
            num_historical_steps: int,
            num_modes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand ``lane -> [time,agent]`` reachability to PNet's target layout."""
        if reachability.edge_index.numel() == 0:
            return (
                reachability.edge_index,
                reachability.path_distance,
                reachability.lane_change_count,
            )
        lane_index, node_index = reachability.edge_index
        time_index = torch.div(node_index, num_agents, rounding_mode='floor')
        agent_index = node_index % num_agents
        mode = torch.arange(num_modes, device=node_index.device).unsqueeze(0)
        target = (
            (agent_index.unsqueeze(1) * num_historical_steps
             + time_index.unsqueeze(1))
            * num_modes
            + mode
        ).reshape(-1)
        source = lane_index.unsqueeze(1).expand(-1, num_modes).reshape(-1)
        return (
            torch.stack([source, target], dim=0),
            reachability.path_distance.repeat_interleave(num_modes),
            reachability.lane_change_count.repeat_interleave(num_modes),
        )
