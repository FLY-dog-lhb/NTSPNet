from typing import Dict, Iterable, List, Set, Tuple

import torch

from utils import wrap_angle


def _incoming_relation(edge_index: torch.Tensor, num_lanes: int) -> List[Set[int]]:
    """Convert ``[related_lane, current_lane]`` edges into per-lane sets."""
    relation = [set() for _ in range(num_lanes)]
    if edge_index.numel() == 0:
        return relation
    for related_lane, current_lane in edge_index.detach().cpu().t().tolist():
        if 0 <= related_lane < num_lanes and 0 <= current_lane < num_lanes:
            relation[current_lane].add(related_lane)
    return relation


def _angle_difference(
        lane_heading: torch.Tensor,
        first_lane: int,
        second_lane: int) -> float:
    difference = wrap_angle(
        lane_heading[first_lane] - lane_heading[second_lane]
    ).abs()
    return float(difference.item())


def _all_pairs(values: Iterable[int]) -> Iterable[Tuple[int, int]]:
    value_list = list(values)
    for first_idx in range(len(value_list)):
        for second_idx in range(first_idx + 1, len(value_list)):
            yield value_list[first_idx], value_list[second_idx]


def build_potential_conflict_topology(
        lane_heading: torch.Tensor,
        lane_drivable: torch.Tensor,
        predecessor_edge_index: torch.Tensor,
        successor_edge_index: torch.Tensor,
        left_change_edge_index: torch.Tensor,
        right_change_edge_index: torch.Tensor,
        adjacent_left_edge_index: torch.Tensor,
        adjacent_right_edge_index: torch.Tensor,
        crossing_edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Build lanelet-level conflict regions and directed lane-change pairs.

    All regular lane relations use the repository convention
    ``[related_lane, current_lane]``. ``lane_change_pair_index`` instead stores
    the mathematical directed pair ``[u, v]``: a vehicle may change from
    lanelet ``u`` to lanelet ``v``.
    """
    num_lanes = lane_heading.numel()
    device = lane_heading.device
    if lane_drivable.shape != (num_lanes,):
        raise ValueError('lane_drivable must have shape [num_lanes]')

    predecessor = _incoming_relation(predecessor_edge_index, num_lanes)
    successor = _incoming_relation(successor_edge_index, num_lanes)
    left_change = _incoming_relation(left_change_edge_index, num_lanes)
    right_change = _incoming_relation(right_change_edge_index, num_lanes)
    adjacent_left = _incoming_relation(adjacent_left_edge_index, num_lanes)
    adjacent_right = _incoming_relation(adjacent_right_edge_index, num_lanes)

    adjacent_all: List[Set[int]] = []
    changeable: List[Set[int]] = []
    for lane_idx in range(num_lanes):
        changeable_now = left_change[lane_idx] | right_change[lane_idx]
        changeable.append(changeable_now)
        adjacent_all.append(
            changeable_now
            | adjacent_left[lane_idx]
            | adjacent_right[lane_idx]
        )

    drivable = lane_drivable.detach().cpu().bool().tolist()
    merge_core: Set[int] = set()
    split_core: Set[int] = set()

    for current_lane in range(num_lanes):
        if not drivable[current_lane]:
            continue

        valid_predecessors = {
            lane for lane in predecessor[current_lane] if drivable[lane]
        }
        if len(valid_predecessors) >= 2:
            aligned_with_current = all(
                _angle_difference(lane_heading, lane, current_lane)
                < torch.pi / 4
                for lane in valid_predecessors
            )
            mutually_aligned = all(
                _angle_difference(lane_heading, first, second)
                < torch.pi / 3
                for first, second in _all_pairs(valid_predecessors)
            )
            if aligned_with_current and mutually_aligned:
                merge_core.add(current_lane)

        valid_successors = {
            lane for lane in successor[current_lane] if drivable[lane]
        }
        if len(valid_successors) >= 2:
            related_to_current = all(
                _angle_difference(lane_heading, lane, current_lane)
                < torch.pi / 3
                for lane in valid_successors
            )
            separates = any(
                _angle_difference(lane_heading, first, second)
                > torch.pi / 12
                for first, second in _all_pairs(valid_successors)
            )
            if related_to_current and separates:
                split_core.add(current_lane)

    merge_region = set(merge_core)
    for core_lane in merge_core:
        merge_region.update(
            lane for lane in predecessor[core_lane] if drivable[lane]
        )

    split_region = set(split_core)
    for core_lane in split_core:
        split_region.update(
            lane for lane in successor[core_lane] if drivable[lane]
        )

    merge_or_split_neighborhood = set(merge_core) | set(split_core)
    for core_lane in merge_core | split_core:
        merge_or_split_neighborhood.update(
            lane for lane in predecessor[core_lane] if drivable[lane]
        )
        merge_or_split_neighborhood.update(
            lane for lane in successor[core_lane] if drivable[lane]
        )

    lane_change_pairs: List[Tuple[int, int]] = []
    lane_change_critical: Set[int] = set()
    for source_lane in range(num_lanes):
        if not drivable[source_lane]:
            continue
        for target_lane in sorted(changeable[source_lane]):
            if not drivable[target_lane]:
                continue
            if _angle_difference(lane_heading, source_lane, target_lane) >= torch.pi / 6:
                continue

            predecessor_adjacent_union: Set[int] = set()
            for source_predecessor in predecessor[source_lane]:
                predecessor_adjacent_union.update(adjacent_all[source_predecessor])
            adjacency_starts = (
                len(predecessor[target_lane] & predecessor_adjacent_union) == 0
            )

            successor_adjacent_union: Set[int] = set()
            for source_successor in successor[source_lane]:
                successor_adjacent_union.update(adjacent_all[source_successor])
            adjacency_ends = (
                len(successor[target_lane] & successor_adjacent_union) == 0
            )

            near_merge_or_split = (
                source_lane in merge_or_split_neighborhood
                or target_lane in merge_or_split_neighborhood
            )
            if adjacency_starts or adjacency_ends or near_merge_or_split:
                lane_change_pairs.append((source_lane, target_lane))
                lane_change_critical.update((source_lane, target_lane))

    crossing_conflict: Set[int] = set()
    if crossing_edge_index.numel() != 0:
        for first_lane, second_lane in crossing_edge_index.detach().cpu().t().tolist():
            if 0 <= first_lane < num_lanes and drivable[first_lane]:
                crossing_conflict.add(first_lane)
            if 0 <= second_lane < num_lanes and drivable[second_lane]:
                crossing_conflict.add(second_lane)

    potential_conflict = (
        crossing_conflict
        | merge_region
        | split_region
        | lane_change_critical
    )

    def to_mask(indices: Set[int]) -> torch.Tensor:
        mask = torch.zeros(num_lanes, dtype=torch.bool, device=device)
        if indices:
            mask[torch.tensor(sorted(indices), dtype=torch.long, device=device)] = True
        return mask

    if lane_change_pairs:
        lane_change_pair_index = torch.tensor(
            lane_change_pairs,
            dtype=torch.long,
            device=device,
        ).t().contiguous()
    else:
        lane_change_pair_index = torch.empty((2, 0), dtype=torch.long, device=device)

    return {
        'crossing_conflict_mask': to_mask(crossing_conflict),
        'merge_core_mask': to_mask(merge_core),
        'merge_region_mask': to_mask(merge_region),
        'split_core_mask': to_mask(split_core),
        'split_region_mask': to_mask(split_region),
        'lane_change_critical_mask': to_mask(lane_change_critical),
        'potential_conflict_mask': to_mask(potential_conflict),
        'lane_change_pair_index': lane_change_pair_index,
    }
