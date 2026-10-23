from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
import torch

import lanelet2
from lanelet2.projection import UtmProjector

from utils import compute_angles_lengths_2D
from utils import wrap_angle


SCHEMA_VERSION = 1
POLYLINE_SIDE = {'left': 0, 'center': 1, 'right': 2}


def _points(line_string) -> torch.Tensor:
    values = [np.asarray([point.x, point.y]) for point in line_string]
    result = torch.from_numpy(np.asarray(values)).float()
    if result.ndim != 2 or result.size(0) < 2 or result.size(1) != 2:
        raise ValueError('lanelet geometry must contain at least two 2-D points')
    return result


def build_map_geometry(map_path: Path) -> dict:
    projector = UtmProjector(lanelet2.io.Origin(0, 0))
    map_api = lanelet2.io.load(str(map_path), projector)
    lanes = list(map_api.laneletLayer)
    if not lanes:
        raise ValueError(f'map contains no lanelets: {map_path}')

    lane_id = torch.as_tensor([lane.id for lane in lanes], dtype=torch.long)
    num_lanes = len(lanes)
    lane_position = torch.zeros((num_lanes, 2), dtype=torch.float)
    lane_heading = torch.zeros(num_lanes, dtype=torch.float)
    lane_length = torch.zeros(num_lanes, dtype=torch.float)
    lane_curvature = torch.zeros(num_lanes, dtype=torch.float)
    lane_turn_direction = torch.zeros(num_lanes, dtype=torch.uint8)
    lane_start_position = torch.zeros((num_lanes, 2), dtype=torch.float)
    lane_end_position = torch.zeros((num_lanes, 2), dtype=torch.float)
    lane_centerline_spacing = torch.zeros(num_lanes, dtype=torch.float)

    polyline_position = []
    polyline_heading = []
    polyline_length = []
    polyline_side = []
    polyline_lane = []

    for lane_index, lane in enumerate(lanes):
        centerline = _points(lane.centerline)
        left_boundary = _points(lane.leftBound)
        right_boundary = _points(lane.rightBound)
        center_vectors = centerline[1:] - centerline[:-1]
        center_lengths = torch.linalg.vector_norm(center_vectors, dim=-1)
        center_index = (centerline.size(0) - 1) // 2

        lane_position[lane_index] = centerline[center_index]
        lane_heading[lane_index] = torch.atan2(
            center_vectors[center_index, 1], center_vectors[center_index, 0]
        )
        lane_length[lane_index] = center_lengths.sum()
        lane_start_position[lane_index] = centerline[0]
        lane_end_position[lane_index] = centerline[-1]
        lane_centerline_spacing[lane_index] = center_lengths.mean()

        if center_vectors.size(0) >= 2:
            vector_heading = torch.atan2(
                center_vectors[:, 1], center_vectors[:, 0]
            )
            lane_curvature[lane_index] = wrap_angle(
                vector_heading[1:] - vector_heading[:-1]
            ).abs().mean()
        start_heading = torch.atan2(center_vectors[0, 1], center_vectors[0, 0])
        end_heading = torch.atan2(center_vectors[-1, 1], center_vectors[-1, 0])
        heading_change = wrap_angle(end_heading - start_heading)
        if heading_change.abs() > 2.5:
            lane_turn_direction[lane_index] = 3
        elif heading_change > 0.3:
            lane_turn_direction[lane_index] = 1
        elif heading_change < -0.3:
            lane_turn_direction[lane_index] = 2

        parts = (
            ('left', left_boundary[1:] - left_boundary[:-1], left_boundary),
            ('right', right_boundary[1:] - right_boundary[:-1], right_boundary),
            ('center', center_vectors, centerline),
        )
        for side_name, vectors, points in parts:
            lengths, headings = compute_angles_lengths_2D(vectors)
            positions = (points[1:] + points[:-1]) / 2
            polyline_position.append(positions)
            polyline_heading.append(headings)
            polyline_length.append(lengths)
            polyline_side.append(torch.full(
                (vectors.size(0),), POLYLINE_SIDE[side_name], dtype=torch.uint8
            ))
            polyline_lane.append(torch.full(
                (vectors.size(0),), lane_index, dtype=torch.long
            ))

    polyline_position_tensor = torch.cat(polyline_position, dim=0)
    polyline_heading_tensor = torch.cat(polyline_heading, dim=0)
    polyline_length_tensor = torch.cat(polyline_length, dim=0)
    polyline_side_tensor = torch.cat(polyline_side, dim=0)
    polyline_lane_tensor = torch.cat(polyline_lane, dim=0)
    polyline_to_lane = torch.stack([
        torch.arange(polyline_lane_tensor.numel(), dtype=torch.long),
        polyline_lane_tensor,
    ], dim=0)

    return {
        'schema_version': SCHEMA_VERSION,
        'scenario_name': map_path.stem,
        'lane_id': lane_id,
        'lane_position': lane_position,
        'lane_heading': lane_heading,
        'lane_length': lane_length,
        'lane_curvature': lane_curvature,
        'lane_turn_direction': lane_turn_direction,
        'lane_start_position': lane_start_position,
        'lane_end_position': lane_end_position,
        'lane_centerline_spacing': lane_centerline_spacing,
        'polyline_position': polyline_position_tensor,
        'polyline_heading': polyline_heading_tensor,
        'polyline_length': polyline_length_tensor,
        'polyline_side': polyline_side_tensor,
        'polyline_to_lane_edge_index': polyline_to_lane,
    }


def write_map_geometry(map_path: Path, output_dir: Path, overwrite: bool) -> Path:
    output_path = output_dir / f'{map_path.stem}.pt'
    if output_path.exists() and not overwrite:
        return output_path
    payload = build_map_geometry(map_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix('.pt.tmp')
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    return output_path


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    map_paths = sorted((args.root / 'maps').glob('*.osm'))
    if not map_paths:
        parser.error(f'no .osm maps found under {args.root / "maps"}')
    outputs = [
        write_map_geometry(path, args.output_dir, args.overwrite)
        for path in map_paths
    ]
    print(f'generated_or_verified={len(outputs)} output_dir={args.output_dir}')


if __name__ == '__main__':
    main()
