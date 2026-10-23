from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from transforms import V22MapGeometryTransform


def make_sample() -> HeteroData:
    data = HeteroData()
    data['scenario_name'] = 'SCENE'
    data['lane'].num_nodes = 2
    data['lane']['id'] = torch.tensor([11.0, 22.0])
    data['lane']['position'] = torch.zeros((2, 2))
    data['lane']['heading'] = torch.zeros(2)
    data['lane']['length'] = torch.zeros(2)
    data['lane']['curvature'] = torch.zeros(2)
    data['lane']['turn_direction'] = torch.zeros(2, dtype=torch.uint8)
    data['lane']['start_position'] = torch.zeros((2, 2))
    data['lane']['end_position'] = torch.zeros((2, 2))
    data['lane']['centerline_spacing'] = torch.ones(2)
    data['polyline'].num_nodes = 1
    data['polyline']['position'] = torch.zeros((1, 2))
    data['polyline']['heading'] = torch.zeros(1)
    data['polyline']['length'] = torch.zeros(1)
    data['polyline']['side'] = torch.zeros(1, dtype=torch.uint8)
    data['polyline', 'lane']['polyline_to_lane_edge_index'] = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    return data


def write_sidecar(cache_dir: Path, lane_id=None) -> None:
    cache_dir.mkdir()
    torch.save({
        'schema_version': 1,
        'scenario_name': 'SCENE',
        'lane_id': (
            torch.tensor([11, 22], dtype=torch.long)
            if lane_id is None else lane_id
        ),
        'lane_position': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        'lane_heading': torch.tensor([0.1, 0.2]),
        'lane_length': torch.tensor([5.0, 6.0]),
        'lane_curvature': torch.tensor([0.01, 0.02]),
        'lane_turn_direction': torch.tensor([1, 2], dtype=torch.uint8),
        'lane_start_position': torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
        'lane_end_position': torch.tensor([[2.0, 2.0], [4.0, 4.0]]),
        'lane_centerline_spacing': torch.tensor([0.5, 0.75]),
        'polyline_position': torch.tensor([
            [0.5, 0.0], [1.5, 0.0], [2.5, 0.0]
        ]),
        'polyline_heading': torch.tensor([0.0, 0.1, 0.2]),
        'polyline_length': torch.tensor([1.0, 1.5, 2.0]),
        'polyline_side': torch.tensor([0, 1, 2], dtype=torch.uint8),
        'polyline_to_lane_edge_index': torch.tensor(
            [[0, 1, 2], [0, 0, 1]], dtype=torch.long
        ),
    }, cache_dir / 'SCENE.pt')


def test_v22_map_geometry_replaces_only_geometry(tmp_path):
    cache_dir = tmp_path / 'map_geometry'
    write_sidecar(cache_dir)
    data = make_sample()
    data['lane']['speed_limit'] = torch.tensor([0.3, 0.4])

    transformed = V22MapGeometryTransform(cache_dir)(data)

    assert torch.equal(
        transformed['lane']['position'],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert torch.equal(
        transformed['lane']['curvature'], torch.tensor([0.01, 0.02])
    )
    assert torch.equal(
        transformed['lane']['speed_limit'], torch.tensor([0.3, 0.4])
    )
    assert transformed['polyline'].num_nodes == 3
    assert transformed['polyline']['position'].shape == (3, 2)
    assert torch.equal(
        transformed['polyline', 'lane']['polyline_to_lane_edge_index'],
        torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
    )


def test_v22_map_geometry_fails_closed_on_lane_order_mismatch(tmp_path):
    cache_dir = tmp_path / 'map_geometry'
    write_sidecar(cache_dir, lane_id=torch.tensor([22, 11]))

    with pytest.raises(ValueError, match='lane order'):
        V22MapGeometryTransform(cache_dir)(make_sample())
