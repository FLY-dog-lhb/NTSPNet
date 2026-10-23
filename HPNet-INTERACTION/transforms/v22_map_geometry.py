from pathlib import Path

import torch
from torch_geometric.transforms import BaseTransform


class V22MapGeometryTransform(BaseTransform):
    """Replace only in-memory map geometry with an unresampled sidecar."""

    SCHEMA_VERSION = 1

    def __init__(self, cache_dir) -> None:
        super(V22MapGeometryTransform, self).__init__()
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(
                f'v22 map geometry cache does not exist: {self.cache_dir}'
            )

    @staticmethod
    def _scenario_name(data) -> str:
        scenario_name = data['scenario_name']
        if isinstance(scenario_name, (list, tuple)):
            if len(scenario_name) != 1:
                raise ValueError('map geometry transform expects one scenario')
            scenario_name = scenario_name[0]
        return str(scenario_name)

    def __call__(self, data):
        scenario_name = self._scenario_name(data)
        sidecar_path = self.cache_dir / f'{scenario_name}.pt'
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                f'v22 map geometry sidecar is missing: {sidecar_path}'
            )
        sidecar = torch.load(sidecar_path, map_location='cpu')
        if int(sidecar.get('schema_version', -1)) != self.SCHEMA_VERSION:
            raise ValueError(
                f'unsupported v22 map geometry schema: {sidecar_path}'
            )
        if str(sidecar.get('scenario_name')) != scenario_name:
            raise ValueError(
                f'v22 map geometry scenario mismatch: {sidecar_path}'
            )

        processed_lane_id = data['lane']['id'].to(torch.long).cpu()
        sidecar_lane_id = sidecar['lane_id'].to(torch.long).cpu()
        if not torch.equal(processed_lane_id, sidecar_lane_id):
            raise ValueError(
                'v22 map geometry lane order does not match processed data: '
                f'{sidecar_path}'
            )

        lane_fields = {
            'position': 'lane_position',
            'heading': 'lane_heading',
            'length': 'lane_length',
            'curvature': 'lane_curvature',
            'turn_direction': 'lane_turn_direction',
            'start_position': 'lane_start_position',
            'end_position': 'lane_end_position',
            'centerline_spacing': 'lane_centerline_spacing',
        }
        for target_name, source_name in lane_fields.items():
            value = sidecar[source_name]
            if value.size(0) != data['lane'].num_nodes:
                raise ValueError(
                    f'{source_name} lane count mismatch in {sidecar_path}'
                )
            data['lane'][target_name] = value.clone()

        polyline_fields = {
            'position': 'polyline_position',
            'heading': 'polyline_heading',
            'length': 'polyline_length',
            'side': 'polyline_side',
        }
        polyline_count = int(sidecar['polyline_position'].size(0))
        for target_name, source_name in polyline_fields.items():
            value = sidecar[source_name]
            if value.size(0) != polyline_count:
                raise ValueError(
                    f'{source_name} polyline count mismatch in {sidecar_path}'
                )
            data['polyline'][target_name] = value.clone()
        data['polyline'].num_nodes = polyline_count

        edge_index = sidecar['polyline_to_lane_edge_index'].to(torch.long)
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f'invalid polyline-to-lane edge index in {sidecar_path}'
            )
        if edge_index.size(1) != polyline_count:
            raise ValueError(
                f'polyline-to-lane edge count mismatch in {sidecar_path}'
            )
        data['polyline', 'lane']['polyline_to_lane_edge_index'] = (
            edge_index.clone()
        )
        return data
