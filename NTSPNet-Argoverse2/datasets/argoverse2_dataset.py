"""AV2 forecasting -> PNet graphs. Raw data is never modified.

The portable reader implements the published AV2 parquet/JSON schema and works
in the existing Python 3.8 environment. ``reader='av2'`` uses the official API.
Both use the JSON centerline, so changing readers does not change map geometry.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from modules.lane_topology import build_potential_conflict_topology
from transforms import LaneAssociationTransform
from utils import update_agent_kinematics

SCHEMA_VERSION = 1
OBJECT_TYPES = ('vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus',
                'static', 'background', 'construction', 'riderless_bicycle', 'unknown')


def read_scene(parquet, map_path, reader='parquet'):
    raw_map = json.loads(map_path.read_text())
    if reader == 'parquet':
        return pd.read_parquet(parquet), raw_map
    try:
        from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
        from av2.map.map_api import ArgoverseStaticMap
        scenario = load_argoverse_scenario_parquet(parquet)
        static_map = ArgoverseStaticMap.from_json(map_path)
    except (ImportError, TypeError) as error:
        raise RuntimeError('Official av2 reader unavailable. Use a compatible Python >=3.9 '
                           'environment, or --reader parquet (same published schema).') from error
    rows = []
    for track in scenario.tracks:
        for state in track.object_states:
            rows.append(dict(track_id=track.track_id, object_type=track.object_type.value,
                             object_category=track.category.value, timestep=state.timestep,
                             observed=state.observed, position_x=state.position[0],
                             position_y=state.position[1], heading=state.heading,
                             velocity_x=state.velocity[0], velocity_y=state.velocity[1],
                             scenario_id=scenario.scenario_id, focal_track_id=scenario.focal_track_id,
                             city=scenario.city_name))
    # Read boundaries and semantic/topological fields through the official map API.
    for lane in static_map.vector_lane_segments.values():
        entry = raw_map['lane_segments'][str(lane.id)]
        for side in ('left', 'right'):
            boundary = getattr(lane, side + '_lane_boundary').xyz
            entry[side + '_lane_boundary'] = [dict(zip(('x', 'y', 'z'), p)) for p in boundary]
            entry[side + '_lane_mark_type'] = getattr(lane, side + '_mark_type').value
            entry[side + '_neighbor_id'] = getattr(lane, side + '_neighbor_id')
        entry.update(predecessors=lane.predecessors, successors=lane.successors,
                     is_intersection=lane.is_intersection, lane_type=lane.lane_type.value)
    return pd.DataFrame(rows), raw_map


def _points(values, origin):
    xyz = np.asarray([[p['x'], p['y'], p.get('z', 0.)] for p in values], dtype=np.float64)
    if xyz.ndim != 2 or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError('Invalid map polyline')
    xyz[:, :2] -= origin
    # Zero-length segments otherwise make width, heading and projection ambiguous.
    keep = np.r_[True, np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) > 1e-5]
    xyz = xyz[keep]
    if len(xyz) < 2:
        raise ValueError('Degenerate map polyline')
    return xyz


def _resample(xyz, count):
    arc = np.r_[0., np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).cumsum()]
    distances = np.linspace(0., arc[-1], count)
    return np.stack([np.interp(distances, arc, xyz[:, d]) for d in range(3)], axis=-1)


def _edges(values):
    return torch.tensor(sorted(set(values)), dtype=torch.long).t().contiguous() if values else torch.empty(2, 0, dtype=torch.long)


def build_map(raw, origin):
    entries = raw['lane_segments']
    lanes = sorted(entries.values() if isinstance(entries, dict) else entries, key=lambda x: x['id'])
    if not lanes:
        raise ValueError('Scenario has no lane segments')
    ids = {lane['id']: i for i, lane in enumerate(lanes)}
    positions, headings, lengths, widths, curvature, turns = [], [], [], [], [], []
    starts, ends, spacings, polygons, heights = [], [], [], [], []
    poly_position, poly_heading, poly_length, poly_side, poly_lane = [], [], [], [], []
    relations = {name: [] for name in ('predecessor', 'successor', 'left_neighbor',
                                      'right_neighbor', 'adjacent_left', 'adjacent_right', 'conflict')}
    for idx, lane in enumerate(lanes):
        left, right = (_points(lane[side + '_lane_boundary'], origin) for side in ('left', 'right'))
        # Freeze this policy in the cache version: use supplied centerline whenever present.
        center = (_points(lane['centerline'], origin) if lane.get('centerline') else
                  (_resample(left, 20) + _resample(right, 20)) / 2.)
        vectors = np.diff(center[:, :2], axis=0)
        segment_lengths = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        delta = (angles[-1] - angles[0] + np.pi) % (2 * np.pi) - np.pi
        positions.append((center[0, :2] + center[-1, :2]) / 2.)
        headings.append(float(np.arctan2(*(center[-1, :2] - center[0, :2])[::-1])))
        lengths.append(segment_lengths.sum()); spacings.append(segment_lengths.mean())
        widths.append(np.linalg.norm(_resample(left, 20)[:, :2] - _resample(right, 20)[:, :2], axis=1).mean())
        curvature.append(np.abs((np.diff(angles) + np.pi) % (2*np.pi) - np.pi).mean() if len(angles) > 1 else 0.)
        turns.append(3 if abs(delta) > 2.5 else 1 if delta > .3 else 2 if delta < -.3 else 0)
        starts.append(center[0, :2]); ends.append(center[-1, :2])
        polygon = Polygon(np.concatenate((left[:, :2], right[::-1, :2])))
        polygons.append(polygon if polygon.is_valid else polygon.buffer(0))
        heights.append(float(np.median(np.r_[left[:, 2], right[:, 2]])))
        # Keep the source model's side encoding: left=0, center=1, right=2.
        for side, line in ((0, left), (1, center), (2, right)):
            vec = np.diff(line[:, :2], axis=0)
            poly_position.extend((line[1:, :2] + line[:-1, :2]) / 2.)
            poly_length.extend(np.linalg.norm(vec, axis=1))
            poly_heading.extend(np.arctan2(vec[:, 1], vec[:, 0]))
            poly_side.extend([side] * len(vec)); poly_lane.extend([idx] * len(vec))
        for direction in ('predecessor', 'successor'):
            relations[direction].extend((ids[j], idx) for j in lane[direction + 's'] if j in ids)
        for side in ('left', 'right'):
            neighbor = lane.get(side + '_neighbor_id')
            if neighbor not in ids:
                continue
            relations['adjacent_' + side].append((ids[neighbor], idx))
            # Conservative proxy, NOT a legal routing rule: plain dashed white only.
            # Solid/mixed/yellow/unknown boundaries never become lane-change edges.
            if lane[side + '_lane_mark_type'] == 'DASHED_WHITE':
                relations[side + '_neighbor'].append((ids[neighbor], idx))
    drivable = torch.tensor([p['lane_type'] in ('VEHICLE', 'BUS') for p in lanes])
    # AV2 has no Lanelet2 conflict graph. Infer geometric candidates, excluding
    # graph-connected/adjacent lanes, nearly parallel lanes and grade separation.
    connected = {tuple(sorted(e)) for key, es in relations.items() if key != 'conflict' for e in es}
    for i in range(len(lanes)):
        if not drivable[i]:
            continue
        for j in range(i + 1, len(lanes)):
            if (i, j) in connected or not drivable[j] or abs(heights[i] - heights[j]) > 2.:
                continue
            if abs(np.sin(headings[i] - headings[j])) < np.sin(np.pi / 12):
                continue
            if polygons[i].intersection(polygons[j]).area > .25:
                relations['conflict'].extend(((i, j), (j, i)))
    data = HeteroData()
    lane = data['lane']; lane.num_nodes = len(lanes)
    lane.id = torch.tensor([p['id'] for p in lanes], dtype=torch.long)
    for name, values in dict(position=positions, heading=headings, length=lengths, width=widths,
                             curvature=curvature, start_position=starts, end_position=ends,
                             centerline_spacing=spacings).items():
        lane[name] = torch.as_tensor(np.asarray(values), dtype=torch.float32)
    lane.turn_direction = torch.tensor(turns)
    lane.drivable = drivable
    lane.visible_mask = torch.ones(len(lanes), dtype=torch.bool)
    lane.has_traffic_sign = torch.zeros(len(lanes))  # unavailable, not inferred right of way
    lane.speed_limit = torch.zeros(len(lanes))
    lane.is_intersection = torch.tensor([p['is_intersection'] for p in lanes])
    lane.lane_type = torch.tensor([('VEHICLE', 'BIKE', 'BUS').index(p['lane_type']) for p in lanes])
    poly = data['polyline']; poly.num_nodes = len(poly_lane)
    for name, values in dict(position=poly_position, heading=poly_heading, length=poly_length).items():
        poly[name] = torch.as_tensor(np.asarray(values), dtype=torch.float32)
    poly.side = torch.tensor(poly_side)
    data['polyline', 'lane'].polyline_to_lane_edge_index = torch.stack((torch.arange(len(poly_lane)), torch.tensor(poly_lane)))
    edge_store = data['lane', 'lane']
    for name, values in relations.items():
        edge_store[name + '_edge_index'] = _edges(values)
    topology = build_potential_conflict_topology(
        lane.heading, lane.drivable, edge_store.predecessor_edge_index, edge_store.successor_edge_index,
        edge_store.left_neighbor_edge_index, edge_store.right_neighbor_edge_index,
        edge_store.adjacent_left_edge_index, edge_store.adjacent_right_edge_index, edge_store.conflict_edge_index)
    for name, value in topology.items():
        (edge_store if name.endswith('index') else lane)[name] = value
    return data


def build_scene(frame, raw_map, split, scenario_id, association=True):
    frame = frame.copy(); frame['track_id'] = frame.track_id.astype(str)
    if frame.duplicated(['track_id', 'timestep']).any():
        raise ValueError('Duplicate track/timestep rows')
    if not frame.timestep.between(0, 109).all():
        raise ValueError('AV2 timesteps must be in [0,109]')
    if not (frame.observed == (frame.timestep < 50)).all():
        raise ValueError('Inconsistent observed flags')
    if split == 'test' and (frame.timestep >= 50).any():
        raise ValueError('Test input must contain observations only; keep test annotations separate')
    historical = frame[frame.observed]
    track_ids = sorted(historical.track_id.unique())
    focal_id = str(frame.focal_track_id.iloc[0])
    focal_last = historical[(historical.track_id == focal_id) & (historical.timestep == 49)]
    if len(focal_last) != 1:
        raise ValueError('Focal track must be visible at timestep 49')
    origin = focal_last[['position_x', 'position_y']].to_numpy()[0]
    data = build_map(raw_map, origin)
    data.scenario_id = scenario_id; data.city = str(frame.city.iloc[0])
    data.track_ids = track_ids; data.origin = torch.tensor(origin, dtype=torch.float64).reshape(1, 2)
    agent = data['agent']; n = len(track_ids); agent.num_nodes = n
    agent.position = torch.zeros(n, 110, 2)
    agent.visible_mask = torch.zeros(n, 110, dtype=torch.bool)
    agent.csv_velocity = torch.zeros(n, 110, 2)
    agent.csv_heading = torch.zeros(n, 110)
    agent.csv_motion_valid_mask = torch.zeros(n, 110, dtype=torch.bool)
    agent.type = torch.zeros(n, dtype=torch.long)
    agent.object_type = torch.zeros(n, dtype=torch.long)
    agent.av2_category = torch.zeros(n, dtype=torch.long)
    agent.length = torch.zeros(n); agent.width = torch.zeros(n)  # not supplied by AV2 MF
    agent.size_valid = torch.zeros(n, dtype=torch.bool)
    agent.focal_mask = torch.tensor([i == focal_id for i in track_ids])
    groups = {key: group.sort_values('timestep') for key, group in frame.groupby('track_id')}
    for i, key in enumerate(track_ids):
        group = groups[key]; steps = torch.tensor(group.timestep.to_numpy(), dtype=torch.long)
        values = group[['position_x', 'position_y', 'heading', 'velocity_x', 'velocity_y']].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError('Non-finite trajectory values')
        agent.position[i, steps] = torch.tensor(values[:, :2] - origin, dtype=torch.float32)
        agent.visible_mask[i, steps] = True
        # Future motion is not input to either trunk. Only positions/masks store supervision.
        observed = steps < 50; hsteps = steps[observed]
        agent.csv_velocity[i, hsteps] = torch.tensor(values[observed, 3:], dtype=torch.float32)
        agent.csv_heading[i, hsteps] = torch.tensor(values[observed, 2], dtype=torch.float32)
        agent.csv_motion_valid_mask[i, hsteps] = True
        kind = str(group.object_type.iloc[0]); kind_idx = OBJECT_TYPES.index(kind)
        agent.object_type[i] = kind_idx
        agent.type[i] = 0 if kind in ('vehicle', 'bus') else kind_idx + 1
        agent.av2_category[i] = int(group.object_category.iloc[0])
    agent.dynamic_mask = agent.object_type < 5
    agent.scored_mask = agent.av2_category >= 2
    agent.category = agent.scored_mask.long()  # legacy module-internal binary semantics
    agent.train_mask = agent.dynamic_mask & (agent.av2_category >= 1)
    update_agent_kinematics(agent, num_historical_steps=50, step_time=.1)
    if association:
        data = LaneAssociationTransform(num_historical_steps=50)(data)
    return data


class Argoverse2Dataset(Dataset):
    def __init__(self, root, split, cache_dir=None, manifest=None, limit=None,
                 reader='parquet', association=True):
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        if reader not in ('parquet', 'av2'):
            raise ValueError('reader must be parquet or av2')
        self.root = Path(root).resolve(); self.split = split
        self.reader = reader; self.association = association
        if manifest:
            self.ids = [x.strip() for x in Path(manifest).read_text().splitlines() if x.strip()]
            if any('/' in x or '\\' in x or x in ('.', '..') for x in self.ids):
                raise ValueError('Manifest must contain scene IDs, not paths')
            if len(set(self.ids)) != len(self.ids):
                raise ValueError('Duplicate scene IDs in manifest')
        else:
            self.ids = sorted(p.name for p in (self.root / split).iterdir() if p.is_dir())
        if limit is not None:
            if limit <= 0:
                raise ValueError('limit must be positive')
            self.ids = self.ids[:limit]
        if not self.ids:
            raise ValueError('No scenarios selected')
        namespace = hashlib.sha256(str((str(self.root), SCHEMA_VERSION, reader, association)).encode()).hexdigest()[:16]
        self.cache_dir = Path(cache_dir).resolve() / ('av2_v%d_' % SCHEMA_VERSION + namespace) / split if cache_dir else None
        if self.cache_dir:
            if self.root == self.cache_dir or self.root in self.cache_dir.parents:
                raise ValueError('Keep processed cache outside the raw dataset root')
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]; directory = self.root / self.split / name
        parquet = directory / ('scenario_' + name + '.parquet')
        map_path = directory / ('log_map_archive_' + name + '.json')
        signatures = [(p.stat().st_size, p.stat().st_mtime_ns) for p in (parquet, map_path)]
        target = self.cache_dir / (name + '.pt') if self.cache_dir else None
        if target and target.exists():
            cached = torch.load(target, map_location='cpu', weights_only=False)
            if cached['signature'] == signatures:
                return cached['data']
        frame, raw_map = read_scene(parquet, map_path, self.reader)
        data = build_scene(frame, raw_map, self.split, name, self.association)
        if target:
            temporary = target.with_suffix('.%d.tmp' % os.getpid())
            torch.save(dict(signature=signatures, data=data), temporary)
            os.replace(temporary, target)
        return data
