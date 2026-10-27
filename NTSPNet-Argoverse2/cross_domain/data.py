"""History-only AV2 -> INTERACTION H10/F30 input, with labels kept separate."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

# Do not import `datasets`: the source INTERACTION package requires Lanelet2.
_spec = importlib.util.spec_from_file_location(
    '_cross_domain_av2_reader', Path(__file__).resolve().parents[1] / 'datasets/argoverse2_dataset.py')
av2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(av2)

PROTOCOL = 'interaction-zero-shot-av2-h10-f30-v1'
DYNAMIC_TYPES = ('vehicle', 'bus', 'pedestrian', 'cyclist', 'motorcyclist')


def build_input(frame, raw_map, scenario_id, future_frame=None):
    """Return graph, labels. The graph contains no values from timesteps >=50.

    Only AV2 40..49 can determine nodes/features. 50..79 are labels outside the
    graph. AV2 0..39 and 80..109 are ignored. Dimensions unavailable in AV2 are
    zero; no statistics or parameters are estimated from the target dataset.
    Optional future_frame contains labels only, and never enters graph building.
    """
    frame = frame.copy()
    frame['track_id'] = frame.track_id.astype(str)
    if frame.duplicated(['track_id', 'timestep']).any():
        raise ValueError('Duplicate track/timestep rows')
    if not (frame.observed == (frame.timestep < 50)).all():
        raise ValueError('Inconsistent observation flags')
    history = frame[frame.timestep.between(40, 49)]
    focal = str(history.focal_track_id.iloc[0])
    last = history[(history.track_id == focal) & (history.timestep == 49)]
    if len(last) != 1:
        raise ValueError('Missing focal observation at timestep 49')
    origin = last[['position_x', 'position_y']].to_numpy(dtype=np.float64)[0]
    history = history[history.object_type.isin(DYNAMIC_TYPES)]
    ids = sorted(history.track_id.unique())
    if not ids:
        raise ValueError('No supported dynamic context agents')
    data = av2.build_map(raw_map, origin)
    # Match the source dataset's lane reference point and local tangent instead
    # of the AV2 training adapter's endpoint midpoint/chord approximation.
    entries = raw_map['lane_segments']
    lanes = sorted(entries.values() if isinstance(entries, dict) else entries, key=lambda x: x['id'])
    for i, lane in enumerate(lanes):
        if lane.get('centerline'):
            center = av2._points(lane['centerline'], origin)
        else:
            center = sum(av2._resample(av2._points(lane[s + '_lane_boundary'], origin), 20)
                         for s in ('left', 'right')) / 2.
        mid = (len(center) - 1) // 2
        data['lane'].position[i] = torch.tensor(center[mid, :2], dtype=torch.float32)
        delta = center[mid + 1, :2] - center[mid, :2]
        data['lane'].heading[i] = float(np.arctan2(delta[1], delta[0]))
    lane, edges = data['lane'], data['lane', 'lane']
    topology = av2.build_potential_conflict_topology(
        lane.heading, lane.drivable, edges.predecessor_edge_index, edges.successor_edge_index,
        edges.left_neighbor_edge_index, edges.right_neighbor_edge_index,
        edges.adjacent_left_edge_index, edges.adjacent_right_edge_index, edges.conflict_edge_index)
    for key, value in topology.items():
        (edges if key.endswith('index') else lane)[key] = value
    data.scenario_id = scenario_id
    data.case_id = torch.tensor([0])
    data.scenario_name = str(history.city.iloc[0])
    agent = data['agent']; agent.num_nodes = len(ids)
    agent.id = torch.arange(len(ids))  # local graph index; original strings stay in labels
    agent.position = torch.zeros(len(ids), 40, 2)
    agent.visible_mask = torch.zeros(len(ids), 40, dtype=torch.bool)
    agent.csv_velocity = torch.zeros(len(ids), 40, 2)
    agent.csv_heading = torch.zeros(len(ids), 40)
    agent.csv_motion_valid_mask = torch.zeros(len(ids), 40, dtype=torch.bool)
    agent.length = torch.zeros(len(ids)); agent.width = torch.zeros(len(ids))
    agent.type = torch.zeros(len(ids), dtype=torch.long)
    agent.category = torch.zeros(len(ids), dtype=torch.long)
    agent.interest = torch.zeros(len(ids), dtype=torch.long)
    vehicle = torch.zeros(len(ids), dtype=torch.bool)
    scored = torch.zeros_like(vehicle)
    for i, key in enumerate(ids):
        rows = history[history.track_id == key].sort_values('timestep')
        steps = torch.tensor(rows.timestep.to_numpy() - 40, dtype=torch.long)
        values = rows[['position_x', 'position_y', 'heading', 'velocity_x', 'velocity_y']].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite observed motion')
        kind = str(rows.object_type.iloc[0])
        is_car = kind in ('vehicle', 'bus')
        agent.type[i] = 0 if is_car else 1
        vehicle[i] = kind == 'vehicle'
        scored[i] = int(rows.object_category.iloc[0]) >= 2
        agent.category[i] = int(vehicle[i] and scored[i])
        agent.position[i, steps] = torch.tensor(values[:, :2] - origin, dtype=torch.float32)
        agent.visible_mask[i, steps] = True
        agent.csv_velocity[i, steps] = torch.tensor(values[:, 3:], dtype=torch.float32)
        heading = values[:, 2] if is_car else np.arctan2(values[:, 4], values[:, 3])
        agent.csv_heading[i, steps] = torch.tensor(heading, dtype=torch.float32)
        agent.csv_motion_valid_mask[i, steps] = True
    av2.update_agent_kinematics(agent, num_historical_steps=10, step_time=.1)
    data = av2.LaneAssociationTransform(num_historical_steps=10)(data)

    target = torch.zeros(len(ids), 30, 2)
    valid = torch.zeros(len(ids), 30, dtype=torch.bool)
    future_source = frame if future_frame is None else future_frame
    if future_frame is not None:
        future_source = future_frame.copy()
        future_source['track_id'] = future_source.track_id.astype(str)
        if (not future_source.timestep.between(50, 109).all() or
                future_source.duplicated(['track_id', 'timestep']).any()):
            raise ValueError('External labels must have unique future track/timestep rows')
    future = future_source[future_source.timestep.between(50, 79)]
    for i, key in enumerate(ids):
        rows = future[future.track_id == key]
        steps = torch.tensor(rows.timestep.to_numpy() - 50, dtype=torch.long)
        values = rows[['position_x', 'position_y']].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite target positions')
        target[i, steps] = torch.tensor(values - origin, dtype=torch.float32)
        valid[i, steps] = True
    # Source validation requires complete history AND future; apply to labels only.
    eligible = vehicle & scored & agent.visible_mask[:, :10].all(1) & valid.all(1)
    labels = dict(position=target, valid=valid, eligible=eligible,
                  focal=torch.tensor([key == focal for key in ids]), track_ids=ids,
                  scored_vehicles=int((vehicle & scored).sum()),
                  scored_nonvehicles=int((~vehicle & scored).sum()))
    return data, labels
