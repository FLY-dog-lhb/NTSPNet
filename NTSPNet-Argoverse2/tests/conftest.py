import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)


@pytest.fixture
def map_data():
    lanes = {}
    for idx, (start, end, y) in enumerate(((-30., 10., 0.), (-30., 10., 3.5), (10., 80., 0.))):
        def line(offset):
            return [dict(x=float(x), y=y + offset, z=0.) for x in np.linspace(start, end, 8)]
        lanes[str(idx)] = dict(id=idx, centerline=line(0), left_lane_boundary=line(1.75),
                               right_lane_boundary=line(-1.75), left_lane_mark_type='DASHED_WHITE',
                               right_lane_mark_type='SOLID_WHITE', lane_type='VEHICLE', is_intersection=idx == 2,
                               left_neighbor_id=1 if idx == 0 else None,
                               right_neighbor_id=0 if idx == 1 else None,
                               predecessors=[0, 1] if idx == 2 else [], successors=[2] if idx < 2 else [])
    return dict(lane_segments=lanes, drivable_areas={}, pedestrian_crossings={})


@pytest.fixture
def frame():
    rows = []
    for actor, kind, category, y, start in (('00042', 'vehicle', 3, 0., 0),
                                           ('99', 'vehicle', 2, 3.5, 0),
                                           ('ped-1', 'pedestrian', 1, 9., 20),
                                           ('future-only', 'vehicle', 0, 1., 55)):
        for step in range(start, 110):
            if actor == 'ped-1' and step in (25, 26):
                continue
            rows.append(dict(track_id=actor, object_type=kind, object_category=category, timestep=step,
                             observed=step < 50, position_x=1000. + .5 * step,
                             position_y=2000. + y, heading=0., velocity_x=5., velocity_y=0.,
                             scenario_id='scene-a', focal_track_id='00042', city='austin'))
    return pd.DataFrame(rows)


@pytest.fixture
def local_frame(frame):
    frame = frame.copy(); frame.position_x -= 1000.; frame.position_y -= 2000.
    return frame


@pytest.fixture
def graph(local_frame, map_data):
    from datasets.argoverse2_dataset import build_scene
    return build_scene(local_frame, map_data, 'train', 'scene-a')
