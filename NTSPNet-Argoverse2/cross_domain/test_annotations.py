"""Read released AV2 Multi-World test labels separately from model inputs."""
from pathlib import Path

import numpy as np
import pandas as pd

from cross_domain.runtime import sha256


class MultiWorldTestAnnotations:
    def __init__(self, path, scenario_ids):
        self.path = Path(path).resolve()
        columns = ['scenario_id', 'track_id', 'is_focal_track', 'gt_trajectory_x', 'gt_trajectory_y']
        rows = pd.read_parquet(self.path, columns=columns,
                               filters=[('scenario_id', 'in', list(scenario_ids))])
        if rows[columns[:3]].isna().any().any():
            raise ValueError('Missing annotation identifiers or focal flag')
        rows['scenario_id'] = rows.scenario_id.astype(str)
        rows['track_id'] = rows.track_id.astype(str)
        if rows.duplicated(['scenario_id', 'track_id']).any():
            raise ValueError('Duplicate test annotation scenario/track')
        if set(rows.scenario_id) != set(scenario_ids):
            raise ValueError('Test annotations do not cover the selected scenarios')
        self.scenes = {key: group for key, group in rows.groupby('scenario_id', sort=False)}
        self.provenance = dict(path=str(self.path), sha256=sha256(self.path),
                               bytes=self.path.stat().st_size,
                               format='AV2 released Multi-World test annotations',
                               selected_annotation_rows=len(rows),
                               full_annotation_horizon=60, evaluated_annotation_indices=[0, 29],
                               mapped_av2_timesteps=[50, 79])

    def future_frame(self, scenario_id, observations):
        """Verify IDs against observations; return ONLY labels for steps 50..79.

        Target type/category, context membership, and all features continue to
        come exclusively from the original observed parquet, never this file.
        """
        if (not observations.timestep.between(0, 49).all() or
                not observations.observed.all()):
            raise ValueError('Test model inputs must contain observations only (0..49)')
        if set(observations.scenario_id.astype(str)) != {scenario_id}:
            raise ValueError('Test scenario ID does not match observations')
        rows = self.scenes[scenario_id]
        scored = set(observations.loc[observations.object_category >= 2, 'track_id'].astype(str))
        if scored != set(rows.track_id):
            raise ValueError('Test scored track IDs differ from Multi-World annotations; '
                             'do not use single-agent annotations')
        focal_ids = set(observations.focal_track_id.astype(str))
        annotation_focal = set(rows.loc[rows.is_focal_track == True, 'track_id'])
        if len(focal_ids) != 1 or annotation_focal != focal_ids:
            raise ValueError('Test focal track mismatch')
        records = []
        for row in rows.itertuples(index=False):
            x = np.asarray(row.gt_trajectory_x, dtype=np.float64)
            y = np.asarray(row.gt_trajectory_y, dtype=np.float64)
            if x.shape != (60,) or y.shape != (60,) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Expected 60 finite future positions per test target')
            records.append(pd.DataFrame(dict(track_id=row.track_id, timestep=np.arange(50, 80),
                                              position_x=x[:30], position_y=y[:30])))
        return pd.concat(records, ignore_index=True)
