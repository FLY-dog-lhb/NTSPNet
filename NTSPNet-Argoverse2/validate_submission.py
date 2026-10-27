"""Independently check exported AV2 predictions against raw observation targets."""
from argparse import ArgumentParser
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import Argoverse2Dataset
from test import sha256


def validate(submission, root, protocol, modes=6, manifest=None, limit=None):
    dataset = Argoverse2Dataset(root, 'test', manifest=manifest, limit=limit, association=False)
    expected = set()
    for scene in tqdm(dataset.ids, desc='Check raw targets'):
        path = Path(root) / 'test' / scene / ('scenario_' + scene + '.parquet')
        frame = pd.read_parquet(path, columns=['track_id', 'focal_track_id', 'object_category', 'timestep'])
        final = frame[frame.timestep == 49]
        ids = ([str(frame.focal_track_id.iloc[0])] if protocol == 'single-agent'
               else final.loc[final.object_category >= 2, 'track_id'].astype(str).unique().tolist())
        if not ids or not set(ids) <= set(final.track_id.astype(str)):
            raise ValueError('Invalid raw scored targets for ' + scene)
        expected.update((scene, key) for key in ids)
    source = pq.ParquetFile(submission)
    required = ['scenario_id', 'track_id', 'probability', 'predicted_trajectory_x', 'predicted_trajectory_y']
    if source.schema_arrow.names != required:
        raise ValueError('Unexpected submission columns/order')
    probabilities = defaultdict(list); row_count = 0
    for batch in source.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            key = row['scenario_id'], row['track_id']
            if key not in expected:
                raise ValueError('Unexpected scene/track: ' + str(key))
            xy = np.asarray([row['predicted_trajectory_x'], row['predicted_trajectory_y']], dtype=float)
            p = row['probability']
            if xy.shape != (2, 60) or not np.isfinite(xy).all():
                raise ValueError('Invalid prediction horizon/coordinates for ' + str(key))
            if p is None or not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Invalid probability for ' + str(key))
            probabilities[key].append(p); row_count += 1
    if set(probabilities) != expected:
        raise ValueError('Missing prediction targets: %d' % len(expected - set(probabilities)))
    scene_probability = {}
    for key, values in probabilities.items():
        if len(values) != modes or not np.isclose(sum(values), 1., atol=1e-5):
            raise ValueError('Invalid number of modes or probability sum for ' + str(key))
        if protocol == 'multi-world':
            if key[0] in scene_probability and not np.allclose(values, scene_probability[key[0]]):
                raise ValueError('World probabilities/order differ between actors')
            scene_probability[key[0]] = values
    return dict(valid=True, scenarios=len(dataset), targets=len(expected), rows=row_count,
                modes=modes, horizon=60, protocol=protocol, sha256=sha256(submission))


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--submission', required=True)
    p.add_argument('--root', default='/home/lhb/data-Argoverse2')
    p.add_argument('--protocol', choices=('single-agent', 'multi-world'), default='single-agent')
    p.add_argument('--modes', type=int, default=6)
    p.add_argument('--test-manifest'); p.add_argument('--test-limit', type=int)
    args = p.parse_args()
    print(json.dumps(validate(args.submission, args.root, args.protocol, args.modes,
                              args.test_manifest, args.test_limit), indent=2))


if __name__ == '__main__':
    main()
