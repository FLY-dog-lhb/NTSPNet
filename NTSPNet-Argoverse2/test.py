"""Stream AV2 ChallengeSubmission-compatible parquet; blind test has no metrics."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from cli import add_data_args, datamodule
from model import PNet

SUBMISSION_SCHEMA = pa.schema([
    ('scenario_id', pa.string()), ('track_id', pa.string()), ('probability', pa.float64()),
    ('predicted_trajectory_x', pa.list_(pa.float64())),
    ('predicted_trajectory_y', pa.list_(pa.float64())),
])


def prediction_rows(scenes):
    rows = []
    for scene in scenes:
        p = scene['probability'].numpy(); trajectory = scene['trajectory'].numpy()
        if trajectory.shape != (len(scene['track_ids']), len(p), 60, 2):
            raise ValueError('Invalid prediction shape')
        if not np.isfinite(trajectory).all() or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
            raise ValueError('Non-finite predictions or invalid probabilities')
        # All actors use one shared, consistently sorted mode order.
        order = np.argsort(-p, kind='stable')
        for track, trajectories in zip(scene['track_ids'], trajectory):
            for k in order:
                rows.append(dict(scenario_id=scene['scenario_id'], track_id=track,
                                 probability=float(p[k]), predicted_trajectory_x=trajectories[k, :, 0].tolist(),
                                 predicted_trajectory_y=trajectories[k, :, 1].tolist()))
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True, help='New directory; existing exports are never overwritten')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    dm = datamodule(args)
    output_dir = Path(args.output_dir).resolve()
    raw_root = Path(args.root).resolve()
    if output_dir == raw_root or raw_root in output_dir.parents:
        parser.error('Keep exports outside the raw dataset directory')
    if output_dir.exists():
        parser.error('Output directory already exists; use a new directory')
    device = args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PNet.restore(args.checkpoint).to(device).eval()
    dm.setup('predict')
    output_dir.mkdir(parents=True)
    temporary = output_dir / 'submission.parquet.partial'
    cases = targets = row_count = 0
    with pq.ParquetWriter(temporary, SUBMISSION_SCHEMA, compression='zstd') as writer, torch.inference_mode():
        for batch_idx, data in enumerate(tqdm(dm.predict_dataloader(), desc='Export')):
            scenes = model.predict_step(data.to(device), batch_idx)
            rows = prediction_rows(scenes)
            writer.write_table(pa.Table.from_pylist(rows, schema=SUBMISSION_SCHEMA))
            cases += len(scenes); targets += sum(len(s['track_ids']) for s in scenes); row_count += len(rows)
    destination = output_dir / 'submission.parquet'
    temporary.rename(destination)
    full_ids = {p.name for p in (raw_root / 'test').iterdir() if p.is_dir()}
    manifest = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha256(args.checkpoint),
                    dataset_root=str(raw_root), dataset_schema=model.hparams.dataset_schema,
                    reader=args.reader, protocol=model.hparams.protocol, cases=cases, targets=targets,
                    rows=row_count, modes=model.hparams.num_modes, horizon=60,
                    complete_split=set(dm.test_dataset.ids) == full_ids,
                    selected_ids_sha256=hashlib.sha256('\n'.join(dm.test_dataset.ids).encode()).hexdigest(),
                    submission_sha256=sha256(destination), local_test_metrics=None)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
