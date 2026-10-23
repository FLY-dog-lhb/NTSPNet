from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

from datasets import INTERACTIONDataset
from model import PNet
from transforms import LaneAssociationTransform
from transforms import LaneRandomOcclusion
from transforms import V22MapGeometryTransform


PREDICTION_COLUMNS = [
    f'{coordinate}{mode}'
    for mode in range(1, 7)
    for coordinate in ('x', 'y', 'psi_rad')
]
SUBMISSION_COLUMNS = [
    'case_id',
    'track_id',
    'frame_id',
    'timestamp_ms',
    'track_to_predict',
    'interesting_agent',
    *PREDICTION_COLUMNS,
]


def parse_args():
    parser = ArgumentParser(
        description=(
            'Export one strict PNet checkpoint in the official INTERACTION '
            'multi-agent submission format without accessing future labels.'
        ),
    )
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--v22_map_geometry_cache_dir', type=Path, required=True)
    parser.add_argument('--original_checkpoint', action='store_true')
    parser.add_argument('--csv_motion_cache_dir', type=Path, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument(
        '--max_samples', type=int, default=0,
        help='Validation-only smoke limit. It is forbidden for split=test.',
    )
    return parser.parse_args()


def validate_args(args):
    for path_name in ('root', 'checkpoint', 'v22_map_geometry_cache_dir'):
        path = getattr(args, path_name)
        if not path.exists():
            raise FileNotFoundError(f'{path_name} does not exist: {path}')
    if args.output_dir.exists():
        raise FileExistsError(
            f'output_dir already exists; refusing to overwrite: '
            f'{args.output_dir}'
        )
    if args.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if args.num_workers < 0:
        raise ValueError('num_workers must be non-negative')
    if args.max_samples < 0:
        raise ValueError('max_samples must be non-negative')
    if args.split == 'test' and args.max_samples != 0:
        raise ValueError('split=test requires max_samples=0 for one complete run')
    if args.original_checkpoint and args.csv_motion_cache_dir is None:
        raise ValueError(
            '--csv_motion_cache_dir is required with --original_checkpoint'
        )
    if (
        args.csv_motion_cache_dir is not None
        and not args.csv_motion_cache_dir.is_dir()
    ):
        raise FileNotFoundError(
            f'CSV motion cache does not exist: {args.csv_motion_cache_dir}'
        )


def load_model(args, device):
    overrides = {}
    if args.original_checkpoint:
        overrides = {
            'trunk_variant': 'original',
            'motion_source': 'csv',
            'csv_motion_cache_dir': str(args.csv_motion_cache_dir.resolve()),
        }
    model = PNet.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint.resolve()),
        map_location='cpu',
        strict=True,
        **overrides,
    )
    expected_trunk = 'original' if args.original_checkpoint else 'original_residual'
    if model.trunk_variant != expected_trunk:
        raise ValueError(
            f'{args.label} loaded trunk_variant={model.trunk_variant!r}; '
            f'expected {expected_trunk!r}'
        )
    if model.motion_source != 'csv':
        raise ValueError(
            f'{args.label} loaded motion_source={model.motion_source!r}; '
            "expected 'csv'"
        )
    return model.eval().to(device)


def build_dataset(args, model):
    transform = Compose([
        V22MapGeometryTransform(args.v22_map_geometry_cache_dir.resolve()),
        LaneRandomOcclusion(0.0),
        LaneAssociationTransform(
            model.num_historical_steps,
            model.observation_step_time,
        ),
    ])
    dataset = INTERACTIONDataset(
        root=str(args.root.resolve()),
        split=args.split,
        transform=transform,
        num_historical_steps=model.num_historical_steps,
        num_future_steps=model.num_future_steps,
        observation_step_time=model.observation_step_time,
        motion_source=model.motion_source,
        csv_motion_cache_dir=model.csv_motion_cache_dir,
    )
    total_samples = len(dataset)
    if args.max_samples:
        dataset = torch.utils.data.Subset(
            dataset, range(min(args.max_samples, total_samples)),
        )
    return dataset, total_samples


def collect_expected_targets(data, expected_targets):
    target_mask = data['agent']['category'] == 1
    agent_batch = data['agent']['batch']
    for batch_index, scenario_name in enumerate(data['scenario_name']):
        scene_mask = target_mask & (agent_batch == batch_index)
        case_id = int(data['case_id'][batch_index].item())
        agent_ids = data['agent']['id'][scene_mask].to(torch.long).tolist()
        if not agent_ids:
            raise ValueError(
                f'no track_to_predict agents in {scenario_name}, case {case_id}'
            )
        for agent_id in agent_ids:
            key = (str(scenario_name), case_id, int(agent_id))
            if key in expected_targets:
                raise ValueError(f'duplicate test target encountered: {key}')
            expected_targets[key] = 30


def frames_from_model_output(model, expected_targets):
    frames = {}
    observed_targets = {}
    for scenario_name, rows in sorted(model.test_output.items()):
        frame = pd.DataFrame(rows, columns=model._columns)
        frame.insert(4, 'track_to_predict', 1)
        frame = frame[SUBMISSION_COLUMNS]
        if frame.empty:
            raise ValueError(f'empty prediction frame: {scenario_name}')
        if frame.duplicated(
            ['case_id', 'track_id', 'frame_id', 'timestamp_ms'],
        ).any():
            raise ValueError(f'duplicate prediction rows: {scenario_name}')
        if not torch.isfinite(torch.from_numpy(
            frame[PREDICTION_COLUMNS].to_numpy(dtype='float64'),
        )).all():
            raise ValueError(f'non-finite prediction values: {scenario_name}')
        if set(frame['track_to_predict'].unique()) != {1}:
            raise ValueError(f'invalid track_to_predict values: {scenario_name}')
        if not set(frame['interesting_agent'].unique()).issubset({0, 1}):
            raise ValueError(f'invalid interesting_agent values: {scenario_name}')

        for (case_id, track_id), group in frame.groupby(
            ['case_id', 'track_id'], sort=False,
        ):
            if len(group) != 30:
                raise ValueError(
                    f'{scenario_name}, case={case_id}, track={track_id} '
                    f'has {len(group)} rows instead of 30'
                )
            if set(group['frame_id']) != set(range(11, 41)):
                raise ValueError(
                    f'invalid future frame IDs for {scenario_name}, '
                    f'case={case_id}, track={track_id}'
                )
            expected_timestamps = {frame_id * 100 for frame_id in range(11, 41)}
            if set(group['timestamp_ms']) != expected_timestamps:
                raise ValueError(
                    f'invalid timestamps for {scenario_name}, '
                    f'case={case_id}, track={track_id}'
                )
            observed_targets[(
                str(scenario_name), int(case_id), int(track_id),
            )] = len(group)
        frames[str(scenario_name)] = frame

    if observed_targets != expected_targets:
        missing = set(expected_targets) - set(observed_targets)
        extra = set(observed_targets) - set(expected_targets)
        raise ValueError(
            'submission target mismatch: '
            f'missing={len(missing)}, extra={len(extra)}'
        )
    return frames


def write_submission(args, model, frames, total_samples, evaluated_samples):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    csv_paths = []
    for scenario_name, frame in frames.items():
        output_path = args.output_dir / f'{scenario_name}_sub.csv'
        frame.to_csv(output_path, index=False)
        csv_paths.append(output_path)

    zip_path = args.output_dir / 'submission.zip'
    with zipfile.ZipFile(zip_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for csv_path in sorted(csv_paths):
            archive.write(csv_path, arcname=csv_path.name)

    manifest = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'label': args.label,
        'split': args.split,
        'checkpoint': str(args.checkpoint.resolve()),
        'trunk_variant': model.trunk_variant,
        'motion_source': model.motion_source,
        'v22_map_geometry_cache_dir': str(
            args.v22_map_geometry_cache_dir.resolve()
        ),
        'total_dataset_samples': int(total_samples),
        'evaluated_samples': int(evaluated_samples),
        'scenario_file_count': len(csv_paths),
        'prediction_row_count': int(sum(len(frame) for frame in frames.values())),
        'submission_zip': str(zip_path.resolve()),
        'format': 'INTERPRET multi-agent X_sub.csv, six modalities',
    }
    manifest_path = args.output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def main():
    args = parse_args()
    validate_args(args)
    pl.seed_everything(1024, workers=True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    device = torch.device(args.device)
    model = load_model(args, device)
    dataset, total_samples = build_dataset(args, model)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
    )

    model.test_output.clear()
    expected_targets = {}
    with torch.inference_mode():
        for batch_index, data in enumerate(tqdm(loader, desc=args.label)):
            collect_expected_targets(data, expected_targets)
            model.test_step(data.to(device), batch_index)

    frames = frames_from_model_output(model, expected_targets)
    manifest = write_submission(
        args=args,
        model=model,
        frames=frames,
        total_samples=total_samples,
        evaluated_samples=len(dataset),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
