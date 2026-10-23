from argparse import ArgumentParser
import os
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


SPLIT_DIRECTORIES = {
    'train': 'train',
    'val': 'val',
    'test': 'test_multi-agent',
}


def scenario_name_from_raw_path(raw_path: Path) -> str:
    parts = raw_path.stem.split('_')
    return '_'.join(parts[:-1])


def load_manifest(path: Optional[str]) -> Optional[Set[str]]:
    if path is None:
        return None
    names = {
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    }
    if not names:
        raise ValueError(f'manifest contains no processed filenames: {path}')
    return names


def build_case_motion(case_df: pd.DataFrame,
                      num_historical_steps: int,
                      num_future_steps: int) -> dict:
    num_steps = num_historical_steps + num_future_steps
    timestep_ids = list(np.sort(case_df['timestamp_ms'].unique()))
    if len(timestep_ids) > num_steps:
        raise ValueError(
            f'case has {len(timestep_ids)} timesteps, expected at most {num_steps}'
        )
    historical_timestamps = timestep_ids[:num_historical_steps]
    historical_df = case_df[
        case_df['timestamp_ms'].isin(historical_timestamps)
    ]
    agent_ids = list(historical_df['track_id'].unique())
    case_df = case_df[case_df['track_id'].isin(agent_ids)]
    timestep_to_index = {
        timestamp: index for index, timestamp in enumerate(timestep_ids)
    }

    num_agents = len(agent_ids)
    agent_id = torch.as_tensor(agent_ids, dtype=torch.long)
    csv_velocity = torch.zeros(num_agents, num_steps, 2, dtype=torch.float)
    csv_heading = torch.zeros(num_agents, num_steps, dtype=torch.float)
    csv_motion_valid_mask = torch.zeros(
        num_agents, num_steps, dtype=torch.bool,
    )

    agent_to_index = {
        track_id: index for index, track_id in enumerate(agent_ids)
    }
    for track_id, track_df in case_df.groupby('track_id', sort=False):
        agent_index = agent_to_index[track_id]
        steps = torch.as_tensor([
            timestep_to_index[timestamp]
            for timestamp in track_df['timestamp_ms']
        ], dtype=torch.long)
        velocity = torch.from_numpy(np.stack([
            track_df['vx'].to_numpy(),
            track_df['vy'].to_numpy(),
        ], axis=-1)).float()
        velocity_heading = torch.atan2(velocity[:, 1], velocity[:, 0])
        if track_df['agent_type'].iloc[0] == 'car':
            heading = torch.from_numpy(
                track_df['psi_rad'].to_numpy(),
            ).float()
        else:
            heading = velocity_heading
        csv_velocity[agent_index, steps] = velocity
        csv_heading[agent_index, steps] = heading
        csv_motion_valid_mask[agent_index, steps] = True

    return {
        'schema_version': 1,
        'agent_id': agent_id,
        'csv_velocity': csv_velocity,
        'csv_heading': csv_heading,
        'csv_motion_valid_mask': csv_motion_valid_mask,
    }


def generate_split(root: Path,
                   output_root: Path,
                   split: str,
                   num_historical_steps: int,
                   num_future_steps: int,
                   manifest: Optional[Set[str]],
                   overwrite: bool) -> None:
    raw_dir = root / SPLIT_DIRECTORIES[split]
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_names = set()

    for raw_path in tqdm(sorted(raw_dir.glob('*.csv')), desc=split):
        scenario_name = scenario_name_from_raw_path(raw_path)
        frame = pd.read_csv(raw_path)
        required_columns = {
            'case_id', 'track_id', 'timestamp_ms', 'agent_type',
            'vx', 'vy', 'psi_rad',
        }
        missing_columns = sorted(required_columns.difference(frame.columns))
        if missing_columns:
            raise ValueError(
                f'{raw_path} is missing columns: {missing_columns}'
            )
        for case_id, case_df in tqdm(
                frame.groupby('case_id', sort=False),
                desc=scenario_name,
                leave=False):
            processed_name = f'{scenario_name}_{int(case_id)}.pt'
            if manifest is not None and processed_name not in manifest:
                continue
            generated_names.add(processed_name)
            output_path = output_dir / processed_name
            if output_path.exists() and not overwrite:
                continue
            payload = build_case_motion(
                case_df=case_df,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
            )
            payload.update({
                'scenario_name': scenario_name,
                'case_id': int(case_id),
                'num_historical_steps': num_historical_steps,
                'num_future_steps': num_future_steps,
            })
            temporary_path = output_path.with_suffix('.pt.tmp')
            torch.save(payload, temporary_path)
            os.replace(temporary_path, output_path)

    if manifest is not None:
        missing_names = sorted(manifest.difference(generated_names))
        if missing_names:
            preview = ', '.join(missing_names[:5])
            raise ValueError(
                f'{len(missing_names)} manifest entries were not found in '
                f'{split} raw CSV files; first entries: {preview}'
            )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Defaults to <root>/csv_motion_v1.',
    )
    parser.add_argument(
        '--split',
        choices=tuple(SPLIT_DIRECTORIES),
        nargs='+',
        default=['train', 'val'],
    )
    parser.add_argument('--num_historical_steps', type=int, default=10)
    parser.add_argument('--num_future_steps', type=int, default=30)
    parser.add_argument(
        '--manifest',
        type=str,
        default=None,
        help='Optional text file containing stable processed .pt filenames.',
    )
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.num_historical_steps <= 0 or args.num_future_steps <= 0:
        parser.error('historical and future step counts must be positive')
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.root / 'csv_motion_v1'
    )
    manifest = load_manifest(args.manifest)
    for selected_split in args.split:
        generate_split(
            root=args.root,
            output_root=output_dir,
            split=selected_split,
            num_historical_steps=args.num_historical_steps,
            num_future_steps=args.num_future_steps,
            manifest=manifest,
            overwrite=args.overwrite,
        )
