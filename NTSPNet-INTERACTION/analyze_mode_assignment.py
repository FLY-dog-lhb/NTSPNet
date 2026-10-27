from argparse import ArgumentParser
from itertools import combinations
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

from analyze_endpoint_errors import conflict_stats
from datamodules.INTERACTION_datamodule import subset_from_manifest
from datasets import INTERACTIONDataset
from model import PNet
from transforms import LaneAssociationTransform
from transforms import LaneRandomOcclusion
from utils import generate_target


def compute_assignment_metrics(
        prediction: torch.Tensor,
        target: torch.Tensor) -> Dict[str, float]:
    """Measure endpoint error under increasingly flexible mode assignment.

    ``prediction`` has shape [N, K, F, 2]. The ordinary joint oracle forces
    every agent in the scene to use the same mode index. The other oracles are
    diagnostics only; they are not metrics used during validation.
    """
    if prediction.ndim != 4 or target.ndim != 3:
        raise ValueError('prediction/target must have rank 4/3')
    if prediction.size(0) == 0:
        raise ValueError('a scored scenario must contain at least one agent')
    if prediction.size(0) != target.size(0):
        raise ValueError('prediction/target agent counts differ')
    if prediction.shape[2:] != target.shape[1:]:
        raise ValueError('prediction/target horizon shapes differ')

    endpoint_error = torch.linalg.vector_norm(
        prediction[:, :, -1] - target[:, -1].unsqueeze(1), dim=-1,
    )  # [N,K]
    num_agents, num_modes = endpoint_error.shape
    per_agent_error, per_agent_mode = endpoint_error.min(dim=1)
    mode_mean_error = endpoint_error.mean(dim=0)
    scene_best_mode = int(mode_mean_error.argmin().item())
    joint_fde = mode_mean_error[scene_best_mode]
    independent_fde = per_agent_error.mean()

    mode_counts = torch.bincount(per_agent_mode, minlength=num_modes)
    disagrees = per_agent_mode != scene_best_mode
    result = {
        'joint_fde': float(joint_fde.item()),
        'independent_fde': float(independent_fde.item()),
        'coordination_gap': float((joint_fde - independent_fde).item()),
        'scene_best_mode': scene_best_mode,
        'unique_agent_best_modes': int((mode_counts > 0).sum().item()),
        'largest_agent_best_mode_share': float(
            mode_counts.max().float().div(num_agents).item()
        ),
        'agent_best_disagrees_scene_count': int(disagrees.sum().item()),
        'agent_best_disagrees_scene_rate': float(
            disagrees.float().mean().item()
        ),
    }

    # Keep one common scene mode, but let at most M agents use their own best
    # mode. This identifies whether a small number of agents dominate the gap.
    gain_from_exception = endpoint_error - per_agent_error.unsqueeze(1)
    for exception_count in (1, 2):
        count = min(exception_count, num_agents)
        saved_error = gain_from_exception.topk(
            count, dim=0, largest=True,
        ).values.sum(dim=0)
        adjusted = (
            endpoint_error.sum(dim=0) - saved_error
        ) / num_agents
        result[f'allow_{exception_count}_exception_fde'] = float(
            adjusted.min().item()
        )

    # Let each agent choose from a scene-level subset of S modes. This is an
    # upper-bound diagnostic for a compact combinatorial joint representation.
    for subset_size in (2, 3):
        size = min(subset_size, num_modes)
        subset_fde = min(
            endpoint_error[:, list(mode_subset)].min(dim=1).values.mean()
            for mode_subset in combinations(range(num_modes), size)
        )
        result[f'best_{subset_size}_mode_subset_fde'] = float(
            subset_fde.item()
        )
    return result


def summarize(frame: pd.DataFrame, prefix: str) -> dict:
    names = [
        'joint_fde',
        'independent_fde',
        'coordination_gap',
        'allow_1_exception_fde',
        'allow_2_exception_fde',
        'best_2_mode_subset_fde',
        'best_3_mode_subset_fde',
        'unique_agent_best_modes',
        'largest_agent_best_mode_share',
        'agent_best_disagrees_scene_rate',
    ]
    means = {
        name: float(frame[f'{prefix}_{name}'].mean())
        for name in names
    }
    joint = means['joint_fde']
    independent = means['independent_fde']
    gap = joint - independent
    recovery = {}
    for name in (
        'allow_1_exception_fde',
        'allow_2_exception_fde',
        'best_2_mode_subset_fde',
        'best_3_mode_subset_fde',
    ):
        recovery[name] = (
            float((joint - means[name]) / gap) if gap > 0 else 0.0
        )
    disagree_count = frame[
        f'{prefix}_agent_best_disagrees_scene_count'
    ].sum()
    return {
        'means': means,
        'gap_recovery_fraction': recovery,
        'pooled_agent_best_disagrees_scene_rate': float(
            disagree_count / frame['scored_agent_count'].sum()
        ),
        'scenes_requiring_multiple_agent_best_modes_rate': float(
            (frame[f'{prefix}_unique_agent_best_modes'] > 1).mean()
        ),
        'scenes_requiring_three_plus_agent_best_modes_rate': float(
            (frame[f'{prefix}_unique_agent_best_modes'] > 2).mean()
        ),
    }


def add_group(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['agent_count_group'] = pd.cut(
        result['scored_agent_count'],
        bins=[0, 1, 4, 8, 15, float('inf')],
        labels=['1', '2to4', '5to8', '9to15', '16plus'],
    ).astype(str)
    return result


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument(
        '--trunk_variant', choices=('extended', 'original'), default=None,
        help='Explicit compatibility override for checkpoints predating this option.',
    )
    parser.add_argument(
        '--motion_source', choices=('causal_difference', 'csv'), default=None,
        help='Explicit compatibility override for checkpoints predating this option.',
    )
    parser.add_argument('--csv_motion_cache_dir', type=Path, default=None)
    args = parser.parse_args()

    for path in (args.root, args.manifest, args.checkpoint):
        if not path.exists():
            parser.error(f'path does not exist: {path}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f'output directory is not empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else ('cpu' if args.device == 'auto' else args.device)
    )

    checkpoint_overrides = {
        key: value
        for key, value in {
            'trunk_variant': args.trunk_variant,
            'motion_source': args.motion_source,
            'csv_motion_cache_dir': (
                str(args.csv_motion_cache_dir.resolve())
                if args.csv_motion_cache_dir is not None else None
            ),
        }.items()
        if value is not None
    }
    model = PNet.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint.resolve()),
        map_location='cpu',
        strict=True,
        **checkpoint_overrides,
    ).eval().to(device)
    dataset = INTERACTIONDataset(
        root=str(args.root.resolve()),
        split='val',
        transform=Compose([
            LaneRandomOcclusion(0.0),
            LaneAssociationTransform(
                model.num_historical_steps,
                model.observation_step_time,
            ),
        ]),
        num_historical_steps=model.num_historical_steps,
        num_future_steps=model.num_future_steps,
        observation_step_time=model.observation_step_time,
        motion_source=model.motion_source,
        csv_motion_cache_dir=model.csv_motion_cache_dir,
    )
    dataset = subset_from_manifest(dataset, str(args.manifest.resolve()))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
    )

    rows = []
    with torch.inference_mode():
        for data in tqdm(loader, desc='mode-assignment'):
            data = data.to(device)
            proposal, refinement, _ = model(data)
            target, _ = generate_target(
                position=data['agent']['position'],
                mask=data['agent']['visible_mask'],
                num_historical_steps=model.num_historical_steps,
                num_future_steps=model.num_future_steps,
            )
            predicted_agent = data['agent']['category'] == 1
            visible = data['agent']['visible_mask'][predicted_agent]
            scored = visible.sum(dim=-1) == (
                model.num_historical_steps + model.num_future_steps
            )
            scored_target = target[predicted_agent, -1][scored]
            scored_batch = data['agent']['batch'][predicted_agent][scored]
            predictions = {
                'proposal': proposal[predicted_agent, -1][scored],
                'refinement': refinement[predicted_agent, -1][scored],
            }
            for batch_index, scenario_name in enumerate(data['scenario_name']):
                scene_mask = scored_batch == batch_index
                if not scene_mask.any():
                    raise RuntimeError(f'no scored agent: {scenario_name}')
                case_id = int(data['case_id'][batch_index].item())
                row = {
                    'sample_id': f'{scenario_name}_{case_id}.pt',
                    'scenario_name': str(scenario_name),
                    'case_id': case_id,
                    'scored_agent_count': int(scene_mask.sum().item()),
                    **conflict_stats(data, batch_index),
                }
                for name, prediction in predictions.items():
                    metrics = compute_assignment_metrics(
                        prediction[scene_mask], scored_target[scene_mask],
                    )
                    row.update({f'{name}_{key}': value for key, value in metrics.items()})
                rows.append(row)

    frame = add_group(pd.DataFrame(rows).sort_values('sample_id'))
    frame.to_csv(args.output_dir / 'per_scene.csv', index=False)
    summary = {
        'schema_version': 1,
        'checkpoint': str(args.checkpoint.resolve()),
        'manifest': str(args.manifest.resolve()),
        'device': str(device),
        'trunk_variant': model.trunk_variant,
        'motion_source': model.motion_source,
        'scenario_count': int(len(frame)),
        'overall': {
            name: summarize(frame, name)
            for name in ('proposal', 'refinement')
        },
        'by_agent_count': {
            group: {
                'scenario_count': int(len(group_frame)),
                **{
                    name: summarize(group_frame, name)
                    for name in ('proposal', 'refinement')
                },
            }
            for group, group_frame in frame.groupby('agent_count_group', observed=True)
        },
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
