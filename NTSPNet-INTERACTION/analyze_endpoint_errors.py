from argparse import ArgumentParser
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

from datamodules.INTERACTION_datamodule import subset_from_manifest
from datasets import INTERACTIONDataset
from model import PNet
from transforms import LaneAssociationTransform
from transforms import LaneRandomOcclusion
from transforms import V22MapGeometryTransform
from utils import generate_target


METRIC_COLUMNS = ('joint_ade', 'joint_fde', 'joint_mr')


def parse_checkpoint_specs(specs: Iterable[str]) -> List[Tuple[str, Path]]:
    parsed = []
    seen_labels = set()
    for spec in specs:
        if '=' not in spec:
            raise ValueError(
                f"checkpoint must use LABEL=/absolute/path.ckpt: {spec}"
            )
        label, path_text = spec.split('=', 1)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', label):
            raise ValueError(f'invalid checkpoint label: {label}')
        if label in seen_labels:
            raise ValueError(f'duplicate checkpoint label: {label}')
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'checkpoint does not exist: {path}')
        parsed.append((label, path))
        seen_labels.add(label)
    return parsed


def checkpoint_config(model: PNet, checkpoint_path: Path) -> dict:
    return {
        'checkpoint': str(checkpoint_path),
        'trunk_variant': model.trunk_variant,
        'motion_source': model.motion_source,
        'preserve_base_edges': bool(model.preserve_base_edges),
        'reachability_residual_scale': float(
            model.reachability_residual_scale
        ),
        'style_film_mode': model.style_film_mode,
        'style_film_bound': float(model.style_film_bound),
        'lambda_endpoint': float(model.lambda_endpoint),
        'lambda_late_horizon': float(model.lambda_late_horizon),
        'lambda_teacher_distillation': float(
            model.lambda_teacher_distillation
        ),
        'teacher_distillation_mode': model.teacher_distillation_mode,
        'teacher_advantage_margin': float(model.teacher_advantage_margin),
        'teacher_trajectory_cache_dir': (
            model.teacher_trajectory_cache_dir
        ),
        'late_horizon_steps': int(model.late_horizon_steps),
        'late_horizon_reduction': model.late_horizon_reduction,
        'conflict_update_scope': model.conflict_update_scope,
        'priority_injection_mode': model.priority_injection_mode,
        'proposal_conflict_mode_scale': float(
            model.proposal_conflict_mode_scale
        ),
        'proposal_conflict_mode_adapter_scale': float(
            model.proposal_conflict_mode_adapter_scale
        ),
        'proposal_conflict_residual_gate_scale': float(
            model.proposal_conflict_residual_gate_scale
        ),
        'scene_mode_coordination_scale': float(
            model.scene_mode_coordination_scale
        ),
        'joint_mode_composition_scale': float(
            model.joint_mode_composition_scale
        ),
        'joint_mode_assignment_mode': model.joint_mode_assignment_mode,
        'joint_mode_max_exception_agents': int(
            model.joint_mode_max_exception_agents
        ),
        'proposal_conflict_training_scope': (
            model.proposal_conflict_training_scope
        ),
        'num_historical_steps': int(model.num_historical_steps),
        'num_future_steps': int(model.num_future_steps),
        'observation_step_time': float(model.observation_step_time),
    }


def compute_scene_metrics(
        prediction: torch.Tensor,
        target: torch.Tensor,
        miss_threshold: float = 2.0) -> dict:
    """Compute the existing oracle joint metrics for one scenario.

    Args:
        prediction: [num_scored_agents, num_modes, num_future_steps, 2].
        target: [num_scored_agents, num_future_steps, 2].
    """
    if prediction.ndim != 4 or target.ndim != 3:
        raise ValueError('prediction/target must have rank 4/3')
    if prediction.size(0) == 0:
        raise ValueError('a scored scenario must contain at least one agent')
    if prediction.shape[0] != target.shape[0]:
        raise ValueError('prediction and target agent counts differ')
    if prediction.shape[2:] != target.shape[1:]:
        raise ValueError('prediction and target horizon shapes differ')

    error = torch.linalg.vector_norm(
        prediction - target.unsqueeze(1), dim=-1,
    )  # [N,K,F]
    mode_ade = error.mean(dim=(0, 2))
    mode_fde = error[:, :, -1].mean(dim=0)
    ade_mode = int(mode_ade.argmin().item())
    fde_mode = int(mode_fde.argmin().item())
    worst_agent_fde_per_mode = error[:, :, -1].max(dim=0).values
    min_worst_agent_fde = worst_agent_fde_per_mode.min()
    oracle_horizon_de = error.mean(dim=0).min(dim=0).values
    fde_mode_horizon_de = error[:, fde_mode].mean(dim=0)
    ade_mode_horizon_de = error[:, ade_mode].mean(dim=0)

    result = {
        'joint_ade': float(mode_ade[ade_mode].item()),
        'joint_fde': float(mode_fde[fde_mode].item()),
        'joint_mr': float(
            (min_worst_agent_fde > miss_threshold).float().item()
        ),
        'ade_best_mode': ade_mode,
        'fde_best_mode': fde_mode,
        'best_mode_disagrees': int(ade_mode != fde_mode),
        'ade_of_fde_mode': float(mode_ade[fde_mode].item()),
        'fde_of_ade_mode': float(mode_fde[ade_mode].item()),
        'min_worst_agent_fde': float(min_worst_agent_fde.item()),
        'worst_agent_fde_at_fde_mode': float(
            worst_agent_fde_per_mode[fde_mode].item()
        ),
    }
    for step, value in enumerate(oracle_horizon_de, start=1):
        result[f'oracle_de_t{step:02d}'] = float(value.item())
    for step, value in enumerate(fde_mode_horizon_de, start=1):
        result[f'fde_mode_de_t{step:02d}'] = float(value.item())
    for step, value in enumerate(ade_mode_horizon_de, start=1):
        result[f'ade_mode_de_t{step:02d}'] = float(value.item())
    return result


def conflict_stats(data, batch_index: int) -> dict:
    edge_index = data['agent'].get('E_VV_conf')
    if edge_index is None or edge_index.numel() == 0:
        return {
            'directed_conflict_edges': 0,
            'conflict_participants': 0,
            'has_conflict': 0,
        }
    agent_batch = data['agent']['batch']
    edge_mask = agent_batch[edge_index[0]] == batch_index
    selected_edges = edge_index[:, edge_mask]
    participants = (
        selected_edges.reshape(-1).unique().numel()
        if selected_edges.numel() != 0 else 0
    )
    directed_edges = int(selected_edges.size(1))
    return {
        'directed_conflict_edges': directed_edges,
        'conflict_participants': int(participants),
        'has_conflict': int(directed_edges > 0),
    }


def priority_label_count(data, batch_index: int) -> int:
    pair_index = data['agent'].get('conflict_instance_pair_index')
    valid_mask = data['agent'].get('priority_pseudo_label_valid_mask')
    if (
        pair_index is None or valid_mask is None
        or pair_index.numel() == 0 or valid_mask.numel() == 0
    ):
        return 0
    agent_batch = data['agent']['batch']
    instance_batch = agent_batch[pair_index[0]]
    return int((valid_mask & (instance_batch == batch_index)).sum().item())


def evaluate_checkpoint(
        label: str,
        checkpoint_path: Path,
        root: Path,
        manifest_path: Optional[Path],
        output_dir: Path,
        batch_size: int,
        num_workers: int,
        device: torch.device,
        checkpoint_overrides: Optional[dict] = None,
        v22_map_geometry_cache_dir: Optional[Path] = None,
        max_samples: int = 0) -> Tuple[pd.DataFrame, dict]:
    model = PNet.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        map_location='cpu',
        strict=True,
        **(checkpoint_overrides or {}),
    )
    model.eval().to(device)
    config = checkpoint_config(model, checkpoint_path)

    transforms = []
    if v22_map_geometry_cache_dir is not None:
        transforms.append(V22MapGeometryTransform(
            v22_map_geometry_cache_dir,
        ))
        config['v22_map_geometry_cache_dir'] = str(
            v22_map_geometry_cache_dir.resolve()
        )
    else:
        config['v22_map_geometry_cache_dir'] = None
    transforms.extend([
        LaneRandomOcclusion(0.0),
        LaneAssociationTransform(
            model.num_historical_steps,
            model.observation_step_time,
        ),
    ])
    dataset = INTERACTIONDataset(
        root=str(root),
        split='val',
        transform=Compose(transforms),
        num_historical_steps=model.num_historical_steps,
        num_future_steps=model.num_future_steps,
        observation_step_time=model.observation_step_time,
        motion_source=model.motion_source,
        csv_motion_cache_dir=model.csv_motion_cache_dir,
    )
    if manifest_path is not None:
        dataset = subset_from_manifest(dataset, str(manifest_path))
    if max_samples > 0:
        dataset = torch.utils.data.Subset(
            dataset, range(min(max_samples, len(dataset))),
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=num_workers > 0,
    )

    rows = []
    with torch.inference_mode():
        for data in tqdm(loader, desc=label):
            data = data.to(device)
            _, trajectory, _ = model(data)
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
            scored_prediction = trajectory[predicted_agent, -1][scored]
            scored_target = target[predicted_agent, -1][scored]
            scored_batch = data['agent']['batch'][predicted_agent][scored]
            all_agent_batch = data['agent']['batch']

            scenario_names = data['scenario_name']
            case_ids = data['case_id']
            for batch_index in range(len(scenario_names)):
                scene_mask = scored_batch == batch_index
                if not scene_mask.any():
                    raise RuntimeError(
                        'scenario contains no fully scored category-1 agent: '
                        f'{scenario_names[batch_index]}_{int(case_ids[batch_index])}'
                    )
                scenario_name = str(scenario_names[batch_index])
                case_id = int(case_ids[batch_index].item())
                row = {
                    'checkpoint': label,
                    'sample_id': f'{scenario_name}_{case_id}.pt',
                    'scenario_name': scenario_name,
                    'case_id': case_id,
                    'agent_count': int(
                        (all_agent_batch == batch_index).sum().item()
                    ),
                    'scored_agent_count': int(scene_mask.sum().item()),
                }
                row.update(conflict_stats(data, batch_index))
                row['valid_priority_labels'] = priority_label_count(
                    data, batch_index,
                )
                row.update(compute_scene_metrics(
                    scored_prediction[scene_mask],
                    scored_target[scene_mask],
                ))
                rows.append(row)

    frame = pd.DataFrame(rows).sort_values('sample_id').reset_index(drop=True)
    output_path = output_dir / f'{label}_scenarios.csv'
    frame.to_csv(output_path, index=False)
    del model, loader, dataset
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return frame, config


def numeric_summary(values: pd.Series) -> dict:
    array = values.to_numpy(dtype=np.float64)
    standard_deviation = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    standard_error = standard_deviation / np.sqrt(len(array))
    return {
        'mean': float(np.mean(array)),
        'std': standard_deviation,
        'sem': float(standard_error),
        'mean_ci95_low': float(np.mean(array) - 1.96 * standard_error),
        'mean_ci95_high': float(np.mean(array) + 1.96 * standard_error),
        'p50': float(np.quantile(array, 0.50)),
        'p75': float(np.quantile(array, 0.75)),
        'p90': float(np.quantile(array, 0.90)),
        'p95': float(np.quantile(array, 0.95)),
    }


def summarize_checkpoint(frame: pd.DataFrame) -> dict:
    horizon_columns = [
        column for column in frame.columns
        if column.startswith('oracle_de_t')
    ]
    horizon_columns.sort()
    summary = {
        'scenario_count': int(len(frame)),
        'metrics': {
            metric: numeric_summary(frame[metric])
            for metric in METRIC_COLUMNS
        },
        'best_mode_disagreement_rate': float(
            frame['best_mode_disagrees'].mean()
        ),
        'has_conflict_rate': float(frame['has_conflict'].mean()),
        'oracle_horizon_mean': {
            column[len('oracle_de_'):]: float(frame[column].mean())
            for column in horizon_columns
        },
        'groups': {},
    }
    for group_name, grouped in (
        ('scenario_name', frame.groupby('scenario_name', sort=True)),
        ('has_conflict', frame.groupby('has_conflict', sort=True)),
    ):
        summary['groups'][group_name] = {
            str(key): {
                'count': int(len(group)),
                **{
                    metric: float(group[metric].mean())
                    for metric in METRIC_COLUMNS
                },
            }
            for key, group in grouped
        }
    return summary


def print_validation_metrics(summary: dict) -> None:
    """Print the headline validation metrics after the detailed summary."""
    print('\nValidation complete. Headline metrics:')
    for label, checkpoint_summary in summary['checkpoints'].items():
        metrics = checkpoint_summary['metrics']
        print(
            f'[{label}] '
            f'scenarios={checkpoint_summary["scenario_count"]} '
            f'joint_ADE={metrics["joint_ade"]["mean"]:.8f} '
            f'joint_FDE={metrics["joint_fde"]["mean"]:.8f} '
            f'joint_MR={metrics["joint_mr"]["mean"]:.8f}'
        )


def compare_frames(
        candidate_label: str,
        candidate: pd.DataFrame,
        reference_label: str,
        reference: pd.DataFrame,
        output_dir: Path) -> Tuple[pd.DataFrame, dict]:
    merged = candidate.merge(
        reference,
        on='sample_id',
        how='inner',
        suffixes=('_candidate', '_reference'),
        validate='one_to_one',
    )
    if len(merged) != len(candidate) or len(merged) != len(reference):
        raise ValueError(
            f'unpaired comparison {candidate_label} vs {reference_label}: '
            f'{len(candidate)}, {len(reference)}, paired={len(merged)}'
        )
    delta_columns = []
    diagnostic_columns = list(METRIC_COLUMNS) + [
        column for column in candidate.columns
        if column.startswith('oracle_de_t')
    ]
    for column in diagnostic_columns:
        delta_column = f'{column}_delta'
        merged[delta_column] = (
            merged[f'{column}_candidate']
            - merged[f'{column}_reference']
        )
        delta_columns.append(delta_column)

    output_path = output_dir / (
        f'{candidate_label}_vs_{reference_label}_paired.csv'
    )
    merged.to_csv(output_path, index=False)
    summary = {
        'candidate': candidate_label,
        'reference': reference_label,
        'paired_scenarios': int(len(merged)),
        'metric_deltas': {},
        'oracle_horizon_delta_mean': {},
        'groups_by_candidate_conflict': {},
        'groups_by_scenario': {},
    }
    for metric in METRIC_COLUMNS:
        delta = merged[f'{metric}_delta']
        delta_summary = numeric_summary(delta)
        summary['metric_deltas'][metric] = {
            **delta_summary,
            'candidate_win_rate': float((delta < 0).mean()),
            'candidate_loss_rate': float((delta > 0).mean()),
        }
    for delta_column in delta_columns:
        if delta_column.startswith('oracle_de_t'):
            summary['oracle_horizon_delta_mean'][
                delta_column[:-len('_delta')]
            ] = float(merged[delta_column].mean())
    for conflict_value, group in merged.groupby(
            'has_conflict_candidate', sort=True):
        summary['groups_by_candidate_conflict'][str(conflict_value)] = {
            'count': int(len(group)),
            **{
                f'{metric}_delta': float(group[f'{metric}_delta'].mean())
                for metric in METRIC_COLUMNS
            },
        }
    for scenario_name, group in merged.groupby(
            'scenario_name_candidate', sort=True):
        summary['groups_by_scenario'][str(scenario_name)] = {
            'count': int(len(group)),
            **{
                f'{metric}_delta': float(group[f'{metric}_delta'].mean())
                for metric in METRIC_COLUMNS
            },
        }
    return merged, summary


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument(
        '--manifest', type=Path, default=None,
        help='Optional processed-filename manifest; omit for full validation.',
    )
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument(
        '--checkpoint', action='append', required=True,
        help='Repeat LABEL=/absolute/path.ckpt for every checkpoint.',
    )
    parser.add_argument(
        '--candidate', action='append', required=True,
        help=(
            'Checkpoint label used as a candidate in paired deltas. Repeat '
            'to analyze multiple trained seeds without rerunning inference.'
        ),
    )
    parser.add_argument(
        '--original_checkpoint_label', action='append', default=[],
        help=(
            'Checkpoint label that predates trunk/motion hparams and must be '
            'strictly loaded as trunk_variant=original, motion_source=csv. '
            'Repeat when needed.'
        ),
    )
    parser.add_argument(
        '--csv_motion_cache_dir', type=Path, default=None,
        help=(
            'CSV motion cache required by every --original_checkpoint_label '
            'compatibility override.'
        ),
    )
    parser.add_argument(
        '--v22_map_geometry_label', action='append', default=[],
        help=(
            'Checkpoint label evaluated with the unresampled v22 map '
            'geometry sidecar. Repeat when needed.'
        ),
    )
    parser.add_argument(
        '--v22_map_geometry_cache_dir', type=Path, default=None,
        help=(
            'Scenario-level map geometry cache required by every '
            '--v22_map_geometry_label.'
        ),
    )
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument(
        '--device', default='auto',
        choices=('auto', 'cpu', 'cuda'),
    )
    parser.add_argument(
        '--max_samples', type=int, default=0,
        help='Optional first-N smoke-test limit after applying the manifest.',
    )
    args = parser.parse_args()

    checkpoints = parse_checkpoint_specs(args.checkpoint)
    labels = {label for label, _ in checkpoints}
    candidate_labels = list(dict.fromkeys(args.candidate))
    missing_candidates = set(candidate_labels) - labels
    if missing_candidates:
        parser.error(
            '--candidate must match one --checkpoint label; missing: '
            + ', '.join(sorted(missing_candidates))
        )
    original_labels = set(args.original_checkpoint_label)
    missing_original_labels = original_labels - labels
    if missing_original_labels:
        parser.error(
            '--original_checkpoint_label must match one --checkpoint label; '
            'missing: ' + ', '.join(sorted(missing_original_labels))
        )
    if original_labels and args.csv_motion_cache_dir is None:
        parser.error(
            '--csv_motion_cache_dir is required with '
            '--original_checkpoint_label'
        )
    if (
        args.csv_motion_cache_dir is not None
        and not args.csv_motion_cache_dir.is_dir()
    ):
        parser.error(
            f'CSV motion cache does not exist: {args.csv_motion_cache_dir}'
        )
    v22_map_geometry_labels = set(args.v22_map_geometry_label)
    missing_v22_map_geometry_labels = v22_map_geometry_labels - labels
    if missing_v22_map_geometry_labels:
        parser.error(
            '--v22_map_geometry_label must match one --checkpoint label; '
            'missing: '
            + ', '.join(sorted(missing_v22_map_geometry_labels))
        )
    if (
        v22_map_geometry_labels
        and args.v22_map_geometry_cache_dir is None
    ):
        parser.error(
            '--v22_map_geometry_cache_dir is required with '
            '--v22_map_geometry_label'
        )
    if (
        args.v22_map_geometry_cache_dir is not None
        and not args.v22_map_geometry_cache_dir.is_dir()
    ):
        parser.error(
            'v22 map geometry cache does not exist: '
            f'{args.v22_map_geometry_cache_dir}'
        )
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_samples < 0:
        parser.error('batch_size must be positive; worker/sample limits non-negative')
    if args.manifest is not None and not args.manifest.is_file():
        parser.error(f'manifest does not exist: {args.manifest}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f'output directory is not empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    torch.manual_seed(2023)

    frames: Dict[str, pd.DataFrame] = {}
    configurations = {}
    for label, checkpoint_path in checkpoints:
        frame, config = evaluate_checkpoint(
            label=label,
            checkpoint_path=checkpoint_path,
            root=args.root.resolve(),
            manifest_path=(
                args.manifest.resolve() if args.manifest is not None else None
            ),
            output_dir=args.output_dir.resolve(),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            checkpoint_overrides=(
                {
                    'trunk_variant': 'original',
                    'motion_source': 'csv',
                    'csv_motion_cache_dir': str(
                        args.csv_motion_cache_dir.resolve()
                    ),
                }
                if label in original_labels else None
            ),
            v22_map_geometry_cache_dir=(
                args.v22_map_geometry_cache_dir.resolve()
                if label in v22_map_geometry_labels else None
            ),
            max_samples=args.max_samples,
        )
        frames[label] = frame
        configurations[label] = config

    summary = {
        'schema_version': 1,
        'root': str(args.root.resolve()),
        'manifest': (
            str(args.manifest.resolve())
            if args.manifest is not None else None
        ),
        'device': str(device),
        'max_samples': int(args.max_samples),
        'configurations': configurations,
        'checkpoints': {
            label: summarize_checkpoint(frame)
            for label, frame in frames.items()
        },
        'comparisons': {},
    }
    for candidate_label in candidate_labels:
        candidate = frames[candidate_label]
        for reference_label, reference in frames.items():
            if reference_label == candidate_label:
                continue
            _, comparison = compare_frames(
                candidate_label=candidate_label,
                candidate=candidate,
                reference_label=reference_label,
                reference=reference,
                output_dir=args.output_dir.resolve(),
            )
            summary['comparisons'][
                f'{candidate_label}_vs_{reference_label}'
            ] = comparison

    summary_path = args.output_dir / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print_validation_metrics(summary)


if __name__ == '__main__':
    main()
