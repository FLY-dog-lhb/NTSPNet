from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from analyze_endpoint_errors import compute_scene_metrics
from analyze_endpoint_errors import parse_checkpoint_specs
from datamodules.INTERACTION_datamodule import subset_from_manifest
from datasets import INTERACTIONDataset
from model import PNet
from transforms import LaneAssociationTransform
from transforms import LaneRandomOcclusion
from utils import generate_target


def select_cases(frame: pd.DataFrame,
                 scenarios: Iterable[str],
                 worst_per_scenario: int,
                 best_per_scenario: int,
                 max_scored_agents: int) -> pd.DataFrame:
    required = {
        'sample_id', 'scenario_name_candidate', 'joint_fde_delta',
        'has_conflict_candidate', 'scored_agent_count_candidate',
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f'paired CSV is missing columns: {missing}')
    filtered = frame[
        frame['scenario_name_candidate'].isin(list(scenarios))
        & (frame['has_conflict_candidate'] == 0)
        & (frame['scored_agent_count_candidate'] <= max_scored_agents)
    ]
    selected = []
    for scenario in scenarios:
        group = filtered[
            filtered['scenario_name_candidate'] == scenario
        ]
        if group.empty:
            raise ValueError(
                f'no eligible samples for requested scenario: {scenario}'
            )
        for role, cases in (
            ('worst', group.nlargest(worst_per_scenario, 'joint_fde_delta')),
            ('best', group.nsmallest(best_per_scenario, 'joint_fde_delta')),
        ):
            cases = cases.copy()
            cases['selection_role'] = role
            cases['selection_rank'] = np.arange(1, len(cases) + 1)
            selected.append(cases)
    result = pd.concat(selected, ignore_index=True)
    if result['sample_id'].duplicated().any():
        raise ValueError('case selection produced duplicate sample IDs')
    return result


def endpoint_geometry_metrics(prediction: torch.Tensor,
                              target: torch.Tensor,
                              best_mode: int,
                              map_points: torch.Tensor,
                              late_start_step: int = 23) -> dict:
    """Summarize the endpoint-selected mode in global coordinates."""
    selected = prediction[:, best_mode]
    tangent = target[:, -1] - target[:, -2]
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    tangent = tangent / tangent_norm.clamp_min(1e-6)
    normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
    endpoint_error = selected[:, -1] - target[:, -1]
    longitudinal = (endpoint_error * tangent).sum(dim=-1)
    lateral = (endpoint_error * normal).sum(dim=-1)

    all_endpoint_error = (
        prediction[:, :, -1] - target[:, -1].unsqueeze(1)
    )
    all_longitudinal = (
        all_endpoint_error * tangent.unsqueeze(1)
    ).sum(dim=-1)
    all_lateral = (
        all_endpoint_error * normal.unsqueeze(1)
    ).sum(dim=-1)
    joint_longitudinal_coverage = all_longitudinal.abs().mean(dim=0).min()
    joint_lateral_coverage = all_lateral.abs().mean(dim=0).min()
    endpoint_pair_distance = torch.cdist(
        prediction[:, :, -1], prediction[:, :, -1],
    )
    num_modes = prediction.size(1)
    if num_modes > 1:
        pair_mask = torch.triu(
            torch.ones(
                num_modes, num_modes, dtype=torch.bool,
                device=prediction.device,
            ),
            diagonal=1,
        )
        endpoint_mode_spread = endpoint_pair_distance[:, pair_mask].mean()
    else:
        endpoint_mode_spread = prediction.new_zeros(())

    start_index = late_start_step - 1
    predicted_late_length = torch.linalg.vector_norm(
        selected[:, start_index + 1:] - selected[:, start_index:-1], dim=-1,
    ).sum(dim=-1)
    target_late_length = torch.linalg.vector_norm(
        target[:, start_index + 1:] - target[:, start_index:-1], dim=-1,
    ).sum(dim=-1)
    late_ratio = predicted_late_length / target_late_length.clamp_min(1e-6)

    if map_points.numel() == 0:
        map_distance = selected.new_full((selected.size(0),), float('nan'))
    else:
        map_distance = torch.cdist(
            selected[:, -1].unsqueeze(0), map_points.unsqueeze(0),
        ).squeeze(0).min(dim=-1).values
    return {
        'endpoint_longitudinal_mean': float(longitudinal.mean().item()),
        'endpoint_longitudinal_abs_mean': float(longitudinal.abs().mean().item()),
        'endpoint_lateral_mean': float(lateral.mean().item()),
        'endpoint_lateral_abs_mean': float(lateral.abs().mean().item()),
        'joint_longitudinal_coverage': float(
            joint_longitudinal_coverage.item()
        ),
        'joint_lateral_coverage': float(joint_lateral_coverage.item()),
        'endpoint_mode_spread_mean': float(endpoint_mode_spread.item()),
        'late_path_ratio_mean': float(late_ratio.mean().item()),
        'endpoint_map_distance_mean': float(map_distance.mean().item()),
    }


def build_dataset(model: PNet, root: Path, manifest: Path):
    dataset = INTERACTIONDataset(
        root=str(root),
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
    return subset_from_manifest(dataset, str(manifest))


def evaluate_cases(label: str,
                   checkpoint: Path,
                   root: Path,
                   manifest: Path,
                   device: torch.device) -> Tuple[List[dict], Dict[str, dict]]:
    model = PNet.load_from_checkpoint(
        checkpoint_path=str(checkpoint), map_location='cpu', strict=True,
    )
    model.eval().to(device)
    dataset = build_dataset(model, root, manifest)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    cases = {}
    with torch.inference_mode():
        for data in loader:
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
            prediction = trajectory[predicted_agent, -1][scored].float().cpu()
            scored_target = target[predicted_agent, -1][scored].float().cpu()
            metrics = compute_scene_metrics(prediction, scored_target)
            map_points = data['polyline']['position'].float().cpu()
            metrics.update(endpoint_geometry_metrics(
                prediction, scored_target, metrics['fde_best_mode'], map_points,
            ))
            sample_id = (
                f"{data['scenario_name'][0]}_{int(data['case_id'][0])}.pt"
            )
            rows.append({'checkpoint': label, 'sample_id': sample_id, **metrics})
            scored_indices = torch.nonzero(
                predicted_agent, as_tuple=False,
            ).squeeze(-1)[scored]
            cases[sample_id] = {
                'prediction': prediction.numpy(),
                'target': scored_target.numpy(),
                'best_mode': metrics['fde_best_mode'],
                'all_history': data['agent']['position'][
                    :, :model.num_historical_steps
                ].float().cpu().numpy(),
                'all_history_mask': data['agent']['visible_mask'][
                    :, :model.num_historical_steps
                ].cpu().numpy(),
                'scored_indices': scored_indices.cpu().numpy(),
                'map_points': map_points.numpy(),
                'metrics': metrics,
            }
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return rows, cases


def plot_case(sample_id: str,
              labels: List[str],
              results: Dict[str, Dict[str, dict]],
              fde_delta: float,
              output_path: Path) -> None:
    shared = results[labels[0]][sample_id]
    bounds = []
    bounds.append(shared['all_history'][shared['all_history_mask']])
    bounds.append(shared['target'].reshape(-1, 2))
    for label in labels:
        bounds.append(results[label][sample_id]['prediction'].reshape(-1, 2))
    points = np.concatenate(bounds, axis=0)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    x_min, y_min = points.min(axis=0) - 6.0
    x_max, y_max = points.max(axis=0) + 6.0

    figure, axes = plt.subplots(
        1, len(labels), figsize=(7 * len(labels), 6), squeeze=False,
    )
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(2, len(shared['target']))))
    for axis, label in zip(axes[0], labels):
        case = results[label][sample_id]
        map_points = case['map_points']
        axis.scatter(
            map_points[:, 0], map_points[:, 1], s=2, c='#d0d0d0',
            linewidths=0, zorder=0,
        )
        for history, mask in zip(
                case['all_history'], case['all_history_mask']):
            axis.plot(
                history[mask, 0], history[mask, 1], color='#888888',
                linewidth=1.0, alpha=0.7,
            )
        best_mode = int(case['best_mode'])
        for agent_index, (prediction, target) in enumerate(zip(
                case['prediction'], case['target'])):
            color = colors[agent_index]
            for mode in range(prediction.shape[0]):
                axis.plot(
                    prediction[mode, :, 0], prediction[mode, :, 1],
                    color=color, linewidth=0.8, alpha=0.18,
                )
            axis.plot(
                prediction[best_mode, :, 0], prediction[best_mode, :, 1],
                color=color, linewidth=2.2, label=f'agent {agent_index} best',
            )
            axis.scatter(
                prediction[best_mode, 23, 0], prediction[best_mode, 23, 1],
                s=28, facecolors='none', edgecolors=[color], zorder=4,
            )
            axis.scatter(
                prediction[best_mode, -1, 0], prediction[best_mode, -1, 1],
                s=42, marker='x', color=color, zorder=5,
            )
            axis.plot(
                target[:, 0], target[:, 1], color='black', linewidth=2.0,
                linestyle='--', label=f'agent {agent_index} target',
            )
        metrics = case['metrics']
        axis.set_title(
            f"{label}: FDE={metrics['joint_fde']:.3f}, "
            f"long={metrics['endpoint_longitudinal_mean']:+.3f}, "
            f"lat_abs={metrics['endpoint_lateral_abs_mean']:.3f}"
        )
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=0.15)
        axis.legend(fontsize=7, loc='best')
    figure.suptitle(f'{sample_id} | version14-version39 FDE={fde_delta:+.3f} m')
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--paired_csv', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument(
        '--scenario', action='append', required=True,
        help='Repeat for every scenario to inspect.',
    )
    parser.add_argument('--worst_per_scenario', type=int, default=4)
    parser.add_argument('--best_per_scenario', type=int, default=1)
    parser.add_argument('--max_scored_agents', type=int, default=5)
    parser.add_argument(
        '--all_eligible', action='store_true',
        help='Analyze every eligible row instead of worst/best case selection.',
    )
    parser.add_argument(
        '--no_plots', action='store_true',
        help='Write numeric diagnostics without rendering per-case PNGs.',
    )
    parser.add_argument(
        '--device', choices=('auto', 'cpu', 'cuda'), default='auto',
    )
    args = parser.parse_args()
    if args.worst_per_scenario <= 0 or args.best_per_scenario <= 0:
        parser.error('worst/best counts must be positive')
    if args.max_scored_agents <= 0:
        parser.error('max_scored_agents must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f'output directory is not empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_specs = parse_checkpoint_specs(args.checkpoint)
    paired = pd.read_csv(args.paired_csv)
    if args.all_eligible:
        selection = paired[
            paired['scenario_name_candidate'].isin(args.scenario)
            & (paired['has_conflict_candidate'] == 0)
            & (
                paired['scored_agent_count_candidate']
                <= args.max_scored_agents
            )
        ].copy()
        if selection.empty:
            parser.error('no eligible rows match the requested scenarios')
        selection['selection_role'] = 'all_eligible'
        selection['selection_rank'] = np.arange(1, len(selection) + 1)
    else:
        selection = select_cases(
            paired, args.scenario, args.worst_per_scenario,
            args.best_per_scenario, args.max_scored_agents,
        )
    selection_path = args.output_dir / 'selection.csv'
    selection.to_csv(selection_path, index=False)
    manifest_path = args.output_dir / 'selected_manifest.txt'
    manifest_path.write_text('\n'.join(selection['sample_id']) + '\n')

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')

    labels = [label for label, _ in checkpoint_specs]
    all_rows = []
    results = {}
    for label, checkpoint in checkpoint_specs:
        rows, cases = evaluate_cases(
            label, checkpoint, args.root.resolve(), manifest_path, device,
        )
        all_rows.extend(rows)
        results[label] = cases
    pd.DataFrame(all_rows).to_csv(
        args.output_dir / 'trajectory_diagnostics.csv', index=False,
    )

    selected_delta = selection.set_index('sample_id')['joint_fde_delta']
    for sample_id in selection['sample_id']:
        for label in labels[1:]:
            if not np.allclose(
                results[labels[0]][sample_id]['target'],
                results[label][sample_id]['target'],
            ):
                raise ValueError(f'target mismatch across checkpoints: {sample_id}')
        if not args.no_plots:
            plot_case(
                sample_id=sample_id,
                labels=labels,
                results=results,
                fde_delta=float(selected_delta[sample_id]),
                output_path=args.output_dir / f'{sample_id[:-3]}.png',
            )


if __name__ == '__main__':
    main()
