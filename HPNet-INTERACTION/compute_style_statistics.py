"""Precompute train-only robust statistics for driving-style pseudo labels."""

from argparse import ArgumentParser
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

from datasets import INTERACTIONDataset
from model import PNet
from transforms import LaneAssociationTransform
from transforms import LaneRandomOcclusion


def resolve_device(device_name: str) -> torch.device:
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return device


def compute_statistics(model: PNet,
                       dataloader: DataLoader,
                       device: torch.device) -> dict:
    model = model.to(device)
    model.eval()
    reliable_indicators = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Collecting style indicators'):
            data = data.to(device)
            model(data)
            reliable_mask = data['agent']['style_prefix_reliable_mask']
            if reliable_mask.any():
                reliable_indicators.append(
                    data['agent']['style_prefix_score_indicator'][reliable_mask].cpu()
                )

    if not reliable_indicators:
        raise RuntimeError(
            'no reliable training style anchors were found; supervision requires '
            f'a car and at least {model.style_min_visible_steps} visible prefix frames'
        )

    indicators = torch.cat(reliable_indicators, dim=0).float()
    median = torch.quantile(indicators, 0.50, dim=0)
    q25 = torch.quantile(indicators, 0.25, dim=0)
    q75 = torch.quantile(indicators, 0.75, dim=0)
    iqr = q75 - q25
    standardized = (
        (indicators - median)
        / (iqr + model.style_statistics_epsilon)
    )
    score_weights = model.Backbone.style_score_weights.detach().cpu()
    scores = (standardized * score_weights).sum(dim=-1)
    q33 = torch.quantile(scores, 0.33)
    q66 = torch.quantile(scores, 0.66)
    labels = torch.where(
        scores <= q33,
        torch.full_like(scores, 2, dtype=torch.long),
        torch.where(
            scores >= q66,
            torch.zeros_like(scores, dtype=torch.long),
            torch.ones_like(scores, dtype=torch.long),
        ),
    )

    return {
        'version': 2,
        'indicator_names': list(model.Backbone.STYLE_SCORE_INDICATOR_NAMES),
        'class_names': list(model.Backbone.STYLE_CLASS_NAMES),
        'score_weights': score_weights,
        'median': median,
        'iqr': iqr,
        'q33': q33,
        'q66': q66,
        'epsilon': model.style_statistics_epsilon,
        'num_reliable_vehicles': int(indicators.size(0)),
        'num_reliable_prefixes': int(indicators.size(0)),
        'class_counts': torch.bincount(labels, minlength=3),
        'configuration': {
            'num_historical_steps': model.num_historical_steps,
            'num_future_steps': model.num_future_steps,
            'observation_step_time': model.observation_step_time,
            'a2a_radius': model.a2a_radius,
            'interaction_distance_margin': model.interaction_distance_margin,
            'lane_arrival_time_threshold': model.lane_arrival_time_threshold,
            'ttc_speed_epsilon': model.ttc_speed_epsilon,
            'interaction_ttc_threshold': model.interaction_ttc_threshold,
            'interaction_collision_distance': model.interaction_collision_distance,
            'traffic_current_radius': model.traffic_current_radius,
            'traffic_lane_lateral_radius': model.traffic_lane_lateral_radius,
            'traffic_topology_base_distance': model.traffic_topology_base_distance,
            'style_ttc_max': model.style_ttc_max,
            'style_min_visible_steps': model.style_min_visible_steps,
        },
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--statistics_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output', type=str, default=None)
    PNet.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.statistics_batch_size <= 0:
        raise ValueError('statistics_batch_size must be positive')
    if args.num_workers < 0:
        raise ValueError('num_workers must be non-negative')

    output_path = Path(
        args.output or Path(args.root) / 'train_style_statistics.pt'
    )
    args.style_statistics_path = None
    model = PNet(**vars(args))
    dataset = INTERACTIONDataset(
        root=args.root,
        split='train',
        transform=Compose([
            LaneRandomOcclusion(0.0),
            LaneAssociationTransform(
                args.num_historical_steps,
                args.observation_step_time,
            ),
        ]),
        num_historical_steps=args.num_historical_steps,
        num_future_steps=args.num_future_steps,
        observation_step_time=args.observation_step_time,
        motion_source=args.motion_source,
        csv_motion_cache_dir=args.csv_motion_cache_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.statistics_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    statistics = compute_statistics(model, dataloader, resolve_device(args.device))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(statistics, output_path)
    print(
        f'Saved statistics for {statistics["num_reliable_prefixes"]} reliable '
        f'style prefixes to {output_path}'
    )
    print(f'Class counts [aggressive, normal, conservative]: {statistics["class_counts"].tolist()}')
