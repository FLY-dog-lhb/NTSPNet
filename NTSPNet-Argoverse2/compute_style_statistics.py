"""Fit style pseudo-label calibration using TRAIN observations only."""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from cli import PROJECT
from datasets import Argoverse2Dataset
from model import PNet


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/home/lhb/data-Argoverse2')
    parser.add_argument('--cache-dir', default=str(PROJECT / 'cache'))
    parser.add_argument('--reader', choices=('parquet', 'av2'), default='parquet')
    parser.add_argument('--manifest', help='Training scene IDs only')
    parser.add_argument('--limit', type=int, default=5000, help='Seeded random training subset; 0 means all selected scenes')
    parser.add_argument('--max-prefixes', type=int, default=200000, help='Uniform reservoir limit')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2048)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--a2a-radius', type=float, default=80.)
    parser.add_argument('--output', default=str(PROJECT / 'artifacts/train_style_statistics.pt'))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error('Statistics output already exists; use a new path')
    if Path(args.root).resolve() in output.parents:
        parser.error('Statistics must be outside the raw dataset')
    if args.limit < 0 or args.max_prefixes < 1:
        parser.error('Invalid sampling limits')
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset = Argoverse2Dataset(args.root, 'train', args.cache_dir, args.manifest, reader=args.reader)
    if args.limit and len(dataset) > args.limit:
        indices = np.sort(rng.choice(len(dataset), args.limit, replace=False))
        dataset.ids = [dataset.ids[i] for i in indices]
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
    # Indicators depend on history and graphs, not learned weights. Small channels
    # reduce work; temporal/radius configuration stays identical to training defaults.
    model = PNet(hidden_dim=16, num_heads=4, num_attn_layers=1, num_hops=1, lambda_style=0, a2a_radius=args.a2a_radius).to(args.device).eval()
    values = np.empty((0, 6), dtype=np.float32); keys = np.empty(0)
    total = 0
    with torch.inference_mode():
        for data in tqdm(loader, desc='Train style calibration'):
            data = data.to(args.device)
            # Physically remove all future supervision before feature extraction.
            data['agent'].position[:, 50:] = 0
            data['agent'].visible_mask[:, 50:] = False
            model(data)
            selected = data['agent'].style_prefix_reliable_mask
            current = data['agent'].style_prefix_score_indicator[selected].cpu().numpy()
            total += len(current)
            values = np.concatenate((values, current)); keys = np.r_[keys, rng.random(len(current))]
            if len(keys) > 2 * args.max_prefixes:
                keep = np.argpartition(keys, args.max_prefixes - 1)[:args.max_prefixes]
                values, keys = values[keep], keys[keep]
    if not total:
        raise RuntimeError('No reliable vehicle history prefixes in selected training scenes')
    if len(keys) > args.max_prefixes:
        keep = np.argpartition(keys, args.max_prefixes - 1)[:args.max_prefixes]
        values = values[keep]
    indicators = torch.from_numpy(values).float()
    provider = model.Backbone
    median = indicators.quantile(.5, dim=0)
    iqr = indicators.quantile(.75, dim=0) - indicators.quantile(.25, dim=0)
    weights = provider.style_score_weights.cpu()
    scores = ((indicators - median) / (iqr + provider.style_statistics_epsilon) * weights).sum(dim=-1)
    configuration_names = ('num_historical_steps', 'num_future_steps', 'observation_step_time',
                           'a2a_radius', 'interaction_distance_margin', 'lane_arrival_time_threshold',
                           'ttc_speed_epsilon', 'interaction_ttc_threshold', 'interaction_collision_distance',
                           'traffic_current_radius', 'traffic_lane_lateral_radius', 'traffic_topology_base_distance',
                           'style_ttc_max', 'style_min_visible_steps')
    payload = dict(version=2, indicator_names=list(provider.STYLE_SCORE_INDICATOR_NAMES),
                   class_names=list(provider.STYLE_CLASS_NAMES), score_weights=weights, median=median, iqr=iqr,
                   q33=scores.quantile(.33), q66=scores.quantile(.66), epsilon=provider.style_statistics_epsilon,
                   configuration={name: getattr(provider, name) for name in configuration_names},
                   source_split='train', dataset_schema=model.hparams.dataset_schema, seed=args.seed,
                   selected_scene_ids=dataset.ids, total_reliable_prefixes=total,
                   sampled_prefixes=len(indicators), centerline_policy='raw_json_else_boundary_midpoint')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps(dict(output=str(output), scenes=len(dataset), total_prefixes=total,
                          sampled_prefixes=len(indicators)), indent=2))


if __name__ == '__main__':
    main()
