#!/usr/bin/env python3
"""Frozen INTERACTION -> AV2 zero-shot evaluation, custom 1s -> 3s protocol.

Run as a standalone process: source INTERACTION modules must be imported before
any AV2 model modules. Neither the AV2 model nor its checkpoint loader is used.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

from cross_domain.runtime import joint_metrics, load_checkpoint, sha256
from cross_domain.reporting import METRIC_KEYS, SceneAccumulator, marginal_actor_metrics


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interaction-source', type=Path, default=root.parent / 'HPNet-INTERACTION')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/home/lhb/data-Argoverse2'))
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--test-annotations', type=Path, help='Released Multi-World test annotation parquet; labels only')
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not already exist')
    parser.add_argument('--limit', type=int, help='Seeded random subset; omit for the full selected split')
    parser.add_argument('--seed', type=int, default=2048)
    parser.add_argument('--manifest', type=Path, help='One scenario ID per line; incompatible with --limit')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.split == 'test' and args.test_annotations is None:
        parser.error('--split test requires --test-annotations; blind histories cannot be scored')
    if args.split == 'val' and args.test_annotations is not None:
        parser.error('--test-annotations is only allowed with --split test')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.limit and args.manifest:
        parser.error('--limit and --manifest cannot be combined')
    if args.threads <= 0:
        parser.error('--threads must be positive')
    output = args.output_dir.resolve()
    raw = args.root.resolve()
    if raw == output or raw in output.parents:
        parser.error('Keep reports outside the raw dataset')
    if output.exists():
        parser.error('Output directory already exists; select a new one')

    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; explicitly select --device cpu')
    model, provenance = load_checkpoint(args.interaction_source, args.checkpoint)
    # Import only after activating the source packages. This also makes lane
    # association and historical kinematics use the source implementation.
    from cross_domain.data import PROTOCOL, av2, build_input
    from torch_geometric.data import Batch
    all_ids = sorted(p.name for p in (raw / args.split).iterdir() if p.is_dir())
    if args.manifest:
        ids = [s.strip() for s in args.manifest.read_text().splitlines() if s.strip()]
        if len(set(ids)) != len(ids) or not set(ids).issubset(all_ids):
            parser.error('Manifest must contain unique existing scenario IDs in the selected split')
    else:
        ids = all_ids
        if args.limit is not None:
            if args.limit > len(ids):
                parser.error('--limit exceeds selected split size')
            ids = sorted(random.Random(args.seed).sample(ids, args.limit))
    if not ids:
        parser.error('No scenarios selected')
    annotations = None
    if args.split == 'test':
        from cross_domain.test_annotations import MultiWorldTestAnnotations
        annotations = MultiWorldTestAnnotations(args.test_annotations, ids)
    initial = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    model.to(args.device)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'scenario_ids.txt').write_text(''.join(s + '\n' for s in ids))
    metadata = dict(protocol=PROTOCOL, report_schema_version=3, status='running', split=args.split, dataset_root=str(raw),
                    selected_scenes=len(ids), available_scenes=len(all_ids),
                    complete_val_split=args.split == 'val' and set(ids) == set(all_ids),
                    complete_split=set(ids) == set(all_ids), seed=args.seed,
                    test_annotations=annotations.provenance if annotations else None,
                    history_timesteps=[40, 49], prediction_timesteps=[50, 79], sample_hz=10,
                    targets='AV2 scored/focal vehicle, complete 10-frame history and 30-frame future',
                    context='dynamic agents observed in 40..49; vehicle/bus=car, pedestrian/cyclist/motorcyclist=pedestrian/bicycle',
                    size_policy='length=width=0: AV2 MF has no measured actor dimensions',
                    map_policy='native centerline; source midpoint-index/tangent; AV2 geometric conflict proxy; unknown traffic rules=0',
                    future_labels_passed_to_model=False, training_or_target_calibration=False,
                    metric_aggregation='equal weight per eligible scene; joint metrics share scene modes, marginal metrics choose per-actor modes; ADE/FDE independently minimized',
                    marginal_mr_definition='mean over actors of (per-actor minimum FDE > 2m), then mean over scenes',
                    subset_definition='single_target: exactly 1 eligible target; multi_target: at least 2 eligible targets',
                    selection_manifest=str(args.manifest.resolve()) if args.manifest else None,
                    num_modes=model.num_modes, device=args.device, torch_version=torch.__version__,
                    cuda_device=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                    provenance=provenance,
                    adapter_sha256={str(p.relative_to(root)): sha256(p) for p in
                                    [Path(__file__).resolve(), root / 'cross_domain/data.py', root / 'cross_domain/runtime.py', root / 'cross_domain/reporting.py', root / 'cross_domain/test_annotations.py',
                                     root / 'datasets/argoverse2_dataset.py']})
    (output / 'run.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    fields = ['scenario_id', 'status', 'context_agents', 'targets', 'scored_vehicles', 'scored_nonvehicles',
              'target_subset', *METRIC_KEYS]
    all_stats = SceneAccumulator()
    subset_stats = {key: SceneAccumulator() for key in ('single_target', 'multi_target')}
    evaluated = focal_count = total_targets = skipped = 0
    start = time.monotonic()
    try:
        with (output / 'per_scene.csv').open('w', newline='') as stream, \
                (output / 'per_actor.csv').open('w', newline='') as actor_stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            actor_writer = csv.DictWriter(actor_stream, fieldnames=[
                'scenario_id', 'track_id', 'target_subset', 'marginal_minADE', 'marginal_minFDE', 'marginal_actorMR'])
            actor_writer.writeheader()
            with torch.inference_mode():
                for index, name in enumerate(ids):
                    folder = raw / args.split / name
                    frame, raw_map = av2.read_scene(folder / ('scenario_' + name + '.parquet'),
                                                  folder / ('log_map_archive_' + name + '.json'))
                    future_frame = annotations.future_frame(name, frame) if annotations else None
                    graph, labels = build_input(frame, raw_map, name, future_frame=future_frame)
                    eligible = labels['eligible']
                    row = dict(scenario_id=name, context_agents=graph['agent'].num_nodes,
                               targets=int(eligible.sum()), scored_vehicles=labels['scored_vehicles'],
                               scored_nonvehicles=labels['scored_nonvehicles'])
                    if not eligible.any():
                        skipped += 1; row['status'] = 'no_eligible_scored_vehicle'
                        row['target_subset'] = 'no_target'
                    else:
                        batch = Batch.from_data_list([graph]).to(args.device)
                        prediction = model(batch)[1][:, -1].cpu()
                        if prediction.shape != (graph['agent'].num_nodes, model.num_modes, 30, 2):
                            raise ValueError('Unexpected prediction shape: %r' % (prediction.shape,))
                        target = labels['position']
                        row.update(joint_metrics(prediction[eligible], target[eligible]))
                        row['target_subset'] = 'multi_target' if row['targets'] >= 2 else 'single_target'
                        actor_metrics = marginal_actor_metrics(prediction[eligible], target[eligible])
                        # Reduce in float64 so scene means and per-actor CSV agree closely.
                        row.update({key: float(value.double().mean()) for key, value in actor_metrics.items()})
                        for suffix in ('ADE', 'FDE'):
                            row['joint_minus_marginal_' + suffix] = (
                                row['minJoint' + suffix] - row['marginal_min' + suffix])
                        eligible_indices = eligible.nonzero(as_tuple=False).flatten().tolist()
                        for local_idx, graph_idx in enumerate(eligible_indices):
                            actor_writer.writerow(dict(
                                scenario_id=name, track_id=labels['track_ids'][graph_idx],
                                target_subset=row['target_subset'],
                                **{key: float(value[local_idx]) for key, value in actor_metrics.items()}))
                        actor_stream.flush()
                        # Fixed constant-velocity reference, same observations and targets;
                        # provides context without fitting anything on AV2.
                        agent = graph['agent']
                        dt = torch.arange(1, 31, dtype=torch.float32)[None, :, None] * .1
                        cv = agent.position[:, 9, None] + agent.csv_velocity[:, 9, None] * dt
                        cv_metrics = joint_metrics(cv[eligible, None], target[eligible])
                        row.update({new: cv_metrics[old] for old, new in
                                    [('minJointADE', 'cv_ADE'), ('minJointFDE', 'cv_FDE'), ('minJointMR', 'cv_MR')]})
                        focal_mask = eligible & labels['focal']
                        if focal_mask.any():
                            focal_metrics = joint_metrics(prediction[focal_mask], target[focal_mask])
                            row.update({new: focal_metrics[old] for old, new in
                                        [('minJointADE', 'focal_minADE'), ('minJointFDE', 'focal_minFDE'), ('minJointMR', 'focal_MR')]})
                            focal_count += 1
                        row['status'] = 'evaluated'
                        evaluated += 1; total_targets += row['targets']
                        all_stats.update(row)
                        subset_stats[row['target_subset']].update(row)
                    writer.writerow(row); stream.flush()
                    if (index + 1) % 10 == 0 or index + 1 == len(ids):
                        print('%d/%d scenes; eligible=%d, targets=%d, elapsed=%.1fs' %
                              (index + 1, len(ids), evaluated, total_targets, time.monotonic() - start), flush=True)
        unchanged = all(torch.equal(value.cpu(), initial[key]) for key, value in model.state_dict().items())
        if not unchanged:
            raise RuntimeError('Model state changed during evaluation')
        all_summary = all_stats.summary()
        metrics = all_summary['metrics']
        assert (evaluated, total_targets, focal_count) == (
            all_summary['evaluated_scenes'], all_summary['evaluated_targets'], all_summary['focal_vehicle_scenes'])
        metadata.update(status='complete', evaluated_scenes=evaluated, skipped_scenes=skipped,
                        evaluated_targets=total_targets, focal_vehicle_scenes=focal_count,
                        model_state_unchanged=True, seconds=time.monotonic() - start, metrics=metrics,
                        marginal_actor_weighted=all_summary['marginal_actor_weighted'],
                        subsets={key: value.summary() for key, value in subset_stats.items()})
        (output / 'summary.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
        print(json.dumps(dict(metrics=metrics, evaluated_scenes=evaluated, output=str(output)), indent=2))
    except Exception as error:
        metadata.update(status='failed', error=repr(error), evaluated_scenes=evaluated)
        raise
    finally:
        (output / 'run.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
