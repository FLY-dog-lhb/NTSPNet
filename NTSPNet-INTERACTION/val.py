from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl

from datasets import INTERACTIONDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from model import PNet

from transforms import LaneAssociationTransform
from transforms import LaneRandomOcclusion
from transforms import V22MapGeometryTransform


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ('1', 'true', 'yes', 'y'):
        return True
    if normalized in ('0', 'false', 'no', 'n'):
        return False
    raise ValueError(f'invalid boolean value: {value}')


def resolve_v22_map_geometry_cache_dir(cli_value, model):
    checkpoint_value = model.hparams.get('v22_map_geometry_cache_dir')
    selected = cli_value if cli_value is not None else checkpoint_value
    if selected is None:
        return None
    cache_dir = Path(selected).expanduser().resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            'v22 map geometry cache does not exist: '
            f'{cache_dir}. Pass --v22_map_geometry_cache_dir when the '
            'checkpoint was moved to another machine.'
        )
    return cache_dir


def print_validation_metrics(results, scenario_count):
    if len(results) != 1:
        raise RuntimeError(
            f'expected one validation result, received {len(results)}'
        )
    metrics = results[0]
    required = ('val_minJointADE', 'val_minJointFDE', 'val_minJointMR')
    missing = [name for name in required if name not in metrics]
    if missing:
        raise RuntimeError(
            'validation did not return required metrics: '
            + ', '.join(missing)
        )
    print('\nValidation complete. Headline metrics:')
    print(
        f'scenarios={scenario_count} '
        f'val_minJointADE={float(metrics["val_minJointADE"]):.8f} '
        f'val_minJointFDE={float(metrics["val_minJointFDE"]):.8f} '
        f'val_minJointMR={float(metrics["val_minJointMR"]):.8f}'
    )

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=parse_bool, default=True)
    parser.add_argument('--persistent_workers', type=parse_bool, default=True)
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument(
        '--csv_motion_cache_dir',
        type=str,
        default=None,
        help=(
            'Optional override for the CSV motion cache path stored in the '
            'checkpoint.'
        ),
    )
    parser.add_argument(
        '--v22_map_geometry_cache_dir',
        type=str,
        default=None,
        help=(
            'Optional override for the v22 map geometry cache path stored '
            'in the checkpoint.'
        ),
    )
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predicted trajectories to val_output/ for visualization')
    args = parser.parse_args()

    checkpoint_overrides = {}
    if args.csv_motion_cache_dir is not None:
        csv_motion_cache_dir = Path(
            args.csv_motion_cache_dir,
        ).expanduser().resolve()
        if not csv_motion_cache_dir.is_dir():
            parser.error(
                f'CSV motion cache does not exist: {csv_motion_cache_dir}'
            )
        checkpoint_overrides['csv_motion_cache_dir'] = str(
            csv_motion_cache_dir
        )
    model = PNet.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location='cpu',
        strict=True,
        **checkpoint_overrides,
    )
    if args.save_predictions:
        model.save_val_predictions = True
    try:
        v22_map_geometry_cache_dir = resolve_v22_map_geometry_cache_dir(
            args.v22_map_geometry_cache_dir,
            model,
        )
    except FileNotFoundError as error:
        parser.error(str(error))
    transforms = []
    if v22_map_geometry_cache_dir is not None:
        transforms.append(V22MapGeometryTransform(
            v22_map_geometry_cache_dir,
        ))
    transforms.extend([
        LaneRandomOcclusion(0.0),
        LaneAssociationTransform(
            model.num_historical_steps,
            model.observation_step_time,
        ),
    ])
    trainer = pl.Trainer(
        devices=args.devices,
        accelerator='gpu',
        logger=False,
        enable_checkpointing=False,
    )
    val_dataset = INTERACTIONDataset(
        root=args.root,
        split='val',
        transform=Compose(transforms),
        num_historical_steps=model.num_historical_steps,
        num_future_steps=model.num_future_steps,
        observation_step_time=model.observation_step_time,
        motion_source=model.motion_source,
        csv_motion_cache_dir=model.csv_motion_cache_dir,
    )
    dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(
            args.persistent_workers and args.num_workers > 0
        ),
    )
    results = trainer.validate(model, dataloader)
    print_validation_metrics(results, len(val_dataset))
