from argparse import ArgumentParser
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import INTERACTIONDataModule
from model import PNet

import torch
#torch.use_deterministic_algorithms(True)
   

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--seed',
        type=int,
        default=1024,
        help='Random seed for model initialization, augmentation and workers.',
    )
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument(
        '--prefetch_factor',
        type=int,
        default=2,
        help='Batches prefetched by each DataLoader worker.',
    )
    parser.add_argument('--flip_p', type=float, default=0.5)
    parser.add_argument('--agent_occlusion_ratio', type=float, default=0.0)
    parser.add_argument('--lane_occlusion_ratio', type=float, default=0.2)
    parser.add_argument(
        '--teacher_trajectory_cache_dir',
        type=str,
        default=None,
        help=(
            'Optional cache root containing train/<sample>.pt trajectories '
            'from compute_teacher_trajectory_cache.py.'
        ),
    )
    parser.add_argument(
        '--v22_map_geometry_cache_dir',
        type=str,
        default=None,
        help=(
            'Optional scenario-level cache of unresampled version-22 map '
            'geometry. Applied before flip, occlusion and lane association.'
        ),
    )
    parser.add_argument(
        '--train_manifest',
        type=str,
        default=None,
        help='Optional stable processed-filename manifest for proxy training.',
    )
    parser.add_argument(
        '--val_manifest',
        type=str,
        default=None,
        help='Optional stable processed-filename manifest for proxy validation.',
    )
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument(
        '--accumulate_grad_batches',
        type=int,
        default=1,
        help='Number of batches to accumulate gradients before optimizer step. '
             'E.g. train_batch_size=2 with accumulate_grad_batches=8 gives effective batch of 16 scenarios per update.',
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        help='Resume training from a Lightning checkpoint (.ckpt). Restores model weights, optimizer state, epoch and step.',
    )
    parser.add_argument(
        '--weights_path',
        type=str,
        default=None,
        help=(
            'Fine-tune from checkpoint model weights only. Starts a new '
            'optimizer, scheduler, epoch and global step. For '
            'original_residual, the checkpoint must use the same trunk '
            'variant and replaces base/residual initialization paths.'
        ),
    )
    parser.add_argument(
        '--base_weights_path',
        type=str,
        default=None,
        help=(
            'Original version-22 checkpoint used only to initialize the '
            'frozen base of trunk_variant=original_residual.'
        ),
    )
    parser.add_argument(
        '--residual_weights_path',
        type=str,
        default=None,
        help=(
            'Mature extended checkpoint used only to initialize the aligned '
            'module-context provider of trunk_variant=original_residual.'
        ),
    )
    PNet.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.seed < 0:
        parser.error('--seed must be non-negative')
    pl.seed_everything(args.seed, workers=True)

    if args.ckpt_path is not None and args.weights_path is not None:
        parser.error('--ckpt_path and --weights_path cannot be used together')

    dual_paths = (args.base_weights_path, args.residual_weights_path)
    if args.trunk_variant == 'original_residual':
        if args.ckpt_path is not None and any(
                path is not None for path in dual_paths):
            parser.error(
                'do not provide base/residual initialization paths when '
                'resuming original_residual with --ckpt_path'
            )
        if args.weights_path is not None and any(
                path is not None for path in dual_paths):
            parser.error(
                'do not provide base/residual initialization paths when '
                'loading a complete original_residual --weights_path'
            )
        if (
            args.ckpt_path is None
            and args.weights_path is None
            and any(path is None for path in dual_paths)
        ):
            parser.error(
                'a new original_residual model requires both '
                '--base_weights_path and --residual_weights_path; a '
                'weights-only continuation may instead provide '
                '--weights_path'
            )
    elif any(path is not None for path in dual_paths):
        parser.error(
            '--base_weights_path/--residual_weights_path require '
            '--trunk_variant original_residual'
        )

    if args.style_statistics_path is None:
        args.style_statistics_path = os.path.join(
            args.root,
            'train_style_statistics.pt',
        )

    if args.weights_path is None:
        model = PNet(**vars(args))
        if args.trunk_variant == 'original_residual' and args.ckpt_path is None:
            model.load_original_residual_weights(
                base_checkpoint=args.base_weights_path,
                residual_checkpoint=args.residual_weights_path,
            )
    else:
        model = PNet.load_from_checkpoint(
            checkpoint_path=args.weights_path,
            map_location='cpu',
            strict=True,
            **vars(args),
        )
    datamodule = INTERACTIONDataModule(**vars(args))
    model_checkpoint = ModelCheckpoint(
        monitor='val_minJointFDE', save_top_k=3, mode='min', save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        devices=args.devices,
        accelerator='gpu',
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)
