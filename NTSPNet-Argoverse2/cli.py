"""Shared command-line arguments. Cache/output always live outside raw data."""
from pathlib import Path
import torch
from datamodules import Argoverse2DataModule

PROJECT = Path(__file__).resolve().parent


def add_data_args(parser):
    parser.add_argument('--root', default='/home/lhb/data-Argoverse2')
    parser.add_argument('--cache-dir', default=str(PROJECT / 'cache'))
    parser.add_argument('--reader', choices=('parquet', 'av2'), default='parquet')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    for split in ('train', 'val', 'test'):
        parser.add_argument('--' + split + '-manifest')
        parser.add_argument('--' + split + '-limit', type=int)


def datamodule(args):
    if args.batch_size < 1 or args.num_workers < 0 or args.threads < 1:
        raise ValueError('Invalid batch size, workers or threads')
    torch.set_num_threads(args.threads)
    names = ('root', 'cache_dir', 'reader', 'batch_size', 'num_workers',
             'train_manifest', 'val_manifest', 'test_manifest', 'train_limit', 'val_limit', 'test_limit')
    return Argoverse2DataModule(**{name: getattr(args, name) for name in names})


def add_model_args(parser):
    parser.add_argument('--trunk', choices=('baseline', 'extended', 'fusion'), default='extended')
    parser.add_argument('--protocol', choices=('single-agent', 'multi-world'), default='single-agent')
    for name, default in (('hidden-dim', 128), ('num-modes', 6), ('num-attn-layers', 3),
                          ('num-heads', 8), ('num-hops', 4), ('duration', 10), ('warmup-epochs', 4), ('max-epochs', 64)):
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in (('a2a-radius', 80.), ('l2a-radius', 80.), ('dropout', .1),
                          ('lr', 5e-4), ('weight-decay', 1e-4), ('lambda-hist-aux', .3),
                          ('lambda-style', .05), ('lambda-priority', .3)):
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--freeze-map', action='store_true')


def model_arguments(args):
    names = ('trunk', 'protocol', 'hidden_dim', 'num_modes', 'num_attn_layers', 'num_heads',
             'num_hops', 'duration', 'warmup_epochs', 'max_epochs', 'a2a_radius', 'l2a_radius',
             'dropout', 'lr', 'weight_decay', 'lambda_hist_aux', 'lambda_style', 'lambda_priority', 'freeze_map')
    return {name: getattr(args, name) for name in names}
