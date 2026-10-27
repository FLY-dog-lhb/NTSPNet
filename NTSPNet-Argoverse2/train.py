from argparse import ArgumentParser
from pathlib import Path
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from cli import add_data_args, add_model_args, datamodule, model_arguments, PROJECT
from model import PNet


def main():
    parser = ArgumentParser(description='Train PNet on Argoverse 2 motion forecasting')
    add_data_args(parser); add_model_args(parser)
    parser.add_argument('--output-dir', default=str(PROJECT / 'runs'))
    parser.add_argument('--seed', type=int, default=2048)
    parser.add_argument('--accelerator', choices=('auto', 'cpu', 'gpu'), default='auto')
    parser.add_argument('--devices', type=int, choices=(1,), default=1, help='Single-device training and exact scene-weighted validation')
    parser.add_argument('--accumulate-grad-batches', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--statistics', help='Train-only statistics; buffers are embedded in checkpoints')
    parser.add_argument('--resume', help='Full optimizer/scheduler/epoch resume')
    parser.add_argument('--weights', help='Strict AV2 weights only; starts fresh optimizer')
    parser.add_argument('--base-checkpoint')
    parser.add_argument('--provider-checkpoint')
    args = parser.parse_args()
    dm = datamodule(args); pl.seed_everything(args.seed, workers=True)
    if args.provider_checkpoint and (args.trunk != 'fusion' or not args.base_checkpoint or args.resume or args.weights):
        parser.error('--provider-checkpoint requires fresh fusion initialization with --base-checkpoint')
    if args.resume and args.statistics:
        parser.error('Resume uses checkpoint calibration buffers; do not replace them with --statistics')
    output_dir = Path(args.output_dir).resolve()
    raw_root = Path(args.root).resolve()
    if output_dir == raw_root or raw_root in output_dir.parents:
        parser.error('Keep training outputs outside the raw dataset directory')
    if sum(bool(x) for x in (args.resume, args.weights, args.base_checkpoint)) > 1:
        parser.error('Choose resume, weights, or base-checkpoint initialization')
    if args.resume or args.weights:
        overrides = {}
        if args.weights:
            overrides['max_epochs'] = args.max_epochs
            overrides.update(lr=args.lr, weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs)
        model = PNet.restore(args.resume or args.weights, **overrides)
    else:
        model = PNet(**model_arguments(args))
        if args.trunk == 'fusion':
            if not args.base_checkpoint or not args.provider_checkpoint:
                parser.error('Fusion needs --base-checkpoint and --provider-checkpoint trained on AV2')
            model.initialize_fusion(args.base_checkpoint, args.provider_checkpoint)
        elif args.base_checkpoint:
            model.initialize_provider(args.base_checkpoint)
        elif args.freeze_map:
            parser.error('Cannot freeze a random MapEncoder; provide --base-checkpoint')
    if args.statistics:
        if model.hparams.trunk != 'extended':
            parser.error('--statistics applies to extended provider training')
        model.load_style_statistics(args.statistics)
    if model.hparams.trunk == 'extended' and model.hparams.lambda_style:
        model.Backbone.require_style_statistics()
    logger = CSVLogger(args.output_dir, name=model.hparams.trunk)
    logger.experiment  # Initialize the logger before creating its run metadata.
    Path(logger.log_dir).mkdir(parents=True, exist_ok=True)
    (Path(logger.log_dir) / 'run_config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    checkpoint = ModelCheckpoint(monitor='val_score', mode='min', save_top_k=1,
                                 save_last=True, filename='epoch={epoch:03d}-fde={val_score:.4f}',
                                 auto_insert_metric_name=False)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         max_epochs=args.max_epochs, max_steps=args.max_steps,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         gradient_clip_val=5., logger=logger,
                         callbacks=[checkpoint, LearningRateMonitor(logging_interval='epoch')],
                         num_sanity_val_steps=1, log_every_n_steps=1,
                         strategy='ddp_find_unused_parameters_true' if args.devices > 1 else 'auto')
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    print('LAST_CHECKPOINT=' + checkpoint.last_model_path)
    print('BEST_CHECKPOINT=' + checkpoint.best_model_path)


if __name__ == '__main__':
    main()
