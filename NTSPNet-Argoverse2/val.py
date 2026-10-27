from argparse import ArgumentParser
from pathlib import Path
import json
import pytorch_lightning as pl
from cli import add_data_args, datamodule
from model import PNet


def main():
    parser = ArgumentParser(description='Evaluate an AV2 checkpoint on validation ground truth')
    add_data_args(parser)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--accelerator', choices=('auto', 'cpu', 'gpu'), default='auto')
    parser.add_argument('--output', help='Optional JSON metrics file (must not exist)')
    args = parser.parse_args()
    if args.output and (Path(args.output).resolve() == Path(args.root).resolve() or Path(args.root).resolve() in Path(args.output).resolve().parents):
        parser.error('Keep metrics outside the raw dataset directory')
    if args.output and Path(args.output).exists():
        parser.error('Output already exists')
    dm = datamodule(args)
    model = PNet.restore(args.checkpoint)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=1, logger=False, enable_checkpointing=False)
    result = trainer.validate(model, datamodule=dm)[0]
    payload = dict(checkpoint=str(Path(args.checkpoint).resolve()), protocol=model.hparams.protocol,
                   scenarios=len(dm.val_dataset), metrics=result)
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
