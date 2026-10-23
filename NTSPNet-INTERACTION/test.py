from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import INTERACTIONDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from model import PNet

from transforms import LaneAssociationTransform
from transforms import LaneRandomOcclusion

if __name__ == '__main__':
    pl.seed_everything(1024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    PNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = PNet.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    trainer = pl.Trainer(devices=args.devices, accelerator='gpu')
    test_dataset = INTERACTIONDataset(
        root=args.root,
        split='test',
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
    dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,num_workers=args.num_workers, pin_memory=args.pin_memory,persistent_workers=args.persistent_workers)
    trainer.test(model, dataloader)
