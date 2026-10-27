import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets.argoverse2_dataset import Argoverse2Dataset


class Argoverse2DataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=1, num_workers=4, cache_dir=None,
                 reader='parquet', train_manifest=None, val_manifest=None,
                 test_manifest=None, train_limit=None, val_limit=None, test_limit=None):
        super().__init__()
        self.root = root; self.batch_size = batch_size; self.num_workers = num_workers
        self.cache_dir = cache_dir; self.reader = reader
        self.manifests = dict(train=train_manifest, val=val_manifest, test=test_manifest)
        self.limits = dict(train=train_limit, val=val_limit, test=test_limit)

    def setup(self, stage=None):
        splits = ('train', 'val') if stage == 'fit' else ('val',) if stage == 'validate' else ('test',) if stage in ('test', 'predict') else ('train', 'val', 'test')
        for split in splits:
            setattr(self, split + '_dataset', Argoverse2Dataset(
                self.root, split, self.cache_dir, self.manifests[split], self.limits[split], self.reader))

    def loader(self, split):
        return DataLoader(getattr(self, split + '_dataset'), batch_size=self.batch_size,
                          shuffle=split == 'train', num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self.loader('train')

    def val_dataloader(self):
        return self.loader('val')

    def test_dataloader(self):
        return self.loader('test')

    def predict_dataloader(self):
        return self.loader('test')
