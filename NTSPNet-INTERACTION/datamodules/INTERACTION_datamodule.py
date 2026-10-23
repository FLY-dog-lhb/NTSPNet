from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from datasets import INTERACTIONDataset

from transforms import HorizontalFlip
from transforms import AgentRandomOcclusion
from transforms import LaneRandomOcclusion
from transforms import LaneAssociationTransform
from transforms import TeacherTrajectoryTransform
from transforms import V22MapGeometryTransform


def subset_from_manifest(dataset, manifest_path: Optional[str]):
    if manifest_path is None:
        return dataset
    manifest_names = [
        line.strip()
        for line in Path(manifest_path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]
    if not manifest_names:
        raise ValueError(f'manifest contains no sample filenames: {manifest_path}')
    if len(set(manifest_names)) != len(manifest_names):
        raise ValueError(f'manifest contains duplicate filenames: {manifest_path}')
    name_to_index = {
        name: index
        for index, name in enumerate(dataset.processed_file_names)
    }
    missing_names = [
        name for name in manifest_names if name not in name_to_index
    ]
    if missing_names:
        preview = ', '.join(missing_names[:5])
        raise ValueError(
            f'{len(missing_names)} manifest samples are absent from the '
            f'Dataset; first entries: {preview}'
        )
    return Subset(dataset, [name_to_index[name] for name in manifest_names])


class INTERACTIONDataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 prefetch_factor: int = 2,
                 flip_p: float = 0.5,
                 agent_occlusion_ratio: float = 0.0,
                 lane_occlusion_ratio: float = 0.2,
                 num_historical_steps: int = 10,
                 num_future_steps: int = 30,
                 observation_step_time: float = 0.1,
                 motion_source: str = 'causal_difference',
                 csv_motion_cache_dir: Optional[str] = None,
                 teacher_trajectory_cache_dir: Optional[str] = None,
                 v22_map_geometry_cache_dir: Optional[str] = None,
                 train_manifest: Optional[str] = None,
                 val_manifest: Optional[str] = None,
                 **kwargs) -> None:
        super(INTERACTIONDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        if prefetch_factor <= 0:
            raise ValueError('prefetch_factor must be positive')
        self.prefetch_factor = prefetch_factor
        if teacher_trajectory_cache_dir is not None and agent_occlusion_ratio != 0:
            raise ValueError(
                'teacher trajectory distillation requires '
                'agent_occlusion_ratio=0 so cached teacher inputs remain aligned'
            )
        train_transforms = []
        if v22_map_geometry_cache_dir is not None:
            train_transforms.append(V22MapGeometryTransform(
                v22_map_geometry_cache_dir,
            ))
        if teacher_trajectory_cache_dir is not None:
            train_transforms.append(TeacherTrajectoryTransform(
                teacher_trajectory_cache_dir,
                split='train',
            ))
        train_transforms.extend([
            HorizontalFlip(flip_p),
            AgentRandomOcclusion(
                agent_occlusion_ratio,
                num_historical_steps,
                observation_step_time,
            ),
            LaneRandomOcclusion(lane_occlusion_ratio),
            LaneAssociationTransform(
                num_historical_steps,
                observation_step_time,
            ),
        ])
        self.train_transform = Compose(train_transforms)
        val_transforms = []
        if v22_map_geometry_cache_dir is not None:
            val_transforms.append(V22MapGeometryTransform(
                v22_map_geometry_cache_dir,
            ))
        val_transforms.extend([
            LaneRandomOcclusion(0.0),
            LaneAssociationTransform(
                num_historical_steps,
                observation_step_time,
            ),
        ])
        self.val_transform = Compose(val_transforms)
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.observation_step_time = observation_step_time
        self.motion_source = motion_source
        self.csv_motion_cache_dir = csv_motion_cache_dir
        self.teacher_trajectory_cache_dir = teacher_trajectory_cache_dir
        self.v22_map_geometry_cache_dir = v22_map_geometry_cache_dir
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest

    def prepare_data(self) -> None:
        INTERACTIONDataset(
            self.root, 'train', self.train_transform,
            self.num_historical_steps, self.num_future_steps,
            self.observation_step_time, self.motion_source,
            self.csv_motion_cache_dir,
        )
        INTERACTIONDataset(
            self.root, 'val', self.val_transform,
            self.num_historical_steps, self.num_future_steps,
            self.observation_step_time, self.motion_source,
            self.csv_motion_cache_dir,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        train_dataset = INTERACTIONDataset(
            self.root, 'train', self.train_transform,
            self.num_historical_steps, self.num_future_steps,
            self.observation_step_time, self.motion_source,
            self.csv_motion_cache_dir,
        )
        val_dataset = INTERACTIONDataset(
            self.root, 'val', self.val_transform,
            self.num_historical_steps, self.num_future_steps,
            self.observation_step_time, self.motion_source,
            self.csv_motion_cache_dir,
        )
        self.train_dataset = subset_from_manifest(
            train_dataset, self.train_manifest,
        )
        self.val_dataset = subset_from_manifest(
            val_dataset, self.val_manifest,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            **self._loader_worker_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            **self._loader_worker_kwargs(),
        )

    def _loader_worker_kwargs(self):
        kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
        }
        if self.num_workers > 0:
            kwargs['prefetch_factor'] = self.prefetch_factor
        return kwargs
