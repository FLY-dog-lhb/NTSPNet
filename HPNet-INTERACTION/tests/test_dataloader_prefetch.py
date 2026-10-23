import pytest

from datamodules import INTERACTIONDataModule
from transforms import AgentRandomOcclusion
from transforms import HorizontalFlip
from transforms import LaneAssociationTransform
from transforms import LaneRandomOcclusion
from transforms import V22MapGeometryTransform


def make_datamodule(**overrides):
    configuration = dict(
        root='/tmp/not-accessed-by-constructor',
        train_batch_size=2,
        val_batch_size=2,
        num_workers=4,
        prefetch_factor=3,
    )
    configuration.update(overrides)
    return INTERACTIONDataModule(**configuration)


def test_worker_loader_kwargs_include_prefetch_factor():
    datamodule = make_datamodule()

    assert datamodule._loader_worker_kwargs() == {
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 3,
    }


def test_single_process_loader_omits_prefetch_and_persistence():
    datamodule = make_datamodule(num_workers=0)

    assert datamodule._loader_worker_kwargs() == {
        'num_workers': 0,
        'pin_memory': True,
        'persistent_workers': False,
    }


def test_prefetch_factor_must_be_positive():
    with pytest.raises(ValueError, match='must be positive'):
        make_datamodule(prefetch_factor=0)


def test_teacher_cache_rejects_agent_occlusion():
    with pytest.raises(ValueError, match='agent_occlusion_ratio=0'):
        make_datamodule(
            teacher_trajectory_cache_dir='/tmp/teacher-cache',
            agent_occlusion_ratio=0.1,
        )


def test_v22_map_geometry_precedes_augmentation_and_association(tmp_path):
    datamodule = make_datamodule(
        v22_map_geometry_cache_dir=str(tmp_path),
    )

    train_transform_types = [
        type(transform)
        for transform in datamodule.train_transform.transforms
    ]
    val_transform_types = [
        type(transform)
        for transform in datamodule.val_transform.transforms
    ]

    assert train_transform_types == [
        V22MapGeometryTransform,
        HorizontalFlip,
        AgentRandomOcclusion,
        LaneRandomOcclusion,
        LaneAssociationTransform,
    ]
    assert val_transform_types == [
        V22MapGeometryTransform,
        LaneRandomOcclusion,
        LaneAssociationTransform,
    ]
