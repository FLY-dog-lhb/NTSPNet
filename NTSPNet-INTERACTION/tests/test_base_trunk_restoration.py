import pandas as pd
import torch
from torch_geometric.data import HeteroData

from compute_csv_motion_cache import build_case_motion
from create_proxy_manifests import proportional_allocation
from create_proxy_manifests import stable_select
from datamodules.INTERACTION_datamodule import subset_from_manifest
from datasets.INTERACTION_dataset import INTERACTIONDataset
from modules.backbone import Backbone
from modules.original_backbone import OriginalBackbone
from transforms.horizontal_flip import HorizontalFlip
from utils import compute_historical_kinematics


def make_backbone(**kwargs):
    configuration = dict(
        hidden_dim=8,
        num_historical_steps=3,
        num_future_steps=2,
        duration=1,
        a2a_radius=80.0,
        l2a_radius=80.0,
        num_attn_layers=1,
        num_modes=2,
        num_heads=2,
        dropout=0.0,
        motion_source='csv',
        preserve_base_edges=True,
    )
    configuration.update(kwargs)
    return Backbone(**configuration)


def make_original_backbone(**kwargs):
    configuration = dict(
        hidden_dim=8,
        num_historical_steps=3,
        num_future_steps=2,
        duration=1,
        a2a_radius=80.0,
        l2a_radius=80.0,
        num_attn_layers=1,
        num_modes=2,
        num_heads=2,
        dropout=0.0,
        motion_source='csv',
    )
    configuration.update(kwargs)
    return OriginalBackbone(**configuration)


def test_csv_motion_cache_matches_original_car_and_pedestrian_semantics():
    frame = pd.DataFrame({
        'case_id': [1, 1, 1, 1],
        'track_id': [3, 4, 3, 4],
        'timestamp_ms': [100, 100, 200, 200],
        'agent_type': ['car', 'pedestrian/bicycle', 'car', 'pedestrian/bicycle'],
        'vx': [2.0, 0.0, 3.0, -1.0],
        'vy': [0.0, 1.0, 0.0, 0.0],
        'psi_rad': [0.25, 0.0, 0.5, 0.0],
    })

    payload = build_case_motion(frame, 1, 1)

    assert payload['agent_id'].tolist() == [3, 4]
    assert torch.allclose(
        payload['csv_velocity'][0],
        torch.tensor([[2.0, 0.0], [3.0, 0.0]]),
    )
    assert torch.allclose(
        payload['csv_heading'][0], torch.tensor([0.25, 0.5]),
    )
    assert torch.allclose(
        payload['csv_heading'][1],
        torch.tensor([torch.pi / 2, torch.pi]),
    )


def test_csv_motion_preserves_first_observation_and_respects_visibility():
    data = HeteroData()
    data['agent']['position'] = torch.tensor([[
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
    ]])
    data['agent']['visible_mask'] = torch.tensor([[True, False, True]])
    data['agent']['csv_velocity'] = torch.tensor([[
        [5.0, 0.0], [6.0, 0.0], [7.0, 0.0],
    ]])
    data['agent']['csv_heading'] = torch.tensor([[0.2, 0.3, 0.4]])
    data['agent']['csv_motion_valid_mask'] = torch.ones((1, 3), dtype=torch.bool)
    causal = compute_historical_kinematics(
        data['agent']['position'],
        data['agent']['visible_mask'],
        step_time=0.1,
    )

    velocity, speed, _, _, heading, _, velocity_theta = (
        make_backbone()._select_main_motion(data, causal)
    )

    assert torch.allclose(velocity[0, 0], torch.tensor([5.0, 0.0]))
    assert speed[0, 0].item() == 5.0
    assert heading[0, 0].item() == torch.tensor(0.2).item()
    assert torch.allclose(velocity[0, 1], torch.zeros(2))
    assert heading[0, 1].item() == 0.0
    assert torch.allclose(velocity_theta[0, 0], torch.tensor(-0.2))


def test_original_backbone_restores_version22_feature_dimensions():
    backbone = make_original_backbone()

    assert backbone.a_emb_layer.mlp[0].in_features == 7
    assert backbone.l2m_emb_layer.mlp[0].in_features == 3
    assert backbone.l2n_emb_layer.mlp[0].in_features == 3
    assert backbone.m2m_a_emb_layer.mlp[0].in_features == 5
    assert not hasattr(backbone, 'style_temporal_encoder')
    assert not hasattr(backbone, 'priority_module')
    assert not hasattr(backbone, 'lane_reachability')


def test_original_backbone_uses_raw_csv_velocity_difference_for_acceleration():
    data = HeteroData()
    data['agent']['position'] = torch.zeros((1, 4, 2))
    data['agent']['visible_mask'] = torch.ones((1, 4), dtype=torch.bool)
    data['agent']['csv_motion_valid_mask'] = torch.ones((1, 4), dtype=torch.bool)
    data['agent']['csv_velocity'] = torch.tensor([[
        [1.0, 0.0], [3.0, 0.0], [6.0, 0.0], [10.0, 0.0],
    ]])
    data['agent']['csv_heading'] = torch.zeros((1, 4))

    speed, _, accel_length, accel_theta, heading, full_heading = (
        make_original_backbone()._csv_motion(data)
    )

    assert speed.tolist() == [[1.0, 3.0, 6.0]]
    assert accel_length.tolist() == [[0.0, 2.0, 3.0]]
    assert torch.equal(accel_theta, torch.zeros_like(accel_theta))
    assert heading.shape == (1, 3)
    assert full_heading.shape == (1, 4)


def test_base_vehicle_graph_restores_all_visible_radius_edges():
    backbone = make_backbone()
    position = torch.tensor([
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[20.0, 0.0], [90.0, 0.0], [10.0, 0.0]],
        [[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]],
    ])
    visible = torch.ones((3, 3), dtype=torch.bool)
    batch = torch.tensor([0, 0, 1])
    sparse_edge = torch.tensor([[0], [1]])
    sparse_attr = torch.tensor([[20.0, 2.0, 1.0, 0.0, 0.0, 0.0]])

    edge_index, edge_attr = backbone._build_base_vehicle_edges(
        position, visible, batch, sparse_edge, sparse_attr,
    )

    assert edge_index.tolist() == [[0, 1, 6, 7], [1, 0, 7, 6]]
    assert torch.equal(edge_attr[0], sparse_attr[0])
    assert edge_attr[1, 0].item() == 20.0
    assert torch.equal(edge_attr[1, 1:], torch.zeros(5))


def test_base_lane_graph_keeps_same_scene_visible_nodes_within_radius():
    backbone = make_backbone()
    edge_index = backbone._build_base_lane_edges(
        lane_position=torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 0.0]]),
        lane_batch=torch.tensor([0, 0, 1]),
        lane_visible_mask=torch.ones(3, dtype=torch.bool),
        target_position=torch.tensor([[10.0, 0.0], [10.0, 0.0]]),
        target_batch=torch.tensor([0, 1]),
        target_visible_mask=torch.ones(2, dtype=torch.bool),
    )

    assert edge_index.tolist() == [[0, 2], [0, 1]]


def test_horizontal_flip_updates_csv_motion_without_touching_validity():
    data = HeteroData()
    data['agent']['position'] = torch.tensor([[[1.0, 2.0]]])
    data['agent']['heading'] = torch.tensor([[0.0]])
    data['agent']['csv_velocity'] = torch.tensor([[[3.0, 4.0]]])
    data['agent']['csv_heading'] = torch.tensor([[0.25]])
    data['agent']['csv_motion_valid_mask'] = torch.tensor([[True]])
    data['lane']['position'] = torch.tensor([[0.0, 0.0]])
    data['lane']['heading'] = torch.tensor([0.0])
    data['polyline']['position'] = torch.tensor([[0.0, 0.0]])
    data['polyline']['heading'] = torch.tensor([0.0])
    data['polyline']['side'] = torch.tensor([1], dtype=torch.uint8)
    data['lane', 'lane']['left_neighbor_edge_index'] = torch.empty((2, 0), dtype=torch.long)
    data['lane', 'lane']['right_neighbor_edge_index'] = torch.empty((2, 0), dtype=torch.long)

    flipped = HorizontalFlip(1.0)(data)

    assert torch.allclose(
        flipped['agent']['csv_velocity'], torch.tensor([[[-3.0, 4.0]]]),
    )
    assert torch.allclose(
        flipped['agent']['csv_heading'], torch.tensor([[torch.pi - 0.25]]),
    )
    assert flipped['agent']['csv_motion_valid_mask'].item()


def test_existing_processed_sample_loads_csv_motion_sidecar(tmp_path):
    raw_dir = tmp_path / 'train'
    processed_dir = tmp_path / 'train_processed'
    sidecar_dir = tmp_path / 'motion' / 'train'
    raw_dir.mkdir()
    processed_dir.mkdir()
    sidecar_dir.mkdir(parents=True)
    frame = pd.DataFrame({
        'case_id': [1, 1],
        'track_id': [7, 7],
        'timestamp_ms': [100, 200],
        'agent_type': ['car', 'car'],
        'vx': [2.0, 3.0],
        'vy': [0.0, 0.0],
        'psi_rad': [0.1, 0.2],
    })
    frame.to_csv(raw_dir / 'SCENE_train.csv', index=False)
    torch.save({
        'agent': {
            'num_nodes': 1,
            'id': torch.tensor([7], dtype=torch.uint8),
            'position': torch.zeros((1, 2, 2)),
        },
    }, processed_dir / 'SCENE_1.pt')
    payload = build_case_motion(frame, 1, 1)
    payload['schema_version'] = 1
    torch.save(payload, sidecar_dir / 'SCENE_1.pt')

    dataset = INTERACTIONDataset(
        root=str(tmp_path),
        split='train',
        num_historical_steps=1,
        num_future_steps=1,
        motion_source='csv',
        csv_motion_cache_dir=str(tmp_path / 'motion'),
    )
    data = dataset[0]

    assert torch.allclose(
        data['agent']['csv_velocity'],
        torch.tensor([[[2.0, 0.0], [3.0, 0.0]]]),
    )
    assert torch.allclose(
        data['agent']['csv_heading'], torch.tensor([[0.1, 0.2]]),
    )


def test_proxy_allocation_is_proportional_complete_and_covers_scenarios():
    allocation = proportional_allocation(
        {'large': 80, 'medium': 15, 'small': 5},
        total=20,
    )

    assert sum(allocation.values()) == 20
    assert allocation == {'large': 16, 'medium': 3, 'small': 1}


def test_proxy_selection_is_stable_and_seeded():
    names = [f'sample_{index}.pt' for index in range(100)]

    first = stable_select(names, 10, split='train', seed=1024)
    repeated = stable_select(reversed(names), 10, split='train', seed=1024)
    different_seed = stable_select(names, 10, split='train', seed=2048)

    assert first == repeated
    assert first != different_seed


def test_manifest_subset_maps_stable_names_in_manifest_order(tmp_path):
    class DatasetStub:
        processed_file_names = ['b.pt', 'a.pt', 'c.pt']

        def __len__(self):
            return len(self.processed_file_names)

        def __getitem__(self, index):
            return self.processed_file_names[index]

    manifest = tmp_path / 'proxy.txt'
    manifest.write_text('c.pt\na.pt\n')

    subset = subset_from_manifest(DatasetStub(), str(manifest))

    assert len(subset) == 2
    assert subset[0] == 'c.pt'
    assert subset[1] == 'a.pt'
