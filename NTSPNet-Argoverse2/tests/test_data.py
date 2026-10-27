import json
import torch
from torch_geometric.data import Batch
from datasets.argoverse2_dataset import build_scene, build_map, Argoverse2Dataset
from modules.lane_reachability import load_lane_association_from_data


def test_history_ids_masks_and_missing_frames(local_frame, map_data):
    graph = build_scene(local_frame.sample(frac=1, random_state=4), map_data, 'train', 'scene-a')
    assert graph.track_ids == ['00042', '99', 'ped-1']
    assert graph['agent'].position.shape == (3, 110, 2)
    assert not graph['agent'].visible_mask[2, 25:27].any()
    assert not graph['agent'].csv_motion_valid_mask[:, 50:].any()
    assert not graph['agent'].csv_velocity[:, 50:].any()
    assert graph['agent'].focal_mask.tolist() == [True, False, False]
    assert torch.equal(graph['agent'].position[0, 49], torch.zeros(2))
    assert not graph['agent'].size_valid.any()


def test_map_neighbors_and_conservative_lane_change(map_data):
    graph = build_map(map_data, [0., 0.])
    assert graph['lane', 'lane'].adjacent_right_edge_index.tolist() == [[0], [1]]
    assert graph['lane', 'lane'].right_neighbor_edge_index.numel() == 0
    assert graph['lane', 'lane'].left_neighbor_edge_index.tolist() == [[1], [0]]
    assert set(graph['polyline'].side.tolist()) == {0, 1, 2}
    assert graph['lane'].speed_limit.sum() == 0


def test_local_lane_indices_offset_in_batch(graph):
    batch = Batch.from_data_list([graph, graph.clone()])
    result = load_lane_association_from_data(batch, 50)
    n = graph['agent'].num_nodes; lanes = graph['lane'].num_nodes
    original = result.main_lane[:n]; second = result.main_lane[n:]
    assert torch.equal(second, torch.where(original >= 0, original + lanes, original))
    edge = batch['lane', 'lane'].lane_change_pair_index
    assert (batch['lane'].batch[edge[0]] == batch['lane'].batch[edge[1]]).all()


def test_blind_test_and_invalid_future(local_frame, map_data):
    import pytest
    test = local_frame[local_frame.observed]
    graph = build_scene(test, map_data, 'test', 'scene-a')
    assert not graph['agent'].visible_mask[:, 50:].any()
    with pytest.raises(ValueError, match='observations only'):
        build_scene(local_frame, map_data, 'test', 'scene-a')


def test_cache_roundtrip_and_raw_change_invalidation(tmp_path, local_frame, map_data):
    raw = tmp_path / 'raw'; scene = raw / 'train' / 'scene-a'; scene.mkdir(parents=True)
    parquet = scene / 'scenario_scene-a.parquet'
    local_frame.to_parquet(parquet)
    (scene / 'log_map_archive_scene-a.json').write_text(json.dumps(map_data))
    dataset = Argoverse2Dataset(raw, 'train', tmp_path / 'cache')
    first = dataset[0]; second = dataset[0]
    assert torch.equal(first['agent'].position, second['agent'].position)
    local_frame.loc[(local_frame.track_id == '99') & (local_frame.timestep == 0), 'position_x'] += 2
    local_frame.to_parquet(parquet)
    updated = dataset[0]
    assert not torch.equal(first['agent'].position, updated['agent'].position)


def test_future_rows_cannot_change_input_nodes_or_geometry(local_frame, map_data):
    reference = build_scene(local_frame, map_data, 'train', 'scene-a')
    changed = local_frame.copy()
    changed.loc[~changed.observed, ['position_x', 'position_y', 'velocity_x', 'heading']] += 9999
    altered = build_scene(changed, map_data, 'train', 'scene-a')
    assert reference.track_ids == altered.track_ids
    for field in ('csv_velocity', 'csv_heading', 'lane_association_main_lane_local'):
        assert torch.equal(reference['agent'][field], altered['agent'][field])
    assert torch.equal(reference['lane'].position, altered['lane'].position)
    assert torch.equal(reference.origin, altered.origin)


def test_manifest_duplicates_and_raw_cache_rejected(tmp_path):
    import pytest
    raw = tmp_path / 'raw'; (raw / 'train' / 'scene-a').mkdir(parents=True)
    manifest = tmp_path / 'ids.txt'; manifest.write_text('scene-a\nscene-a\n')
    with pytest.raises(ValueError, match='Duplicate scene IDs'):
        Argoverse2Dataset(raw, 'train', manifest=manifest)
    with pytest.raises(ValueError, match='outside the raw'):
        Argoverse2Dataset(raw, 'train', cache_dir=raw / 'cache')
