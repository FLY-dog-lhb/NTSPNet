import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cross_domain.data import build_input
from cross_domain.test_annotations import MultiWorldTestAnnotations
from tests.test_cross_domain import assert_graph_equal


@pytest.fixture
def annotation_rows(local_frame):
    rows = []
    for key, group in local_frame[local_frame.object_category >= 2].groupby('track_id'):
        future = group[group.timestep >= 50].sort_values('timestep')
        rows.append(dict(scenario_id='scene-a', track_id=key, is_focal_track=key == '00042',
                         gt_trajectory_x=future.position_x.to_numpy(),
                         gt_trajectory_y=future.position_y.to_numpy()))
    return pd.DataFrame(rows)


def reader(tmp_path, rows):
    path = tmp_path / 'labels.parquet'; rows.to_parquet(path, index=False)
    return MultiWorldTestAnnotations(path, ['scene-a'])


def test_external_labels_match_val_and_do_not_change_graph(tmp_path, annotation_rows, local_frame, map_data):
    annotations = reader(tmp_path, annotation_rows)
    history = local_frame[local_frame.observed]
    future = annotations.future_frame('scene-a', history)
    full_graph, full_labels = build_input(local_frame, map_data, 'scene-a')
    graph, labels = build_input(history, map_data, 'scene-a', future_frame=future)
    blind_graph, blind_labels = build_input(history, map_data, 'scene-a')
    assert_graph_equal(full_graph, graph)
    assert_graph_equal(blind_graph, graph)
    assert not blind_labels['eligible'].any()
    assert torch.equal(labels['eligible'], full_labels['eligible'])
    mask = labels['eligible']
    assert torch.equal(labels['position'][mask], full_labels['position'][mask])
    assert labels['position'][0, 0, 0] == .5  # future index0 maps to AV2 timestep50
    altered = future.copy(); altered[['position_x', 'position_y']] += 10000.
    other, other_labels = build_input(history, map_data, 'scene-a', future_frame=altered)
    assert_graph_equal(graph, other)
    assert not torch.equal(labels['position'], other_labels['position'])
    assert '00042' in labels['track_ids']  # preserve string IDs, including leading zeros


def test_refuse_missing_scenario_and_duplicate_annotations(tmp_path, annotation_rows):
    path = tmp_path / 'labels.parquet'; annotation_rows.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='cover'):
        MultiWorldTestAnnotations(path, ['unknown'])
    with pytest.raises(ValueError, match='Duplicate'):
        reader(tmp_path, pd.concat([annotation_rows, annotation_rows]))


def test_refuse_wrong_targets_focal_or_blind_input(tmp_path, annotation_rows, local_frame):
    history = local_frame[local_frame.observed]
    annotations = reader(tmp_path, annotation_rows.iloc[:1])
    with pytest.raises(ValueError, match='track IDs'):
        annotations.future_frame('scene-a', history)
    rows = annotation_rows.copy(); rows.is_focal_track = ~rows.is_focal_track
    with pytest.raises(ValueError, match='focal'):
        reader(tmp_path, rows).future_frame('scene-a', history)
    with pytest.raises(ValueError, match='observations only'):
        reader(tmp_path, annotation_rows).future_frame('scene-a', local_frame)


@pytest.mark.parametrize('values', [np.zeros(59), np.full(60, np.nan)])
def test_refuse_invalid_future_shape_or_nan(tmp_path, annotation_rows, local_frame, values):
    rows = annotation_rows.copy(); rows.at[0, 'gt_trajectory_x'] = values
    with pytest.raises(ValueError, match='60 finite'):
        reader(tmp_path, rows).future_frame('scene-a', local_frame[local_frame.observed])


def test_external_labels_cannot_supply_history(local_frame, map_data):
    future = local_frame[local_frame.timestep.between(49, 79)]
    with pytest.raises(ValueError, match='unique future'):
        build_input(local_frame[local_frame.observed], map_data, 'scene-a', future_frame=future)


def test_cli_requires_labels_for_test(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'evaluate_interaction.py'
    result = subprocess.run([sys.executable, str(script), '--split', 'test', '--checkpoint', 'unused.ckpt',
                             '--output-dir', str(tmp_path/'output')], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'requires --test-annotations' in result.stderr
    assert not (tmp_path/'output').exists()
