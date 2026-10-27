"""Regression tests for cross-domain protocol boundaries and metric semantics."""
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from cross_domain.data import build_input
from cross_domain.runtime import activate_source, joint_metrics


def assert_graph_equal(left, right):
    assert left.node_types == right.node_types
    assert left.edge_types == right.edge_types
    for a, b in zip(left.stores, right.stores):
        assert set(a.keys()) == set(b.keys())
        for key in a.keys():
            if torch.is_tensor(a[key]):
                assert torch.equal(a[key], b[key]), key
            else:
                assert a[key] == b[key], key


def test_all_graph_fields_independent_of_future_and_discarded_history(local_frame, map_data):
    graph, labels = build_input(local_frame, map_data, 'scene')
    changed = local_frame.copy()
    outside = ~changed.timestep.between(40, 49)
    changed.loc[outside, ['position_x', 'position_y', 'velocity_x', 'velocity_y', 'heading']] += 12345.
    changed.loc[outside, 'object_category'] = 0
    changed.loc[outside, 'object_type'] = 'static'
    other, altered_labels = build_input(changed, map_data, 'scene')
    assert_graph_equal(graph, other)
    assert not torch.equal(labels['position'], altered_labels['position'])
    assert labels['track_ids'] == ['00042', '99', 'ped-1']
    assert labels['eligible'].tolist() == [True, True, False]
    agent = graph['agent']
    assert agent.type.tolist() == [0, 0, 1]
    for key in ('position', 'visible_mask', 'csv_velocity', 'csv_heading', 'csv_motion_valid_mask'):
        assert not agent[key][:, 10:].any(), key
    assert agent.position.shape == (3, 40, 2)
    assert not agent.length.any() and not agent.width.any()


def test_missing_history_or_future_excludes_only_metric_target(local_frame, map_data):
    base, labels = build_input(local_frame, map_data, 'scene')
    missing_future = local_frame[~((local_frame.track_id == '99') & (local_frame.timestep == 79))]
    graph, labels = build_input(missing_future, map_data, 'scene')
    assert_graph_equal(base, graph)
    assert labels['eligible'].tolist() == [True, False, False]
    missing_history = local_frame[~((local_frame.track_id == '99') & (local_frame.timestep == 40))]
    graph, labels = build_input(missing_history, map_data, 'scene')
    assert graph['agent'].num_nodes == 3
    assert labels['eligible'].tolist() == [True, False, False]


def test_context_must_be_observed_in_selected_window(local_frame, map_data):
    removed = local_frame[~((local_frame.track_id == 'ped-1') & local_frame.timestep.between(40, 49))]
    graph, labels = build_input(removed, map_data, 'scene')
    assert labels['track_ids'] == ['00042', '99']
    assert graph['agent'].num_nodes == 2


def test_source_lane_reference_point(local_frame, map_data):
    graph, _ = build_input(local_frame, map_data, 'scene')
    lane = map_data['lane_segments']['0']
    center = lane['centerline']
    # Focal origin x=24.5; middle-index center instead of endpoints' midpoint.
    assert graph['lane'].position[0, 0].item() == pytest.approx(center[3]['x'] - 24.5)


def test_joint_metrics_independent_oracles_and_worst_agent_miss():
    target = torch.zeros(2, 30, 2)
    pred = torch.zeros(2, 2, 30, 2)
    pred[:, 0, :-1, 0] = 0.
    pred[:, 0, -1, 0] = 4.
    pred[:, 1, :, 0] = 3.
    pred[:, 1, -1, 0] = 1.
    score = joint_metrics(pred, target)
    assert score['minJointADE'] == pytest.approx(4 / 30)
    assert score['minJointFDE'] == 1.
    assert score['minJointMR'] == 0.
    pred[0, 0, -1, 0] = 0.; pred[1, 0, -1, 0] = 3.
    pred[0, 1, -1, 0] = 3.; pred[1, 1, -1, 0] = 0.
    score = joint_metrics(pred, target)
    assert score['minJointFDE'] == 1.5
    assert score['minJointMR'] == 1.  # per-agent oracle would incorrectly be zero


def test_refuse_mixed_av2_and_interaction_modules():
    with pytest.raises(RuntimeError, match='fresh process'):
        activate_source(Path(__file__).resolve().parents[2] / 'HPNet-INTERACTION')


def test_metrics_match_live_interaction_implementation():
    root = Path(__file__).resolve().parents[1]
    source = root.parent / 'HPNet-INTERACTION'
    if not source.is_dir():
        pytest.skip('Source INTERACTION project unavailable')
    code = '''
import sys
from cross_domain.runtime import activate_source, joint_metrics
activate_source(sys.argv[1])
from metrics import minJointADE, minJointFDE, minJointMR
import torch
torch.set_num_threads(2)
torch.manual_seed(2048)
for n in (1, 3, 7):
    prediction = torch.randn(n, 6, 30, 2) * 2
    target = torch.randn(n, 30, 2)
    actual = joint_metrics(prediction, target)
    for cls in (minJointADE, minJointFDE, minJointMR):
        metric = cls()
        metric.update([prediction], [target], [torch.ones(n, 30, dtype=torch.bool)])
        assert abs(actual[cls.__name__] - float(metric.compute())) < 1e-6
'''
    result = subprocess.run([sys.executable, '-c', code, str(source)], cwd=str(root),
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
