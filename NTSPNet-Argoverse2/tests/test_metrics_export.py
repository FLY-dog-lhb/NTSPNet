import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from metrics.forecasting import trajectory_metrics, scene_probabilities
from test import prediction_rows, SUBMISSION_SCHEMA


def test_fde_selected_brier_not_minimum_penalized_fde():
    prediction = torch.zeros(1, 2, 60, 2)
    prediction[0, 1, :, 0] = .1
    target = torch.zeros(1, 60, 2)
    p = torch.tensor([[.01, .99]])
    result = trajectory_metrics(prediction, target, p)
    assert torch.allclose(result, torch.tensor([[0., 0., 0., .9801]]))


def test_joint_modes_cannot_be_selected_per_actor():
    prediction = torch.zeros(2, 2, 60, 2)
    prediction[0, 1, :, 0] = 10
    prediction[1, 0, :, 0] = 10
    target = torch.zeros(2, 60, 2); p = torch.full((2, 2), .5)
    marginal = trajectory_metrics(prediction, target, p)
    joint = trajectory_metrics(prediction, target, p, joint=True)
    assert marginal[:, 1].sum() == 0
    assert joint[0, 1] == 5
    assert joint[0, 2] == .5
    assert joint[0, 3] == 5.25


def test_submission_schema_shared_probability_and_world_order(tmp_path):
    scenes = [dict(scenario_id='s', track_ids=['00042', 'id-b'], probability=torch.tensor([.2, .8]),
                   trajectory=torch.arange(2 * 2 * 60 * 2).reshape(2, 2, 60, 2).double())]
    rows = prediction_rows(scenes)
    path = tmp_path / 'submission.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=SUBMISSION_SCHEMA), path)
    frame = pd.read_parquet(path)
    assert list(frame.columns) == ['scenario_id', 'track_id', 'probability', 'predicted_trajectory_x', 'predicted_trajectory_y']
    assert set(frame.track_id) == {'00042', 'id-b'}
    assert len(frame) == 4
    assert all(len(x) == 60 for x in frame.predicted_trajectory_x)
    assert all(np.isclose(g.probability.sum(), 1.) for _, g in frame.groupby('track_id'))
    assert frame.iloc[0].predicted_trajectory_x[0] == 120


def test_checker_rejects_missing_targets_and_accepts_complete_export(tmp_path):
    from validate_submission import validate
    import pytest
    root = tmp_path / 'raw'; scene = root / 'test' / 's'; scene.mkdir(parents=True)
    pd.DataFrame(dict(track_id=['00042', 'id-b'], focal_track_id=['00042', '00042'],
                      object_category=[3, 2], timestep=[49, 49])).to_parquet(scene / 'scenario_s.parquet')
    scenarios = [dict(scenario_id='s', track_ids=['00042'], probability=torch.tensor([.2, .8]),
                       trajectory=torch.zeros(1, 2, 60, 2))]
    file = tmp_path / 'submission.parquet'
    pq.write_table(pa.Table.from_pylist(prediction_rows(scenarios), schema=SUBMISSION_SCHEMA), file)
    assert validate(file, root, 'single-agent', modes=2)['valid']
    with pytest.raises(ValueError, match='Missing prediction targets'):
        validate(file, root, 'multi-world', modes=2)
