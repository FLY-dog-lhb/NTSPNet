import pandas as pd
import pytest
import torch

from analyze_endpoint_case_trajectories import endpoint_geometry_metrics
from analyze_endpoint_case_trajectories import select_cases


def test_select_cases_keeps_worst_and_best_per_scenario():
    frame = pd.DataFrame({
        'sample_id': ['a1.pt', 'a2.pt', 'a3.pt', 'b1.pt', 'b2.pt'],
        'scenario_name_candidate': ['a', 'a', 'a', 'b', 'b'],
        'joint_fde_delta': [3.0, -2.0, 1.0, 4.0, -1.0],
        'has_conflict_candidate': [0, 0, 1, 0, 0],
        'scored_agent_count_candidate': [1, 2, 1, 3, 2],
    })
    result = select_cases(frame, ['a', 'b'], 1, 1, 3)
    assert result['sample_id'].tolist() == [
        'a1.pt', 'a2.pt', 'b1.pt', 'b2.pt',
    ]
    assert result['selection_role'].tolist() == [
        'worst', 'best', 'worst', 'best',
    ]


def test_endpoint_geometry_metrics_separates_longitudinal_and_lateral():
    target = torch.tensor([[
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
    ]])
    prediction = target[:, None].repeat(1, 2, 1, 1)
    prediction[:, 1, -1] += torch.tensor([2.0, -3.0])
    result = endpoint_geometry_metrics(
        prediction, target, best_mode=1,
        map_points=torch.tensor([[5.0, -3.0]]), late_start_step=1,
    )
    assert result['endpoint_longitudinal_mean'] == pytest.approx(2.0)
    assert result['endpoint_lateral_mean'] == pytest.approx(-3.0)
    assert result['endpoint_lateral_abs_mean'] == pytest.approx(3.0)
    assert result['joint_longitudinal_coverage'] == pytest.approx(0.0)
    assert result['joint_lateral_coverage'] == pytest.approx(0.0)
    assert result['endpoint_mode_spread_mean'] == pytest.approx(
        (13.0 ** 0.5),
    )
    assert result['endpoint_map_distance_mean'] == pytest.approx(0.0)
