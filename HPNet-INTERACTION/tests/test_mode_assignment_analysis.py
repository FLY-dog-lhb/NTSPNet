import pytest
import torch

from analyze_mode_assignment import compute_assignment_metrics


def prediction_from_endpoint_errors(errors: torch.Tensor) -> torch.Tensor:
    prediction = torch.zeros((*errors.shape, 1, 2))
    prediction[:, :, 0, 0] = errors
    return prediction


def test_one_exception_recovers_single_agent_mode_disagreement():
    errors = torch.tensor([
        [0.0, 4.0, 5.0],
        [4.0, 0.0, 5.0],
        [4.0, 0.0, 5.0],
    ])
    target = torch.zeros((3, 1, 2))

    result = compute_assignment_metrics(
        prediction_from_endpoint_errors(errors), target,
    )

    assert result['joint_fde'] == pytest.approx(4.0 / 3.0)
    assert result['independent_fde'] == 0.0
    assert result['unique_agent_best_modes'] == 2
    assert result['agent_best_disagrees_scene_rate'] == pytest.approx(1.0 / 3.0)
    assert result['allow_1_exception_fde'] == 0.0
    assert result['best_2_mode_subset_fde'] == 0.0


def test_three_distinct_agent_modes_need_two_exceptions_or_three_modes():
    errors = torch.tensor([
        [0.0, 3.0, 3.0],
        [3.0, 0.0, 3.0],
        [3.0, 3.0, 0.0],
    ])
    target = torch.zeros((3, 1, 2))

    result = compute_assignment_metrics(
        prediction_from_endpoint_errors(errors), target,
    )

    assert result['joint_fde'] == 2.0
    assert result['allow_1_exception_fde'] == 1.0
    assert result['allow_2_exception_fde'] == 0.0
    assert result['best_2_mode_subset_fde'] == 1.0
    assert result['best_3_mode_subset_fde'] == 0.0
