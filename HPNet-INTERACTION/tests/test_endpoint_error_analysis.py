import pytest
import torch

from analyze_endpoint_errors import compute_scene_metrics
from analyze_endpoint_errors import print_validation_metrics


def test_compute_scene_metrics_exposes_ade_fde_mode_disagreement():
    target = torch.zeros((2, 3, 2))
    prediction = torch.zeros((2, 2, 3, 2))
    prediction[0, 0, :, 0] = torch.tensor([0.5, 1.0, 1.0])
    prediction[1, 0, :, 0] = torch.tensor([0.5, 1.0, 3.0])
    prediction[:, 1, :, 0] = torch.tensor([1.0, 1.5, 1.5])

    result = compute_scene_metrics(prediction, target)

    assert result['joint_ade'] == pytest.approx(7.0 / 6.0)
    assert result['joint_fde'] == pytest.approx(1.5)
    assert result['joint_mr'] == 0.0
    assert result['ade_best_mode'] == 0
    assert result['fde_best_mode'] == 1
    assert result['best_mode_disagrees'] == 1
    assert result['oracle_de_t01'] == pytest.approx(0.5)
    assert result['oracle_de_t02'] == pytest.approx(1.0)
    assert result['oracle_de_t03'] == pytest.approx(1.5)
    assert result['worst_agent_fde_at_fde_mode'] == pytest.approx(1.5)


def test_compute_scene_metrics_rejects_empty_scored_scenario():
    with pytest.raises(ValueError, match='at least one agent'):
        compute_scene_metrics(
            torch.empty((0, 2, 3, 2)),
            torch.empty((0, 3, 2)),
        )


def test_print_validation_metrics_reports_headline_values(capsys):
    summary = {
        'checkpoints': {
            'candidate': {
                'scenario_count': 11794,
                'metrics': {
                    'joint_ade': {'mean': 0.16599081955298628},
                    'joint_fde': {'mean': 0.5359339260531181},
                    'joint_mr': {'mean': 0.11777174834661693},
                },
            },
        },
    }

    print_validation_metrics(summary)

    output = capsys.readouterr().out
    assert 'Validation complete. Headline metrics:' in output
    assert '[candidate] scenarios=11794' in output
    assert 'joint_ADE=0.16599082' in output
    assert 'joint_FDE=0.53593393' in output
    assert 'joint_MR=0.11777175' in output
