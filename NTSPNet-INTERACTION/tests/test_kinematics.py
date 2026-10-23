import math

import torch

from utils.kinematics import compute_historical_kinematics
from utils.kinematics import update_agent_kinematics


def test_velocity_acceleration_and_jerk_use_backward_differences():
    position = torch.tensor([[
        [0.0, 0.0],
        [1.0, 0.0],
        [3.0, 0.0],
        [7.0, 0.0],
        [14.0, 0.0],
    ]])
    visible = torch.ones((1, 5), dtype=torch.bool)

    result = compute_historical_kinematics(position, visible, step_time=1.0)

    assert torch.allclose(result.velocity[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 4.0, 7.0]))
    assert torch.allclose(result.speed[0], torch.tensor([0.0, 1.0, 2.0, 4.0, 7.0]))
    assert torch.allclose(result.acceleration[0, :, 0], torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))
    assert torch.allclose(
        result.jerk[0, :, 0],
        torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]),
    )
    assert result.acceleration_change.data_ptr() == result.jerk.data_ptr()


def test_step_time_sets_physical_velocity_and_acceleration_units():
    position = torch.tensor([[[0.0, 0.0], [0.1, 0.0], [0.3, 0.0], [0.7, 0.0]]])
    visible = torch.ones((1, 4), dtype=torch.bool)

    result = compute_historical_kinematics(position, visible, step_time=0.1)

    assert torch.allclose(result.velocity[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 4.0]))
    assert torch.allclose(result.acceleration[0, :, 0], torch.tensor([0.0, 0.0, 10.0, 20.0]))
    assert torch.allclose(result.jerk[0, :, 0], torch.tensor([0.0, 0.0, 0.0, 100.0]))


def test_heading_and_wrapped_heading_change_follow_the_trajectory():
    position = torch.tensor([[
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]])
    visible = torch.ones((1, 4), dtype=torch.bool)

    result = compute_historical_kinematics(position, visible, step_time=1.0)

    assert torch.allclose(
        result.heading[0],
        torch.tensor([0.0, 0.0, math.pi / 2, math.pi]),
        atol=1e-6,
    )
    assert torch.allclose(
        result.heading_change[0],
        torch.tensor([0.0, 0.0, math.pi / 2, math.pi / 2]),
        atol=1e-6,
    )


def test_missing_observations_do_not_create_cross_gap_derivatives():
    position = torch.tensor([[
        [0.0, 0.0],
        [1.0, 0.0],
        [100.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
    ]])
    visible = torch.tensor([[True, True, False, True, True]])

    result = compute_historical_kinematics(position, visible, step_time=1.0)

    assert result.velocity_valid_mask.tolist() == [[False, True, False, False, True]]
    assert torch.allclose(result.velocity[0, :, 0], torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0]))
    assert not result.acceleration_valid_mask.any()
    assert not result.jerk_valid_mask.any()


def test_agent_store_exposes_canonical_jerk_and_compatibility_alias():
    agent_store = {
        'position': torch.tensor([[[0.0, 0.0], [0.1, 0.0], [0.3, 0.0], [0.7, 0.0]]]),
        'visible_mask': torch.ones((1, 4), dtype=torch.bool),
        'heading': torch.zeros((1, 4)),
    }

    result = update_agent_kinematics(agent_store, num_historical_steps=4, step_time=0.1)

    assert torch.allclose(agent_store['jerk'], result.jerk)
    assert agent_store['acceleration_change'].data_ptr() == agent_store['jerk'].data_ptr()
    assert agent_store['acceleration_change_valid_mask'].data_ptr() == agent_store['jerk_valid_mask'].data_ptr()
