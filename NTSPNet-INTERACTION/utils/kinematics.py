from typing import MutableMapping, NamedTuple

import torch

from utils.process_data import wrap_angle


class HistoricalKinematics(NamedTuple):
    velocity: torch.Tensor
    speed: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor
    heading: torch.Tensor
    heading_change: torch.Tensor
    velocity_valid_mask: torch.Tensor
    acceleration_valid_mask: torch.Tensor
    jerk_valid_mask: torch.Tensor
    heading_valid_mask: torch.Tensor

    @property
    def acceleration_change(self) -> torch.Tensor:
        """Compatibility alias for historical processed data."""
        return self.jerk

    @property
    def acceleration_change_valid_mask(self) -> torch.Tensor:
        """Compatibility alias for historical processed data."""
        return self.jerk_valid_mask


def compute_historical_kinematics(position: torch.Tensor,
                                  visible_mask: torch.Tensor,
                                  step_time: float,
                                  speed_epsilon: float = 1e-4) -> HistoricalKinematics:
    """Compute causal motion features from a historical position sequence.

    Velocity, acceleration, and jerk use causal backward finite differences.
    Heading change is the wrapped frame-to-frame angular delta. Invalid
    derivatives are zeroed.
    """
    if position.ndim != 3 or position.size(-1) != 2:
        raise ValueError('position must have shape [num_agents, num_steps, 2]')
    if visible_mask.shape != position.shape[:2]:
        raise ValueError('visible_mask must have shape [num_agents, num_steps]')
    if step_time <= 0:
        raise ValueError('step_time must be positive')

    num_agents, num_steps = position.shape[:2]
    vector_shape = (num_agents, num_steps, 2)
    scalar_shape = (num_agents, num_steps)
    velocity = position.new_zeros(vector_shape)
    acceleration = position.new_zeros(vector_shape)
    jerk = position.new_zeros(vector_shape)
    heading = position.new_zeros(scalar_shape)
    heading_change = position.new_zeros(scalar_shape)
    velocity_valid_mask = torch.zeros(scalar_shape, dtype=torch.bool, device=position.device)
    acceleration_valid_mask = torch.zeros_like(velocity_valid_mask)
    jerk_valid_mask = torch.zeros_like(velocity_valid_mask)
    heading_valid_mask = torch.zeros_like(velocity_valid_mask)

    if num_steps <= 1:
        return HistoricalKinematics(
            velocity=velocity,
            speed=torch.linalg.vector_norm(velocity, dim=-1),
            acceleration=acceleration,
            jerk=jerk,
            heading=heading,
            heading_change=heading_change,
            velocity_valid_mask=velocity_valid_mask,
            acceleration_valid_mask=acceleration_valid_mask,
            jerk_valid_mask=jerk_valid_mask,
            heading_valid_mask=heading_valid_mask,
        )

    adjacent_visible = visible_mask[:, 1:] & visible_mask[:, :-1]
    velocity_valid_mask[:, 1:] = adjacent_visible
    velocity_delta = (position[:, 1:] - position[:, :-1]) / step_time
    velocity[:, 1:] = torch.where(
        adjacent_visible.unsqueeze(-1),
        velocity_delta,
        torch.zeros_like(velocity_delta),
    )
    speed = torch.linalg.vector_norm(velocity, dim=-1)

    motion_heading = torch.atan2(velocity[..., 1], velocity[..., 0])
    moving = velocity_valid_mask & (speed > speed_epsilon)
    for step in range(1, num_steps):
        continuous_track = visible_mask[:, step] & visible_mask[:, step - 1]
        carry_previous = continuous_track & heading_valid_mask[:, step - 1]
        heading[:, step] = torch.where(
            moving[:, step],
            motion_heading[:, step],
            torch.where(carry_previous, heading[:, step - 1], torch.zeros_like(heading[:, step])),
        )
        heading_valid_mask[:, step] = moving[:, step] | carry_previous

    valid_heading_change = (
        adjacent_visible
        & heading_valid_mask[:, 1:]
        & heading_valid_mask[:, :-1]
    )
    raw_heading_change = wrap_angle(heading[:, 1:] - heading[:, :-1])
    heading_change[:, 1:] = torch.where(
        valid_heading_change,
        raw_heading_change,
        torch.zeros_like(raw_heading_change),
    )

    if num_steps > 2:
        acceleration_valid_mask[:, 2:] = (
            velocity_valid_mask[:, 2:] & velocity_valid_mask[:, 1:-1]
        )
        acceleration_delta = (velocity[:, 2:] - velocity[:, 1:-1]) / step_time
        acceleration[:, 2:] = torch.where(
            acceleration_valid_mask[:, 2:].unsqueeze(-1),
            acceleration_delta,
            torch.zeros_like(acceleration_delta),
        )

    if num_steps > 3:
        jerk_valid_mask[:, 3:] = (
            acceleration_valid_mask[:, 3:] & acceleration_valid_mask[:, 2:-1]
        )
        jerk_delta = (acceleration[:, 3:] - acceleration[:, 2:-1]) / step_time
        jerk[:, 3:] = torch.where(
            jerk_valid_mask[:, 3:].unsqueeze(-1),
            jerk_delta,
            torch.zeros_like(jerk_delta),
        )

    return HistoricalKinematics(
        velocity=velocity,
        speed=speed,
        acceleration=acceleration,
        jerk=jerk,
        heading=heading,
        heading_change=heading_change,
        velocity_valid_mask=velocity_valid_mask,
        acceleration_valid_mask=acceleration_valid_mask,
        jerk_valid_mask=jerk_valid_mask,
        heading_valid_mask=heading_valid_mask,
    )


def update_agent_kinematics(agent_store: MutableMapping[str, torch.Tensor],
                            num_historical_steps: int,
                            step_time: float) -> HistoricalKinematics:
    """Compute and write motion fields while retaining legacy model aliases."""
    position = agent_store['position'][:, :num_historical_steps]
    visible_mask = agent_store['visible_mask'][:, :num_historical_steps]
    kinematics = compute_historical_kinematics(position, visible_mask, step_time)

    agent_store['velocity'] = kinematics.velocity
    agent_store['speed'] = kinematics.speed
    agent_store['acceleration'] = kinematics.acceleration
    agent_store['jerk'] = kinematics.jerk
    agent_store['acceleration_change'] = kinematics.jerk
    agent_store['heading_change'] = kinematics.heading_change
    agent_store['velocity_valid_mask'] = kinematics.velocity_valid_mask
    agent_store['acceleration_valid_mask'] = kinematics.acceleration_valid_mask
    agent_store['jerk_valid_mask'] = kinematics.jerk_valid_mask
    agent_store['acceleration_change_valid_mask'] = kinematics.jerk_valid_mask
    agent_store['heading_valid_mask'] = kinematics.heading_valid_mask

    if 'heading' in agent_store and agent_store['heading'].size(1) >= num_historical_steps:
        agent_store['heading'][:, :num_historical_steps] = kinematics.heading
    else:
        agent_store['heading'] = kinematics.heading

    velocity_angle = torch.atan2(kinematics.velocity[..., 1], kinematics.velocity[..., 0])
    velocity_theta = wrap_angle(velocity_angle - kinematics.heading)
    agent_store['velocity_length'] = kinematics.speed
    agent_store['velocity_theta'] = torch.where(
        kinematics.velocity_valid_mask,
        velocity_theta,
        torch.zeros_like(velocity_theta),
    )

    acceleration_length = torch.linalg.vector_norm(kinematics.acceleration, dim=-1)
    acceleration_angle = torch.atan2(
        kinematics.acceleration[..., 1],
        kinematics.acceleration[..., 0],
    )
    acceleration_theta = wrap_angle(acceleration_angle - kinematics.heading)
    agent_store['accel_length'] = acceleration_length
    agent_store['accel_theta'] = torch.where(
        kinematics.acceleration_valid_mask,
        acceleration_theta,
        torch.zeros_like(acceleration_theta),
    )
    return kinematics
