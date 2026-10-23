import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse

from layers import GraphAttention
from layers import TwoLayerMLP
from utils import compute_angles_lengths_2D
from utils import drop_edge_between_samples
from utils import init_weights
from utils import transform_point_to_local_coordinate
from utils import transform_traj_to_global_coordinate
from utils import transform_traj_to_local_coordinate
from utils import wrap_angle


class OriginalBackbone(nn.Module):
    """Version-22 trajectory trunk with the original dense radius graphs."""

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_attn_layers: int,
                 num_modes: int,
                 num_heads: int,
                 dropout: float,
                 motion_source: str = 'csv') -> None:
        super().__init__()
        if motion_source != 'csv':
            raise ValueError(
                "OriginalBackbone requires motion_source='csv' to preserve "
                'the version-22 motion semantics'
            )
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.duration = duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_attn_layers = num_attn_layers
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.dropout = dropout
        self.motion_source = motion_source

        self.mode_tokens = nn.Embedding(num_modes, hidden_dim)
        self.a_emb_layer = TwoLayerMLP(7, hidden_dim, hidden_dim)
        self.l2m_emb_layer = TwoLayerMLP(3, hidden_dim, hidden_dim)
        self.t2m_emb_layer = TwoLayerMLP(4, hidden_dim, hidden_dim)
        self.m2m_h_emb_layer = TwoLayerMLP(4, hidden_dim, hidden_dim)
        self.m2m_a_emb_layer = TwoLayerMLP(5, hidden_dim, hidden_dim)
        self.m2m_s_emb_layer = TwoLayerMLP(3, hidden_dim, hidden_dim)

        self.l2m_attn_layer = GraphAttention(
            hidden_dim, num_heads, dropout, True, False,
        )
        self.t2m_attn_layer = GraphAttention(
            hidden_dim, num_heads, dropout, True, False,
        )
        self.m2m_h_attn_layers = nn.ModuleList([
            GraphAttention(hidden_dim, num_heads, dropout, True, True)
            for _ in range(num_attn_layers)
        ])
        self.m2m_a_attn_layers = nn.ModuleList([
            GraphAttention(hidden_dim, num_heads, dropout, True, True)
            for _ in range(num_attn_layers)
        ])
        self.m2m_s_attn_layers = nn.ModuleList([
            GraphAttention(hidden_dim, num_heads, dropout, True, True)
            for _ in range(num_attn_layers)
        ])
        self.mode_pair_emb = nn.Embedding(num_modes * num_modes, hidden_dim)

        self.traj_propose = TwoLayerMLP(
            hidden_dim, hidden_dim, num_future_steps * 2,
        )
        self.proposal_to_anchor = TwoLayerMLP(
            num_future_steps * 2, hidden_dim, hidden_dim,
        )
        self.l2n_emb_layer = TwoLayerMLP(3, hidden_dim, hidden_dim)
        self.t2n_emb_layer = TwoLayerMLP(4, hidden_dim, hidden_dim)
        self.l2n_attn_layer = GraphAttention(
            hidden_dim, num_heads, dropout, True, False,
        )
        self.t2n_attn_layer = GraphAttention(
            hidden_dim, num_heads, dropout, True, False,
        )
        self.n2n_h_attn_layers = nn.ModuleList([
            GraphAttention(hidden_dim, num_heads, dropout, True, True)
            for _ in range(num_attn_layers)
        ])
        self.n2n_a_attn_layers = nn.ModuleList([
            GraphAttention(hidden_dim, num_heads, dropout, True, True)
            for _ in range(num_attn_layers)
        ])
        self.n2n_s_attn_layers = nn.ModuleList([
            GraphAttention(hidden_dim, num_heads, dropout, True, True)
            for _ in range(num_attn_layers)
        ])
        self.traj_refine = TwoLayerMLP(
            hidden_dim, hidden_dim, num_future_steps * 2,
        )
        self.prob_decoder = TwoLayerMLP(hidden_dim, hidden_dim, 1)
        self.prob_norm = nn.Softmax(dim=-1)
        self.apply(init_weights)

    def _csv_motion(self, data: Batch):
        required = ('csv_velocity', 'csv_heading', 'csv_motion_valid_mask')
        missing = [name for name in required if name not in data['agent']]
        if missing:
            raise ValueError(
                'OriginalBackbone requires CSV motion sidecars; missing '
                f'{missing}'
            )
        steps = self.num_historical_steps
        visible = data['agent']['visible_mask']
        valid = data['agent']['csv_motion_valid_mask'] & visible
        velocity = data['agent']['csv_velocity'].to(
            dtype=data['agent']['position'].dtype,
        )
        heading = data['agent']['csv_heading'].to(
            dtype=data['agent']['position'].dtype,
        )
        velocity = torch.where(
            valid.unsqueeze(-1), velocity, torch.zeros_like(velocity),
        )
        heading = torch.where(valid, heading, torch.zeros_like(heading))
        speed, velocity_angle = compute_angles_lengths_2D(velocity)
        velocity_theta = wrap_angle(velocity_angle - heading)

        # Version 22 used raw adjacent-frame velocity differences, not m/s^2.
        acceleration = torch.zeros_like(velocity)
        acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        accel_length, accel_angle = compute_angles_lengths_2D(acceleration)
        accel_theta = wrap_angle(accel_angle - heading)
        return (
            speed[:, :steps],
            velocity_theta[:, :steps],
            accel_length[:, :steps],
            accel_theta[:, :steps],
            heading[:, :steps],
            heading,
        )

    def _temporal_edges(self,
                        source_position: torch.Tensor,
                        source_heading: torch.Tensor,
                        source_valid: torch.Tensor,
                        target_position: torch.Tensor,
                        target_heading: torch.Tensor,
                        target_valid: torch.Tensor,
                        refinement: bool = False):
        num_agents = source_valid.size(0)
        valid = source_valid.unsqueeze(2) & target_valid.reshape(
            num_agents, -1,
        ).unsqueeze(1)
        edge_index = dense_to_sparse(valid)[0]
        target_time = torch.div(
            edge_index[1], self.num_modes, rounding_mode='floor',
        )
        keep = target_time >= edge_index[0]
        keep &= target_time - edge_index[0] <= self.duration
        edge_index = edge_index[:, keep]
        vector = transform_point_to_local_coordinate(
            source_position[edge_index[0]],
            target_position[edge_index[1]],
            target_heading[edge_index[1]],
        )
        length, theta = compute_angles_lengths_2D(vector)
        heading = wrap_angle(
            source_heading[edge_index[0]] - target_heading[edge_index[1]],
        )
        interval = edge_index[0] - torch.div(
            edge_index[1], self.num_modes, rounding_mode='floor',
        )
        if refinement:
            interval = interval - self.num_future_steps // 2
        return edge_index, torch.stack([length, theta, heading, interval], -1)

    def _lane_edges(self,
                    data: Batch,
                    target_position: torch.Tensor,
                    target_heading: torch.Tensor,
                    target_batch: torch.Tensor,
                    target_valid: torch.Tensor):
        lane_position = data['lane']['position']
        lane_heading = data['lane']['heading']
        valid = data['lane']['visible_mask'].unsqueeze(1) & target_valid.unsqueeze(0)
        valid = drop_edge_between_samples(
            valid, batch=(data['lane']['batch'], target_batch),
        )
        edge_index = dense_to_sparse(valid)[0]
        distance = torch.norm(
            lane_position[edge_index[0]] - target_position[edge_index[1]],
            p=2, dim=-1,
        )
        edge_index = edge_index[:, distance < self.l2a_radius]
        vector = transform_point_to_local_coordinate(
            lane_position[edge_index[0]], target_position[edge_index[1]],
            target_heading[edge_index[1]],
        )
        length, theta = compute_angles_lengths_2D(vector)
        heading = wrap_angle(
            lane_heading[edge_index[0]] - target_heading[edge_index[1]],
        )
        return edge_index, torch.stack([length, theta, heading], -1)

    def _agent_edges(self,
                     position: torch.Tensor,
                     heading: torch.Tensor,
                     speed: torch.Tensor,
                     velocity_theta: torch.Tensor,
                     valid_by_group: torch.Tensor,
                     agent_batch: torch.Tensor):
        valid = valid_by_group.unsqueeze(2) & valid_by_group.unsqueeze(1)
        valid = drop_edge_between_samples(valid, agent_batch)
        edge_index = dense_to_sparse(valid)[0]
        edge_index = edge_index[:, edge_index[1] != edge_index[0]]
        distance = torch.norm(
            position[edge_index[1]] - position[edge_index[0]], p=2, dim=-1,
        )
        edge_index = edge_index[:, distance < self.a2a_radius]
        vector = transform_point_to_local_coordinate(
            position[edge_index[0]], position[edge_index[1]],
            heading[edge_index[1]],
        )
        length, theta = compute_angles_lengths_2D(vector)
        heading_diff = wrap_angle(
            heading[edge_index[0]] - heading[edge_index[1]],
        )
        speed_diff = speed[edge_index[0]] - speed[edge_index[1]]
        velocity_theta_diff = wrap_angle(
            velocity_theta[edge_index[0]] - velocity_theta[edge_index[1]],
        )
        attrs = torch.stack([
            length, theta, heading_diff, speed_diff, velocity_theta_diff,
        ], -1)
        return edge_index, attrs

    def _history_edges(self,
                       position: torch.Tensor,
                       heading: torch.Tensor,
                       valid: torch.Tensor):
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        edge_index = dense_to_sparse(pair_valid)[0]
        interval = edge_index[1] - edge_index[0]
        edge_index = edge_index[:, (interval > 0) & (interval <= self.duration)]
        vector = transform_point_to_local_coordinate(
            position[edge_index[0]], position[edge_index[1]],
            heading[edge_index[1]],
        )
        length, theta = compute_angles_lengths_2D(vector)
        heading_diff = wrap_angle(
            heading[edge_index[0]] - heading[edge_index[1]],
        )
        time_diff = edge_index[0] - edge_index[1]
        return edge_index, torch.stack([
            length, theta, heading_diff, time_diff,
        ], -1)

    @staticmethod
    def _apply_representation_residual(
            representation: torch.Tensor,
            residuals,
            stage: str) -> torch.Tensor:
        if residuals is None:
            return representation
        if stage not in residuals:
            raise ValueError(
                f'missing {stage!r} representation residual'
            )
        residual = residuals[stage]
        if residual.shape != representation.shape:
            raise ValueError(
                f'{stage} representation residual shape must be '
                f'{tuple(representation.shape)}, received '
                f'{tuple(residual.shape)}'
            )
        if not torch.isfinite(residual).all():
            raise ValueError(
                f'{stage} representation residual contains non-finite values'
            )
        return representation + residual

    def forward(self,
                data: Batch,
                l_embs: torch.Tensor,
                representation_residuals=None):
        h, k = self.num_historical_steps, self.num_modes
        (
            speed,
            velocity_theta,
            accel_length,
            accel_theta,
            agent_heading,
            temporal_source_heading,
        ) = self._csv_motion(data)
        length = data['agent']['length'].unsqueeze(-1).repeat_interleave(h, -1)
        width = data['agent']['width'].unsqueeze(-1).repeat_interleave(h, -1)
        agent_type = data['agent']['type'].to(speed.dtype).unsqueeze(-1).repeat_interleave(h, -1)
        agent_input = torch.stack([
            speed, velocity_theta, accel_length, accel_theta,
            length, width, agent_type,
        ], -1)
        agent_embs = self.a_emb_layer(input=agent_input)

        num_agents = length.size(0)
        mode_embs = self.mode_tokens.weight.unsqueeze(0).repeat_interleave(h, 0)
        mode_embs = mode_embs.unsqueeze(1).repeat_interleave(
            num_agents, 1,
        ).reshape(-1, self.hidden_dim)
        mode_batch = data['agent']['batch'].unsqueeze(1).repeat_interleave(k, 1)
        historical_position = data['agent']['position'][:, :h]
        historical_valid = data['agent']['visible_mask'][:, :h]
        mode_position = historical_position.unsqueeze(2).repeat_interleave(k, 2)
        mode_heading = agent_heading.unsqueeze(2).repeat_interleave(k, 2)
        mode_valid = historical_valid.unsqueeze(2).repeat_interleave(k, 2)

        t2m_edge, t2m_attr = self._temporal_edges(
            historical_position.reshape(-1, 2),
            temporal_source_heading.reshape(-1),
            historical_valid, mode_position.reshape(-1, 2),
            mode_heading.reshape(-1), mode_valid,
        )
        l2m_edge, l2m_attr = self._lane_edges(
            data, mode_position.reshape(-1, 2), mode_heading.reshape(-1),
            mode_batch.unsqueeze(1).repeat_interleave(h, 1).reshape(-1),
            mode_valid.reshape(-1),
        )

        agent_position_grouped = mode_position.permute(1, 2, 0, 3).reshape(-1, 2)
        agent_heading_grouped = mode_heading.permute(1, 2, 0).reshape(-1)
        speed_grouped = speed.transpose(0, 1).unsqueeze(1).repeat_interleave(k, 1).reshape(-1)
        theta_grouped = velocity_theta.transpose(0, 1).unsqueeze(1).repeat_interleave(k, 1).reshape(-1)
        valid_grouped = mode_valid.permute(1, 2, 0).reshape(h * k, -1)
        m2m_a_edge, m2m_a_attr = self._agent_edges(
            agent_position_grouped, agent_heading_grouped, speed_grouped,
            theta_grouped, valid_grouped, data['agent']['batch'],
        )
        history_position = mode_position.permute(2, 0, 1, 3).reshape(-1, 2)
        history_heading = mode_heading.permute(2, 0, 1).reshape(-1)
        history_valid = mode_valid.permute(2, 0, 1).reshape(-1, h)
        m2m_h_edge, m2m_h_attr = self._history_edges(
            history_position, history_heading, history_valid,
        )
        mode_valid_grouped = mode_valid.transpose(0, 1).reshape(-1, k)
        m2m_s_edge = dense_to_sparse(
            mode_valid_grouped.unsqueeze(2) & mode_valid_grouped.unsqueeze(1),
        )[0]
        m2m_s_edge = m2m_s_edge[:, m2m_s_edge[0] != m2m_s_edge[1]]
        mode_pair = (
            (m2m_s_edge[0] % k) * k + (m2m_s_edge[1] % k)
        )

        temporal_embs = self.t2m_attn_layer(
            x=[agent_embs.reshape(-1, self.hidden_dim), mode_embs],
            edge_index=t2m_edge,
            edge_attr=self.t2m_emb_layer(input=t2m_attr),
        )
        lane_embs = self.l2m_attn_layer(
            x=[l_embs, mode_embs], edge_index=l2m_edge,
            edge_attr=self.l2m_emb_layer(input=l2m_attr),
        )
        mode_embs = (temporal_embs + lane_embs).reshape(
            num_agents, h, k, self.hidden_dim,
        ).transpose(0, 1).reshape(-1, self.hidden_dim)
        m2m_a_emb = self.m2m_a_emb_layer(input=m2m_a_attr)
        m2m_h_emb = self.m2m_h_emb_layer(input=m2m_h_attr)
        m2m_s_emb = self.mode_pair_emb(mode_pair)
        for layer in range(self.num_attn_layers):
            mode_embs = mode_embs.reshape(h, num_agents, k, self.hidden_dim).transpose(1, 2).reshape(-1, self.hidden_dim)
            mode_embs = self.m2m_a_attn_layers[layer](mode_embs, m2m_a_edge, m2m_a_emb)
            mode_embs = mode_embs.reshape(h, k, num_agents, self.hidden_dim).permute(1, 2, 0, 3).reshape(-1, self.hidden_dim)
            mode_embs = self.m2m_h_attn_layers[layer](mode_embs, m2m_h_edge, m2m_h_emb)
            mode_embs = mode_embs.reshape(k, num_agents, h, self.hidden_dim).transpose(0, 2).reshape(-1, self.hidden_dim)
            mode_embs = self.m2m_s_attn_layers[layer](mode_embs, m2m_s_edge, m2m_s_emb)
        mode_embs = mode_embs.reshape(h, num_agents, k, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        mode_embs = self._apply_representation_residual(
            mode_embs,
            representation_residuals,
            stage='proposal',
        )

        traj_propose = self.traj_propose(mode_embs).reshape(
            num_agents, h, k, self.num_future_steps, 2,
        )
        traj_propose = transform_traj_to_global_coordinate(
            traj_propose, mode_position, mode_heading,
        )
        proposal = traj_propose.detach()
        anchor_position = proposal[:, :, :, self.num_future_steps // 2]
        _, anchor_heading = compute_angles_lengths_2D(
            proposal[:, :, :, self.num_future_steps // 2]
            - proposal[:, :, :, self.num_future_steps // 2 - 1],
        )
        proposal_local = transform_traj_to_local_coordinate(
            proposal, anchor_position, anchor_heading,
        )
        refine_embs = self.proposal_to_anchor(
            proposal_local.reshape(-1, self.num_future_steps * 2),
        )

        t2n_edge, t2n_attr = self._temporal_edges(
            historical_position.reshape(-1, 2),
            temporal_source_heading.reshape(-1),
            historical_valid, anchor_position.reshape(-1, 2),
            anchor_heading.reshape(-1), mode_valid, refinement=True,
        )
        l2n_edge, l2n_attr = self._lane_edges(
            data, anchor_position.reshape(-1, 2), anchor_heading.reshape(-1),
            mode_batch.unsqueeze(1).repeat_interleave(h, 1).reshape(-1),
            mode_valid.reshape(-1),
        )
        refine_agent_position = anchor_position.permute(1, 2, 0, 3).reshape(-1, 2)
        refine_agent_heading = anchor_heading.permute(1, 2, 0).reshape(-1)
        n2n_a_edge, n2n_a_attr = self._agent_edges(
            refine_agent_position, refine_agent_heading, speed_grouped,
            theta_grouped, valid_grouped, data['agent']['batch'],
        )
        refine_history_position = anchor_position.permute(2, 0, 1, 3).reshape(-1, 2)
        refine_history_heading = anchor_heading.permute(2, 0, 1).reshape(-1)
        n2n_h_edge, n2n_h_attr = self._history_edges(
            refine_history_position, refine_history_heading, history_valid,
        )
        refine_mode_position = anchor_position.transpose(0, 1).reshape(-1, 2)
        refine_mode_heading = anchor_heading.transpose(0, 1).reshape(-1)
        n2n_s_edge = dense_to_sparse(
            mode_valid_grouped.unsqueeze(2) & mode_valid_grouped.unsqueeze(1),
        )[0]
        n2n_s_edge = n2n_s_edge[:, n2n_s_edge[0] != n2n_s_edge[1]]
        refine_mode_vector = transform_point_to_local_coordinate(
            refine_mode_position[n2n_s_edge[0]],
            refine_mode_position[n2n_s_edge[1]],
            refine_mode_heading[n2n_s_edge[1]],
        )
        mode_length, mode_theta = compute_angles_lengths_2D(refine_mode_vector)
        mode_heading_diff = wrap_angle(
            refine_mode_heading[n2n_s_edge[0]]
            - refine_mode_heading[n2n_s_edge[1]],
        )
        n2n_s_attr = torch.stack([
            mode_length, mode_theta, mode_heading_diff,
        ], -1)

        temporal_refine = self.t2n_attn_layer(
            x=[agent_embs.reshape(-1, self.hidden_dim), refine_embs],
            edge_index=t2n_edge,
            edge_attr=self.t2n_emb_layer(input=t2n_attr),
        )
        lane_refine = self.l2n_attn_layer(
            x=[l_embs, refine_embs], edge_index=l2n_edge,
            edge_attr=self.l2n_emb_layer(input=l2n_attr),
        )
        refine_embs = (temporal_refine + lane_refine).reshape(
            num_agents, h, k, self.hidden_dim,
        ).transpose(0, 1).reshape(-1, self.hidden_dim)
        n2n_a_emb = self.m2m_a_emb_layer(input=n2n_a_attr)
        n2n_h_emb = self.m2m_h_emb_layer(input=n2n_h_attr)
        n2n_s_emb = self.m2m_s_emb_layer(input=n2n_s_attr)
        for layer in range(self.num_attn_layers):
            refine_embs = refine_embs.reshape(h, num_agents, k, self.hidden_dim).transpose(1, 2).reshape(-1, self.hidden_dim)
            refine_embs = self.n2n_a_attn_layers[layer](refine_embs, n2n_a_edge, n2n_a_emb)
            refine_embs = refine_embs.reshape(h, k, num_agents, self.hidden_dim).permute(1, 2, 0, 3).reshape(-1, self.hidden_dim)
            refine_embs = self.n2n_h_attn_layers[layer](refine_embs, n2n_h_edge, n2n_h_emb)
            refine_embs = refine_embs.reshape(k, num_agents, h, self.hidden_dim).transpose(0, 2).reshape(-1, self.hidden_dim)
            refine_embs = self.n2n_s_attn_layers[layer](refine_embs, n2n_s_edge, n2n_s_emb)
        refine_embs = refine_embs.reshape(h, num_agents, k, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        refine_embs = self._apply_representation_residual(
            refine_embs,
            representation_residuals,
            stage='refinement',
        )

        traj_refine = self.traj_refine(refine_embs).reshape(
            num_agents, h, k, self.num_future_steps, 2,
        )
        traj_output = transform_traj_to_global_coordinate(
            proposal_local + traj_refine, anchor_position, anchor_heading,
        )
        probability = self.prob_decoder(refine_embs.detach()).reshape(
            num_agents, h, k,
        )
        return traj_propose, traj_output, self.prob_norm(probability)
