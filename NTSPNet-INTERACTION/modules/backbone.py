import heapq
from itertools import permutations
from pathlib import Path
from typing import Mapping
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse

from layers import GraphAttention
from layers import TwoLayerMLP
from modules.lane_reachability import LaneReachabilityBuilder
from modules.lane_reachability import load_lane_association_from_data
from modules.priority import PriorityModule
from modules.vehicle_interaction_graph import VehicleInteractionGraphBuilder
from utils import compute_angles_lengths_2D
from utils import compute_historical_kinematics
from utils import init_weights
from utils import wrap_angle
from utils import transform_point_to_local_coordinate
from utils import transform_point_to_global_coordinate
from utils import transform_traj_to_global_coordinate
from utils import transform_traj_to_local_coordinate

class Backbone(nn.Module):

    STYLE_FEATURE_NAMES = (
        'speed',
        'acceleration_magnitude',
        'jerk_magnitude',
        'heading',
        'heading_change',
        'degree_centrality',
        'closeness_centrality',
        'degree_centrality_change',
        'closeness_centrality_change',
        'THW',
        'TTC',
        'lane_lateral_distance',
        'same_lane_arrival_time_difference',
    )
    STYLE_SCORE_INDICATOR_NAMES = (
        'speed_p90',
        'acceleration_magnitude_p90',
        'jerk_magnitude_mean',
        'heading_change_intensity',
        'lateral_change_intensity',
        'ttc_risk_mean',
    )
    STYLE_CLASS_NAMES = ('aggressive', 'normal', 'conservative')

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
                 interaction_distance_margin: float = 5.0,
                 observation_step_time: float = 0.1,
                 lane_arrival_time_threshold: float = 1.0,
                 ttc_speed_epsilon: float = 1e-3,
                 interaction_ttc_threshold: float = 3.0,
                 interaction_collision_distance: float = 3.0,
                 conflict_lane_radius: float = 10.0,
                 traffic_base_radius: float = 10.0,
                 traffic_speed_scale: float = 1.0,
                 traffic_backward_attention_ratio: float = 0.5,
                 traffic_max_attention_nodes: int = 5,
                 traffic_current_radius: float = 8.0,
                 traffic_lane_lateral_radius: float = 4.0,
                 traffic_topology_base_distance: float = 5.0,
                 traffic_lane_change_lateral_radius: float = 1.0,
                 traffic_lane_change_lateral_speed: float = 0.3,
                 style_thw_max: float = 5.0,
                 style_ttc_max: float = 10.0,
                 style_arrival_time_fill: float = 3.0,
                 style_transformer_layers: int = 2,
                 style_gap_reset_steps: int = 3,
                 style_pool_min_length: int = 3,
                 style_min_visible_steps: int = 5,
                 style_min_last_segment_steps: int = 5,
                 style_statistics_epsilon: float = 1e-3,
                 style_statistics_path: str = None,
                 motion_source: str = 'causal_difference',
                 preserve_base_edges: bool = False,
                 reachability_residual_scale: float = 0.0,
                 style_film_mode: str = 'legacy',
                 style_film_bound: float = 0.1,
                 conflict_update_scope: str = 'legacy',
                 priority_injection_mode: str = 'legacy',
                 proposal_conflict_mode_scale: float = 0.0,
                 proposal_conflict_mode_adapter_scale: float = 0.0,
                 proposal_conflict_residual_gate_scale: float = 0.0,
                 scene_mode_coordination_scale: float = 0.0,
                 joint_mode_composition_scale: float = 0.0,
                 joint_mode_assignment_mode: str = 'independent',
                 joint_mode_max_exception_agents: int = 2) -> None:
        super(Backbone, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.duration = duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.interaction_distance_margin = interaction_distance_margin
        self.observation_step_time = observation_step_time
        self.observation_window = num_historical_steps * observation_step_time
        self.prediction_window = num_future_steps * observation_step_time
        self.lane_arrival_time_threshold = lane_arrival_time_threshold
        self.ttc_speed_epsilon = ttc_speed_epsilon
        self.interaction_ttc_threshold = interaction_ttc_threshold
        self.interaction_collision_distance = interaction_collision_distance
        self.conflict_lane_radius = conflict_lane_radius
        self.traffic_base_radius = traffic_base_radius
        self.traffic_speed_scale = traffic_speed_scale
        self.traffic_backward_attention_ratio = traffic_backward_attention_ratio
        self.traffic_max_attention_nodes = traffic_max_attention_nodes
        self.traffic_current_radius = traffic_current_radius
        self.traffic_lane_lateral_radius = traffic_lane_lateral_radius
        self.traffic_topology_base_distance = traffic_topology_base_distance
        self.traffic_lane_change_lateral_radius = traffic_lane_change_lateral_radius
        self.traffic_lane_change_lateral_speed = traffic_lane_change_lateral_speed
        self.style_thw_max = style_thw_max
        self.style_ttc_max = style_ttc_max
        self.style_arrival_time_fill = style_arrival_time_fill
        self.style_transformer_layers = style_transformer_layers
        self.style_gap_reset_steps = style_gap_reset_steps
        self.style_pool_min_length = style_pool_min_length
        self.style_min_visible_steps = style_min_visible_steps
        self.style_min_last_segment_steps = style_min_last_segment_steps
        self.style_statistics_epsilon = style_statistics_epsilon
        self.style_statistics_path = style_statistics_path
        self.motion_source = motion_source
        self.preserve_base_edges = preserve_base_edges
        self.reachability_residual_scale = reachability_residual_scale
        self.style_film_mode = style_film_mode
        self.style_film_bound = style_film_bound
        self.conflict_update_scope = conflict_update_scope
        self.priority_injection_mode = priority_injection_mode
        self.proposal_conflict_mode_scale = proposal_conflict_mode_scale
        self.proposal_conflict_mode_adapter_scale = (
            proposal_conflict_mode_adapter_scale
        )
        self.proposal_conflict_residual_gate_scale = (
            proposal_conflict_residual_gate_scale
        )
        self.scene_mode_coordination_scale = scene_mode_coordination_scale
        self.joint_mode_composition_scale = joint_mode_composition_scale
        self.joint_mode_assignment_mode = joint_mode_assignment_mode
        self.joint_mode_max_exception_agents = (
            joint_mode_max_exception_agents
        )
        if motion_source not in ('causal_difference', 'csv'):
            raise ValueError(
                "motion_source must be 'causal_difference' or 'csv'"
            )
        if reachability_residual_scale < 0:
            raise ValueError('reachability_residual_scale must be non-negative')
        if style_film_mode not in ('legacy', 'bounded_identity'):
            raise ValueError(
                "style_film_mode must be 'legacy' or 'bounded_identity'"
            )
        if style_film_bound <= 0:
            raise ValueError('style_film_bound must be positive')
        if conflict_update_scope not in ('legacy', 'participants_only'):
            raise ValueError(
                "conflict_update_scope must be 'legacy' or "
                "'participants_only'"
            )
        if priority_injection_mode not in ('legacy', 'auxiliary_only'):
            raise ValueError(
                "priority_injection_mode must be 'legacy' or "
                "'auxiliary_only'"
            )
        if not 0.0 <= proposal_conflict_mode_scale <= 1.0:
            raise ValueError(
                'proposal_conflict_mode_scale must be between 0 and 1'
            )
        if not 0.0 <= proposal_conflict_mode_adapter_scale <= 1.0:
            raise ValueError(
                'proposal_conflict_mode_adapter_scale must be between 0 and 1'
            )
        if not 0.0 <= proposal_conflict_residual_gate_scale <= 1.0:
            raise ValueError(
                'proposal_conflict_residual_gate_scale must be between 0 and 1'
            )
        if not 0.0 <= scene_mode_coordination_scale <= 1.0:
            raise ValueError(
                'scene_mode_coordination_scale must be between 0 and 1'
            )
        if not 0.0 <= joint_mode_composition_scale <= 1.0:
            raise ValueError(
                'joint_mode_composition_scale must be between 0 and 1'
            )
        if joint_mode_assignment_mode not in (
                'independent', 'permutation', 'limited_permutation'):
            raise ValueError(
                "joint_mode_assignment_mode must be 'independent' or "
                "'permutation' or 'limited_permutation'"
            )
        if joint_mode_assignment_mode in (
                'permutation', 'limited_permutation') and num_modes > 8:
            raise ValueError(
                'permutation assignment supports at most 8 modes'
            )
        if joint_mode_max_exception_agents <= 0:
            raise ValueError(
                'joint_mode_max_exception_agents must be positive'
            )
        self.return_style_attention_weights = False
        if style_arrival_time_fill < 0:
            raise ValueError('style_arrival_time_fill must be non-negative')
        if style_ttc_max <= 0:
            raise ValueError('style_ttc_max must be positive')
        if style_transformer_layers <= 0:
            raise ValueError('style_transformer_layers must be positive')
        if style_gap_reset_steps <= 0:
            raise ValueError('style_gap_reset_steps must be positive')
        if style_pool_min_length <= 0:
            raise ValueError('style_pool_min_length must be positive')
        if style_min_visible_steps <= 0:
            raise ValueError('style_min_visible_steps must be positive')
        if style_min_last_segment_steps <= 0:
            raise ValueError('style_min_last_segment_steps must be positive')
        if style_statistics_epsilon <= 0:
            raise ValueError('style_statistics_epsilon must be positive')
        self.num_attn_layers = num_attn_layers
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.dropout = dropout

        self.mode_tokens = nn.Embedding(num_modes, hidden_dim)     #[K,D]
        self.proposal_conflict_mode_tokens = nn.Embedding(
            num_modes,
            hidden_dim,
        )
        self.proposal_conflict_residual_gate_tokens = nn.Embedding(
            num_modes,
            hidden_dim,
        )
        self.scene_mode_coordination = nn.Linear(
            hidden_dim,
            hidden_dim,
        )
        self.joint_mode_composer = nn.Linear(
            2 * hidden_dim,
            num_modes,
        )
        if joint_mode_assignment_mode in (
                'permutation', 'limited_permutation'):
            joint_mode_permutations = torch.tensor(
                list(permutations(range(num_modes))),
                dtype=torch.long,
            )
        else:
            joint_mode_permutations = torch.empty(
                (0, num_modes), dtype=torch.long,
            )
        self.register_buffer(
            'joint_mode_permutations',
            joint_mode_permutations,
            persistent=False,
        )

        self.a_emb_layer = TwoLayerMLP(input_dim=13, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.style_input_projection = nn.Linear(len(self.STYLE_FEATURE_NAMES), hidden_dim)
        self.style_time_embedding = nn.Embedding(num_historical_steps, hidden_dim)
        style_transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.style_temporal_encoder = nn.TransformerEncoder(
            encoder_layer=style_transformer_layer,
            num_layers=style_transformer_layers,
        )
        self.style_pool_query = nn.Embedding(1, hidden_dim)
        self.style_attn_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.style_embedding_mlp = TwoLayerMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.style_classifier = TwoLayerMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=len(self.STYLE_CLASS_NAMES),
        )
        self.style_probability_norm = nn.Softmax(dim=-1)
        self.register_buffer(
            'style_score_weights',
            torch.tensor([0.25, 0.20, 0.15, 0.15, 0.10, 0.15]),
        )
        num_style_indicators = len(self.STYLE_SCORE_INDICATOR_NAMES)
        self.register_buffer('style_indicator_median', torch.zeros(num_style_indicators))
        self.register_buffer('style_indicator_iqr', torch.ones(num_style_indicators))
        self.register_buffer('style_score_q33', torch.tensor(0.0))
        self.register_buffer('style_score_q66', torch.tensor(0.0))
        self.register_buffer('style_statistics_ready', torch.tensor(False))

        self.a2l_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l2l_traffic_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        self.l2m_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.t2m_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.m2m_h_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_a_emb_layer = TwoLayerMLP(input_dim=10, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_s_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.t2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.a2l_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.l2l_traffic_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True)

        self.m2m_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.mode_pair_emb = nn.Embedding(num_modes * num_modes, hidden_dim)

        self.traj_propose = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps*2)

        self.proposal_to_anchor = TwoLayerMLP(input_dim=self.num_future_steps*2, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2n_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.t2n_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.t2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.n2n_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_refine = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps*2)

        self.prob_decoder = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.prob_norm = nn.Softmax(dim=-1)

        self.priority_module = PriorityModule(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            softmin_temperature=0.2,
            simultaneous_threshold=0.5,
            order_scale=2.0,
        )
        self.conflict_edge_emb_layer = TwoLayerMLP(
            input_dim=10,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.vehicle_priority_attn_layer = GraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            has_edge_attr=True,
            if_self_attention=True,
        )
        self.vehicle_to_lane_source_mlp = TwoLayerMLP(
            input_dim=3 * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.proposal_conflict_attn_layer = GraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            has_edge_attr=True,
            if_self_attention=True,
        )
        self.refinement_conflict_attn_layer = GraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            has_edge_attr=True,
            if_self_attention=True,
        )
        self.proposal_style_film = TwoLayerMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=2 * hidden_dim,
        )
        self.refinement_style_film = TwoLayerMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=2 * hidden_dim,
        )

        self.vehicle_interaction_graph = VehicleInteractionGraphBuilder(
            distance_margin=interaction_distance_margin,
            observation_window=self.observation_window,
            ttc_threshold=interaction_ttc_threshold,
            collision_distance=interaction_collision_distance,
            max_distance=a2a_radius,
            lane_arrival_time_threshold=lane_arrival_time_threshold,
            thw_max=style_thw_max,
            safety_ttc_max=style_ttc_max,
            eps=ttc_speed_epsilon,
        )
        self.lane_reachability_builder = LaneReachabilityBuilder(
            centerline_side_index=1,
            default_lateral_distance=2.5,
            centerline_spacing=2.0,
            lane_change_distance=12.0,
            max_lane_changes=1,
            tau_max=5.0,
            conflict_arrival_threshold=2.0,
            softmin_temperature=0.2,
        )

        self.apply(init_weights)
        nn.init.zeros_(self.proposal_conflict_mode_tokens.weight)
        nn.init.zeros_(self.proposal_conflict_residual_gate_tokens.weight)
        nn.init.zeros_(self.scene_mode_coordination.weight)
        nn.init.zeros_(self.scene_mode_coordination.bias)
        nn.init.zeros_(self.joint_mode_composer.weight)
        nn.init.zeros_(self.joint_mode_composer.bias)
        if self.style_film_mode == 'bounded_identity':
            self._initialize_style_film_as_identity()
        if style_statistics_path is not None and Path(style_statistics_path).is_file():
            self.load_style_statistics(style_statistics_path)

    def _initialize_style_film_as_identity(self) -> None:
        """Zero only FiLM output layers after global parameter initialization."""
        for film_generator in (
            self.proposal_style_film,
            self.refinement_style_film,
        ):
            output_layer = film_generator.mlp[-1]
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def load_style_statistics(self, statistics_path: str) -> None:
        """Load train-only robust-normalization and score quantile statistics."""
        statistics = torch.load(
            statistics_path,
            map_location='cpu',
            weights_only=True,
        )
        if not isinstance(statistics, Mapping):
            raise ValueError('style statistics file must contain a mapping')
        expected_names = list(self.STYLE_SCORE_INDICATOR_NAMES)
        if statistics.get('indicator_names') != expected_names:
            raise ValueError(
                'style statistics indicator order does not match the current model: '
                f'expected {expected_names}'
            )
        median = torch.as_tensor(statistics.get('median'), dtype=torch.float)
        iqr = torch.as_tensor(statistics.get('iqr'), dtype=torch.float)
        if median.shape != self.style_indicator_median.shape:
            raise ValueError('style statistics median has an invalid shape')
        if iqr.shape != self.style_indicator_iqr.shape:
            raise ValueError('style statistics IQR has an invalid shape')
        if not torch.isfinite(median).all() or not torch.isfinite(iqr).all():
            raise ValueError('style statistics median and IQR must be finite')
        if (iqr < 0).any():
            raise ValueError('style statistics IQR must be non-negative')
        q33 = torch.as_tensor(statistics.get('q33'), dtype=torch.float).reshape(())
        q66 = torch.as_tensor(statistics.get('q66'), dtype=torch.float).reshape(())
        if not torch.isfinite(q33) or not torch.isfinite(q66) or q33 > q66:
            raise ValueError('style score quantiles must be finite and satisfy Q33 <= Q66')
        file_epsilon = float(statistics.get('epsilon', self.style_statistics_epsilon))
        if abs(file_epsilon - self.style_statistics_epsilon) > 1e-12:
            raise ValueError(
                'style statistics epsilon does not match style_statistics_epsilon: '
                f'{file_epsilon} != {self.style_statistics_epsilon}'
            )
        file_weights = torch.as_tensor(
            statistics.get('score_weights', self.style_score_weights),
            dtype=torch.float,
        )
        if not torch.allclose(file_weights, self.style_score_weights.cpu()):
            raise ValueError('style statistics score weights do not match the current model')
        configuration = statistics.get('configuration')
        expected_configuration = {
            'num_historical_steps': self.num_historical_steps,
            'num_future_steps': self.num_future_steps,
            'observation_step_time': self.observation_step_time,
            'a2a_radius': self.a2a_radius,
            'interaction_distance_margin': self.interaction_distance_margin,
            'lane_arrival_time_threshold': self.lane_arrival_time_threshold,
            'ttc_speed_epsilon': self.ttc_speed_epsilon,
            'interaction_ttc_threshold': self.interaction_ttc_threshold,
            'interaction_collision_distance': self.interaction_collision_distance,
            'traffic_current_radius': self.traffic_current_radius,
            'traffic_lane_lateral_radius': self.traffic_lane_lateral_radius,
            'traffic_topology_base_distance': self.traffic_topology_base_distance,
            'style_ttc_max': self.style_ttc_max,
            'style_min_visible_steps': self.style_min_visible_steps,
        }
        if not isinstance(configuration, Mapping):
            raise ValueError('style statistics file is missing its configuration')
        for key, expected_value in expected_configuration.items():
            file_value = configuration.get(key)
            if file_value is None or abs(float(file_value) - float(expected_value)) > 1e-12:
                raise ValueError(
                    f'style statistics configuration mismatch for {key}: '
                    f'{file_value} != {expected_value}'
                )

        self.style_indicator_median.copy_(median)
        self.style_indicator_iqr.copy_(iqr)
        self.style_score_q33.copy_(q33)
        self.style_score_q66.copy_(q66)
        self.style_statistics_ready.fill_(True)

    def require_style_statistics(self) -> None:
        if bool(self.style_statistics_ready.item()):
            return
        path_hint = self.style_statistics_path or '<root>/train_style_statistics.pt'
        raise RuntimeError(
            'driving-style pseudo-label statistics are unavailable; run '
            '`python compute_style_statistics.py --root ...` first or resume from a '
            f'checkpoint containing the statistics (expected path: {path_hint})'
        )

    def _embed_vehicle_interaction_edges(self,
                                         edge_index: torch.Tensor,
                                         graph_edge_attr: torch.Tensor,
                                         position: torch.Tensor,
                                         heading: torch.Tensor,
                                         velocity_length: torch.Tensor,
                                         velocity_theta: torch.Tensor) -> torch.Tensor:
        edge_vector = transform_point_to_local_coordinate(
            position[edge_index[0]],
            position[edge_index[1]],
            heading[edge_index[1]],
        )
        edge_length, edge_theta = compute_angles_lengths_2D(edge_vector)
        edge_heading = wrap_angle(heading[edge_index[0]] - heading[edge_index[1]])
        edge_speed_diff = velocity_length[edge_index[0]] - velocity_length[edge_index[1]]
        edge_velocity_theta_diff = wrap_angle(
            velocity_theta[edge_index[0]] - velocity_theta[edge_index[1]]
        )
        graph_cues = graph_edge_attr[:, 1:].clone()
        graph_cues[:, 0] = graph_cues[:, 0] / self.interaction_ttc_threshold
        edge_attr_input = torch.cat([
            torch.stack([
                edge_length,
                edge_theta,
                edge_heading,
                edge_speed_diff,
                edge_velocity_theta_diff,
            ], dim=-1),
            graph_cues,
        ], dim=-1)
        return self.m2m_a_emb_layer(input=edge_attr_input)

    def _select_main_motion(self,
                            data: Batch,
                            causal_kinematics) -> tuple:
        """Select trunk motion while keeping causal features for new modules."""
        if self.motion_source == 'causal_difference':
            velocity = causal_kinematics.velocity
            speed = causal_kinematics.speed
            acceleration = causal_kinematics.acceleration
            jerk = causal_kinematics.jerk
            heading = causal_kinematics.heading
            heading_change = causal_kinematics.heading_change
            velocity_angle = torch.atan2(velocity[..., 1], velocity[..., 0])
            velocity_theta = torch.where(
                causal_kinematics.velocity_valid_mask,
                wrap_angle(velocity_angle - heading),
                torch.zeros_like(velocity_angle),
            )
            return (
                velocity, speed, acceleration, jerk,
                heading, heading_change, velocity_theta,
            )

        required_fields = (
            'csv_velocity', 'csv_heading', 'csv_motion_valid_mask',
        )
        missing_fields = [
            field_name for field_name in required_fields
            if field_name not in data['agent']
        ]
        if missing_fields:
            raise ValueError(
                'motion_source=csv requires CSV motion fields '
                f'{missing_fields}; generate and load csv_motion_v1 sidecars'
            )
        visible_mask = data['agent']['visible_mask'][
            :, :self.num_historical_steps
        ]
        csv_valid_mask = data['agent']['csv_motion_valid_mask'][
            :, :self.num_historical_steps
        ] & visible_mask
        velocity = data['agent']['csv_velocity'][
            :, :self.num_historical_steps
        ].to(dtype=data['agent']['position'].dtype)
        heading = data['agent']['csv_heading'][
            :, :self.num_historical_steps
        ].to(dtype=data['agent']['position'].dtype)
        velocity = torch.where(
            csv_valid_mask.unsqueeze(-1), velocity, torch.zeros_like(velocity),
        )
        heading = torch.where(
            csv_valid_mask, heading, torch.zeros_like(heading),
        )
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        velocity_angle = torch.atan2(velocity[..., 1], velocity[..., 0])
        velocity_theta = torch.where(
            csv_valid_mask,
            wrap_angle(velocity_angle - heading),
            torch.zeros_like(velocity_angle),
        )

        acceleration = torch.zeros_like(velocity)
        jerk = torch.zeros_like(velocity)
        heading_change = torch.zeros_like(heading)
        if self.num_historical_steps > 1:
            velocity_pair_valid = csv_valid_mask[:, 1:] & csv_valid_mask[:, :-1]
            raw_acceleration = (
                velocity[:, 1:] - velocity[:, :-1]
            ) / self.observation_step_time
            acceleration[:, 1:] = torch.where(
                velocity_pair_valid.unsqueeze(-1),
                raw_acceleration,
                torch.zeros_like(raw_acceleration),
            )
            raw_heading_change = wrap_angle(
                heading[:, 1:] - heading[:, :-1]
            )
            heading_change[:, 1:] = torch.where(
                velocity_pair_valid,
                raw_heading_change,
                torch.zeros_like(raw_heading_change),
            )
        if self.num_historical_steps > 2:
            acceleration_valid = (
                csv_valid_mask[:, 2:]
                & csv_valid_mask[:, 1:-1]
                & csv_valid_mask[:, :-2]
            )
            raw_jerk = (
                acceleration[:, 2:] - acceleration[:, 1:-1]
            ) / self.observation_step_time
            jerk[:, 2:] = torch.where(
                acceleration_valid.unsqueeze(-1),
                raw_jerk,
                torch.zeros_like(raw_jerk),
            )
        return (
            velocity, speed, acceleration, jerk,
            heading, heading_change, velocity_theta,
        )

    def _build_base_vehicle_edges(self,
                                  position: torch.Tensor,
                                  visible_mask: torch.Tensor,
                                  batch: torch.Tensor,
                                  sparse_edge_index: torch.Tensor,
                                  sparse_edge_attr: torch.Tensor) -> tuple:
        """Build original same-time radius edges and retain sparse graph cues."""
        num_agents, num_steps = position.shape[:2]
        same_sample = batch.unsqueeze(1) == batch.unsqueeze(0)
        different_agent = ~torch.eye(
            num_agents, dtype=torch.bool, device=position.device,
        )
        visible_by_time = visible_mask.transpose(0, 1)
        valid_pair = (
            visible_by_time.unsqueeze(2)
            & visible_by_time.unsqueeze(1)
            & same_sample.unsqueeze(0)
            & different_agent.unsqueeze(0)
        )
        candidate = torch.nonzero(valid_pair, as_tuple=False)
        if candidate.numel() == 0:
            return (
                torch.empty((2, 0), dtype=torch.long, device=position.device),
                position.new_zeros((0, 6)),
            )
        time_index, source_agent, target_agent = candidate.unbind(dim=1)
        distance = torch.linalg.vector_norm(
            position[source_agent, time_index]
            - position[target_agent, time_index],
            dim=-1,
        )
        keep = distance < self.a2a_radius
        time_index = time_index[keep]
        source_agent = source_agent[keep]
        target_agent = target_agent[keep]
        distance = distance[keep]
        edge_index = torch.stack([
            time_index * num_agents + source_agent,
            time_index * num_agents + target_agent,
        ], dim=0)
        edge_attr = position.new_zeros((edge_index.size(1), 6))
        edge_attr[:, 0] = distance

        if sparse_edge_index.numel() != 0 and edge_index.numel() != 0:
            num_nodes = num_agents * num_steps
            base_key = edge_index[0] * num_nodes + edge_index[1]
            sparse_key = (
                sparse_edge_index[0] * num_nodes + sparse_edge_index[1]
            )
            sorted_key, permutation = base_key.sort()
            match_position = torch.searchsorted(sorted_key, sparse_key)
            safe_position = match_position.clamp(
                max=sorted_key.numel() - 1,
            )
            matched = (
                (match_position < sorted_key.numel())
                & (sorted_key[safe_position] == sparse_key)
            )
            edge_attr[permutation[safe_position[matched]]] = (
                sparse_edge_attr[matched]
            )
        return edge_index, edge_attr

    @staticmethod
    def _expand_vehicle_edges_across_modes(edge_index: torch.Tensor,
                                           edge_attr: torch.Tensor,
                                           num_agents: int,
                                           num_modes: int) -> tuple:
        if edge_index.numel() == 0:
            return edge_index, edge_attr
        source, target = edge_index
        source_time = torch.div(source, num_agents, rounding_mode='floor')
        target_time = torch.div(target, num_agents, rounding_mode='floor')
        source_agent = source % num_agents
        target_agent = target % num_agents
        mode = torch.arange(num_modes, device=source.device).unsqueeze(0)
        expanded_source = (
            (source_time.unsqueeze(1) * num_modes + mode) * num_agents
            + source_agent.unsqueeze(1)
        )
        expanded_target = (
            (target_time.unsqueeze(1) * num_modes + mode) * num_agents
            + target_agent.unsqueeze(1)
        )
        return (
            torch.stack([
                expanded_source.reshape(-1),
                expanded_target.reshape(-1),
            ], dim=0),
            edge_attr.repeat_interleave(num_modes, dim=0),
        )

    def _build_base_lane_edges(self,
                               lane_position: torch.Tensor,
                               lane_batch: torch.Tensor,
                               lane_visible_mask: torch.Tensor,
                               target_position: torch.Tensor,
                               target_batch: torch.Tensor,
                               target_visible_mask: torch.Tensor) -> torch.Tensor:
        """Build original radius-based lane-to-target edges in bounded chunks."""
        if lane_position.numel() == 0 or target_position.numel() == 0:
            return torch.empty(
                (2, 0), dtype=torch.long, device=lane_position.device,
            )
        edge_chunks = []
        for start in range(0, target_position.size(0), 2048):
            end = min(start + 2048, target_position.size(0))
            same_sample = (
                lane_batch.unsqueeze(1)
                == target_batch[start:end].unsqueeze(0)
            )
            valid = (
                lane_visible_mask.unsqueeze(1)
                & target_visible_mask[start:end].unsqueeze(0)
                & same_sample
            )
            distance = torch.cdist(
                lane_position, target_position[start:end],
            )
            lane_index, local_target = torch.nonzero(
                valid & (distance < self.l2a_radius),
                as_tuple=True,
            )
            if lane_index.numel() != 0:
                edge_chunks.append(torch.stack([
                    lane_index,
                    local_target + start,
                ], dim=0))
        if not edge_chunks:
            return torch.empty(
                (2, 0), dtype=torch.long, device=lane_position.device,
            )
        return torch.cat(edge_chunks, dim=1)

    @staticmethod
    def _match_reachability_weight(base_edge_index: torch.Tensor,
                                   reachable_edge_index: torch.Tensor,
                                   reachable_weight: torch.Tensor,
                                   num_targets: int,
                                   dtype: torch.dtype) -> torch.Tensor:
        result = torch.zeros(
            base_edge_index.size(1),
            dtype=dtype,
            device=base_edge_index.device,
        )
        if base_edge_index.numel() == 0 or reachable_edge_index.numel() == 0:
            return result
        base_key = base_edge_index[0] * num_targets + base_edge_index[1]
        reachable_key = (
            reachable_edge_index[0] * num_targets
            + reachable_edge_index[1]
        )
        sorted_key, permutation = base_key.sort()
        match_position = torch.searchsorted(sorted_key, reachable_key)
        safe_position = match_position.clamp(max=sorted_key.numel() - 1)
        matched = (
            (match_position < sorted_key.numel())
            & (sorted_key[safe_position] == reachable_key)
        )
        result[permutation[safe_position[matched]]] = reachable_weight[matched]
        return result

    def _apply_style_film(self,
                          node_embedding: torch.Tensor,
                          style_prefix_embedding: torch.Tensor,
                          film_generator: nn.Module) -> torch.Tensor:
        """Apply causal per-anchor FiLM to ``[agent,time,mode]`` nodes."""
        film_parameter = film_generator(style_prefix_embedding)
        gamma, beta = film_parameter.chunk(2, dim=-1)
        if self.style_film_mode == 'bounded_identity':
            gamma = self.style_film_bound * torch.tanh(gamma)
            beta = self.style_film_bound * torch.tanh(beta)
        gamma = gamma.unsqueeze(2).expand(
            -1, -1, self.num_modes, -1,
        ).reshape(-1, self.hidden_dim)
        beta = beta.unsqueeze(2).expand(
            -1, -1, self.num_modes, -1,
        ).reshape(-1, self.hidden_dim)
        if self.style_film_mode == 'bounded_identity':
            return node_embedding + gamma * node_embedding + beta
        return (1.0 + gamma) * node_embedding + beta

    def _apply_conflict_attention(self,
                                  node_embedding: torch.Tensor,
                                  conflict_edge_index: torch.Tensor,
                                  conflict_edge_embedding: torch.Tensor,
                                  priority_head_bias: torch.Tensor,
                                  attention_layer: nn.Module,
                                  num_agents: int) -> torch.Tensor:
        """Apply final-step conflict attention within the configured scope."""
        if conflict_edge_index.numel() == 0:
            return node_embedding
        conflict_output = attention_layer(
            x=node_embedding,
            edge_index=conflict_edge_index,
            edge_attr=conflict_edge_embedding,
            attention_bias=priority_head_bias,
        )
        if self.conflict_update_scope == 'legacy':
            update_mask = torch.zeros(
                (self.num_historical_steps, self.num_modes, num_agents),
                dtype=torch.bool,
                device=node_embedding.device,
            )
            update_mask[self.num_historical_steps - 1] = True
            update_mask = update_mask.reshape(-1)
        else:
            update_mask = torch.zeros(
                node_embedding.size(0),
                dtype=torch.bool,
                device=node_embedding.device,
            )
            update_mask[conflict_edge_index.reshape(-1)] = True
        return torch.where(
            update_mask.unsqueeze(-1),
            conflict_output,
            node_embedding,
        )

    def _condition_proposal_conflict_edges(
            self,
            edge_embedding: torch.Tensor) -> torch.Tensor:
        """Apply legacy and isolated mode conditioning to proposal edges.

        Expanded conflict edges are ordered edge-major and mode-minor.  A
        zero scale for both paths returns the original tensor directly.  The
        legacy path reuses global mode tokens for version-20 reproduction;
        the adapter path uses independently trainable zero-initialized tokens.
        """
        if (
            (
                self.proposal_conflict_mode_scale == 0.0
                and self.proposal_conflict_mode_adapter_scale == 0.0
            )
            or edge_embedding.numel() == 0
        ):
            return edge_embedding
        if edge_embedding.size(0) % self.num_modes != 0:
            raise ValueError(
                'expanded conflict edge count must be divisible by num_modes'
            )
        num_conflict_edges = edge_embedding.size(0) // self.num_modes
        mode_index = torch.arange(
            self.num_modes,
            device=edge_embedding.device,
        ).repeat(num_conflict_edges)
        conditioned_embedding = edge_embedding
        if self.proposal_conflict_mode_scale > 0.0:
            legacy_mode_residual = torch.tanh(
                self.mode_tokens.weight[mode_index]
            )
            conditioned_embedding = conditioned_embedding * (
                1.0
                + self.proposal_conflict_mode_scale * legacy_mode_residual
            )
        if self.proposal_conflict_mode_adapter_scale > 0.0:
            adapter_mode_residual = torch.tanh(
                self.proposal_conflict_mode_tokens.weight[mode_index]
            )
            conditioned_embedding = conditioned_embedding * (
                1.0
                + self.proposal_conflict_mode_adapter_scale
                * adapter_mode_residual
            )
        return conditioned_embedding

    def _gate_proposal_conflict_residual(
            self,
            conflict_input: torch.Tensor,
            conflict_output: torch.Tensor,
            conflict_edge_index: torch.Tensor,
            num_agents: int) -> torch.Tensor:
        """Apply a mode-specific adjustment to participant residuals only.

        Proposal conflict nodes use the flattened ``[H,K,N,D]`` layout.
        The zero-initialized gate is added to the existing attention output,
        so historical checkpoints retain their exact forward semantics while
        the gate still receives a first-step gradient.
        """
        if (
            self.proposal_conflict_residual_gate_scale == 0.0
            or conflict_edge_index.numel() == 0
        ):
            return conflict_output
        expected_nodes = (
            self.num_historical_steps * self.num_modes * num_agents
        )
        if conflict_input.shape != conflict_output.shape:
            raise ValueError('proposal conflict input/output shapes must match')
        if conflict_input.ndim != 2 or conflict_input.size(0) != expected_nodes:
            raise ValueError(
                'proposal conflict nodes must use flattened [H,K,N,D] layout'
            )

        mode_index = torch.arange(
            self.num_modes,
            device=conflict_output.device,
        ).view(1, self.num_modes, 1).expand(
            self.num_historical_steps,
            self.num_modes,
            num_agents,
        ).reshape(-1)
        gate = torch.tanh(
            self.proposal_conflict_residual_gate_tokens.weight[mode_index]
        )
        participant_mask = torch.zeros(
            conflict_output.size(0),
            dtype=torch.bool,
            device=conflict_output.device,
        )
        participant_mask[conflict_edge_index.reshape(-1)] = True
        residual = conflict_output - conflict_input
        adjustment = (
            self.proposal_conflict_residual_gate_scale * gate * residual
        )
        adjustment = torch.where(
            participant_mask.unsqueeze(-1),
            adjustment,
            torch.zeros_like(adjustment),
        )
        return conflict_output + adjustment

    def _coordinate_scene_modes(
            self,
            node_embedding: torch.Tensor,
            agent_batch: torch.Tensor,
            valid_mask: torch.Tensor,
            num_scenes: int) -> torch.Tensor:
        """Add a bounded scene-conditioned residual to proposal modes.

        ``node_embedding`` uses flattened ``[N,H,K,D]`` order. For each
        scene, historical step and mode, visible agent embeddings are mean
        pooled. One shared residual is then broadcast to all visible agents in
        that scene-mode hypothesis. The output layer is zero-initialized, so
        enabling the branch preserves old checkpoints exactly while still
        providing a first-step gradient.
        """
        if self.scene_mode_coordination_scale == 0.0:
            return node_embedding
        num_agents = agent_batch.numel()
        expected_shape = (
            num_agents
            * self.num_historical_steps
            * self.num_modes,
            self.hidden_dim,
        )
        if node_embedding.shape != expected_shape:
            raise ValueError(
                'scene mode nodes must use flattened [N,H,K,D] layout'
            )
        expected_mask_shape = (
            num_agents,
            self.num_historical_steps,
            self.num_modes,
        )
        if valid_mask.shape != expected_mask_shape:
            raise ValueError(
                'scene mode valid mask must have shape [N,H,K]'
            )
        if agent_batch.ndim != 1 or agent_batch.dtype != torch.long:
            raise ValueError('agent_batch must be a one-dimensional long tensor')
        if num_scenes <= 0:
            raise ValueError('num_scenes must be positive')
        if agent_batch.numel() and (
                agent_batch.min() < 0 or agent_batch.max() >= num_scenes):
            raise ValueError('agent_batch contains an invalid scene index')

        node_view = node_embedding.reshape(
            num_agents,
            self.num_historical_steps,
            self.num_modes,
            self.hidden_dim,
        )
        valid_weight = valid_mask.to(node_embedding.dtype).unsqueeze(-1)
        scene_sum = node_embedding.new_zeros(
            num_scenes,
            self.num_historical_steps,
            self.num_modes,
            self.hidden_dim,
        )
        scene_sum.index_add_(0, agent_batch, node_view * valid_weight)
        scene_count = node_embedding.new_zeros(
            num_scenes,
            self.num_historical_steps,
            self.num_modes,
            1,
        )
        scene_count.index_add_(0, agent_batch, valid_weight)
        scene_context = scene_sum / scene_count.clamp_min(1.0)
        agent_context = scene_context[agent_batch]
        residual = torch.tanh(
            self.scene_mode_coordination(agent_context)
        )
        residual = residual * valid_weight
        coordinated = (
            node_view
            + self.scene_mode_coordination_scale * residual
        )
        return coordinated.reshape(-1, self.hidden_dim)

    @staticmethod
    def _sinkhorn_joint_mode_assignment(
            assignment_logits: torch.Tensor,
            num_iterations: int = 8) -> torch.Tensor:
        """Return a differentiable approximately doubly stochastic matrix."""
        if assignment_logits.ndim < 2:
            raise ValueError('assignment_logits must have at least two axes')
        if assignment_logits.size(-1) != assignment_logits.size(-2):
            raise ValueError('assignment_logits must end in a square matrix')
        if num_iterations <= 0:
            raise ValueError('num_iterations must be positive')
        log_assignment = assignment_logits
        for _ in range(num_iterations):
            log_assignment = log_assignment - torch.logsumexp(
                log_assignment, dim=-1, keepdim=True,
            )
            log_assignment = log_assignment - torch.logsumexp(
                log_assignment, dim=-2, keepdim=True,
            )
        return log_assignment.exp()

    def _joint_mode_permutation_scores(
            self, assignment_logits: torch.Tensor) -> torch.Tensor:
        """Score every local-mode permutation for each leading index."""
        if self.joint_mode_permutations.numel() == 0:
            raise RuntimeError(
                'permutation table is unavailable for the current mode'
            )
        flat_logits = assignment_logits.reshape(
            -1, self.num_modes, self.num_modes,
        )
        permutation_count = self.joint_mode_permutations.size(0)
        gather_index = self.joint_mode_permutations.view(
            1, permutation_count, self.num_modes, 1,
        ).expand(flat_logits.size(0), -1, -1, -1)
        candidate_logits = flat_logits.unsqueeze(1).expand(
            -1, permutation_count, -1, -1,
        )
        permutation_scores = torch.gather(
            candidate_logits, dim=-1, index=gather_index,
        ).squeeze(-1).sum(dim=-1)
        return permutation_scores.reshape(
            *assignment_logits.shape[:-2], permutation_count,
        )

    def _hard_permutation_assignment(
            self, assignment_logits: torch.Tensor) -> torch.Tensor:
        """Find the maximum-score bijection for every leading index."""
        permutation_scores = self._joint_mode_permutation_scores(
            assignment_logits,
        )
        best_permutation = permutation_scores.argmax(dim=-1)
        return self.joint_mode_permutations[best_permutation]

    def _limit_joint_mode_permutation_exceptions(
            self,
            hard_index: torch.Tensor,
            permutation_gain: torch.Tensor,
            agent_batch: torch.Tensor,
            anchor_valid: torch.Tensor,
            exception_agent_mask: torch.Tensor,
            num_scenes: int) -> torch.Tensor:
        """Keep only the strongest positive permutation gains per scene."""
        expected_index_shape = (
            agent_batch.numel(),
            self.num_historical_steps,
            self.num_modes,
        )
        if hard_index.shape != expected_index_shape:
            raise ValueError(
                'hard permutation indices must have shape [N,H,K]'
            )
        if anchor_valid.shape != expected_index_shape[:2]:
            raise ValueError('anchor_valid must have shape [N,H]')
        if permutation_gain.shape != expected_index_shape[:2]:
            raise ValueError('permutation_gain must have shape [N,H]')
        if (
                exception_agent_mask.shape != (agent_batch.numel(),)
                or exception_agent_mask.dtype != torch.bool):
            raise ValueError(
                'exception_agent_mask must be a boolean tensor of shape [N]'
            )

        identity_index = torch.arange(
            self.num_modes,
            device=hard_index.device,
        ).view(1, 1, self.num_modes)
        nonidentity = (hard_index != identity_index).any(dim=-1)
        candidate = (
            anchor_valid
            & exception_agent_mask.unsqueeze(-1)
            & nonidentity
            & (permutation_gain > 0)
        )

        selected = torch.zeros_like(candidate)
        top_count = min(
            self.joint_mode_max_exception_agents,
            agent_batch.numel(),
        )
        if top_count > 0:
            negative_infinity = torch.full_like(
                permutation_gain, -torch.inf,
            )
            for scene_index in range(num_scenes):
                scene_candidate = candidate & (
                    agent_batch == scene_index
                ).unsqueeze(-1)
                ranked_gain = torch.where(
                    scene_candidate,
                    permutation_gain,
                    negative_infinity,
                )
                top_gain, top_agent = ranked_gain.topk(
                    top_count, dim=0,
                )
                scene_selected = torch.zeros_like(selected)
                scene_selected.scatter_(
                    dim=0,
                    index=top_agent,
                    src=torch.isfinite(top_gain),
                )
                selected |= scene_selected

        return torch.where(
            selected.unsqueeze(-1),
            hard_index,
            identity_index.expand_as(hard_index),
        )

    def _compose_joint_modes(
            self,
            node_embedding: torch.Tensor,
            trajectory_propose: torch.Tensor,
            trajectory_refine: torch.Tensor,
            agent_batch: torch.Tensor,
            valid_mask: torch.Tensor,
            num_scenes: int,
            exception_agent_mask: Optional[torch.Tensor] = None) -> tuple:
        """Compose agent-local trajectories into scene-level joint modes.

        The composer predicts, for every agent and joint-mode slot, which of
        that agent's existing local modes should occupy the slot. Forward uses
        a hard one-hot selection so trajectories are never averaged. During
        training, the hard choice uses a straight-through soft assignment:
        row-wise softmax for independent mode and doubly stochastic Sinkhorn
        for permutation modes. Limited permutation additionally keeps only
        the strongest positive-gain predicted-agent exceptions in each scene
        and anchor. Zero-initialized logits receive a small fixed diagonal
        preference, making an old checkpoint an exact identity mapping at
        startup.
        """
        if self.joint_mode_composition_scale == 0.0:
            return trajectory_propose, trajectory_refine
        num_agents = agent_batch.numel()
        expected_node_shape = (
            num_agents
            * self.num_historical_steps
            * self.num_modes,
            self.hidden_dim,
        )
        expected_trajectory_shape = (
            num_agents,
            self.num_historical_steps,
            self.num_modes,
            self.num_future_steps,
            2,
        )
        if node_embedding.shape != expected_node_shape:
            raise ValueError(
                'joint mode nodes must use flattened [N,H,K,D] layout'
            )
        if (
                trajectory_propose.shape != expected_trajectory_shape
                or trajectory_refine.shape != expected_trajectory_shape):
            raise ValueError(
                'joint mode trajectories must have shape [N,H,K,F,2]'
            )
        if valid_mask.shape != expected_trajectory_shape[:3]:
            raise ValueError(
                'joint mode valid mask must have shape [N,H,K]'
            )
        if agent_batch.ndim != 1 or agent_batch.dtype != torch.long:
            raise ValueError('agent_batch must be a one-dimensional long tensor')
        if num_scenes <= 0:
            raise ValueError('num_scenes must be positive')
        if agent_batch.numel() and (
                agent_batch.min() < 0 or agent_batch.max() >= num_scenes):
            raise ValueError('agent_batch contains an invalid scene index')
        if exception_agent_mask is None:
            exception_agent_mask = torch.ones(
                num_agents,
                dtype=torch.bool,
                device=agent_batch.device,
            )
        if (
                exception_agent_mask.shape != (num_agents,)
                or exception_agent_mask.dtype != torch.bool):
            raise ValueError(
                'exception_agent_mask must be a boolean tensor of shape [N]'
            )

        node_view = node_embedding.reshape(
            num_agents,
            self.num_historical_steps,
            self.num_modes,
            self.hidden_dim,
        )
        valid_weight = valid_mask.to(node_embedding.dtype).unsqueeze(-1)
        scene_sum = node_embedding.new_zeros(
            num_scenes,
            self.num_historical_steps,
            self.num_modes,
            self.hidden_dim,
        )
        scene_sum.index_add_(0, agent_batch, node_view * valid_weight)
        scene_count = node_embedding.new_zeros(
            num_scenes,
            self.num_historical_steps,
            self.num_modes,
            1,
        )
        scene_count.index_add_(0, agent_batch, valid_weight)
        scene_context = scene_sum / scene_count.clamp_min(1.0)
        composer_input = torch.cat([
            node_view,
            scene_context[agent_batch],
        ], dim=-1)
        learned_logits = self.joint_mode_composer(composer_input)
        identity = torch.eye(
            self.num_modes,
            dtype=learned_logits.dtype,
            device=learned_logits.device,
        ).view(1, 1, self.num_modes, self.num_modes)
        # A small margin breaks the initial argmax tie without saturating the
        # soft assignment used by the straight-through backward pass.
        assignment_logits = learned_logits + 0.1 * identity
        if self.joint_mode_assignment_mode == 'independent':
            soft_assignment = torch.softmax(assignment_logits, dim=-1)
            soft_assignment = torch.where(
                valid_mask.unsqueeze(-1),
                soft_assignment,
                identity.expand_as(soft_assignment),
            )
            hard_index = soft_assignment.argmax(dim=-1)
        else:
            # All local modes at a historical anchor share validity.  A
            # bijection therefore remains meaningful only when that anchor is
            # valid; invalid anchors preserve the input ordering exactly.
            anchor_valid = valid_mask.all(dim=-1)
            soft_assignment = self._sinkhorn_joint_mode_assignment(
                assignment_logits,
            )
            soft_assignment = torch.where(
                anchor_valid.unsqueeze(-1).unsqueeze(-1),
                soft_assignment,
                identity.expand_as(soft_assignment),
            )
            if self.joint_mode_assignment_mode == 'limited_permutation':
                permutation_scores = self._joint_mode_permutation_scores(
                    assignment_logits,
                )
                best_permutation = permutation_scores.argmax(dim=-1)
                hard_index = self.joint_mode_permutations[best_permutation]
                identity_score = assignment_logits.diagonal(
                    dim1=-2, dim2=-1,
                ).sum(dim=-1)
                permutation_gain = (
                    permutation_scores.max(dim=-1).values
                    - identity_score
                )
                hard_index = (
                    self._limit_joint_mode_permutation_exceptions(
                        hard_index=hard_index,
                        permutation_gain=permutation_gain,
                        agent_batch=agent_batch,
                        anchor_valid=anchor_valid,
                        exception_agent_mask=exception_agent_mask,
                        num_scenes=num_scenes,
                    )
                )
            else:
                hard_index = self._hard_permutation_assignment(
                    assignment_logits,
                )
            identity_index = torch.arange(
                self.num_modes,
                device=hard_index.device,
            ).view(1, 1, self.num_modes)
            hard_index = torch.where(
                anchor_valid.unsqueeze(-1),
                hard_index,
                identity_index.expand_as(hard_index),
            )

        def compose(trajectory: torch.Tensor) -> torch.Tensor:
            gather_index = hard_index.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, self.num_future_steps, 2,
            )
            hard_selected = torch.gather(
                trajectory, dim=2, index=gather_index,
            )
            soft_selected = torch.einsum(
                'nhkj,nhjfc->nhkfc', soft_assignment, trajectory,
            )
            selected = hard_selected + (
                soft_selected - soft_selected.detach()
            )
            if self.joint_mode_composition_scale == 1.0:
                return selected
            return trajectory + self.joint_mode_composition_scale * (
                selected - trajectory
            )

        return compose(trajectory_propose), compose(trajectory_refine)

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs):
        """Migrate old checkpoints with exact-zero isolated parameters."""
        adapter_key = prefix + 'proposal_conflict_mode_tokens.weight'
        if adapter_key not in state_dict:
            state_dict[adapter_key] = torch.zeros_like(
                self.proposal_conflict_mode_tokens.weight,
            )
        residual_gate_key = (
            prefix + 'proposal_conflict_residual_gate_tokens.weight'
        )
        if residual_gate_key not in state_dict:
            state_dict[residual_gate_key] = torch.zeros_like(
                self.proposal_conflict_residual_gate_tokens.weight,
            )
        coordination_weight_key = (
            prefix + 'scene_mode_coordination.weight'
        )
        if coordination_weight_key not in state_dict:
            state_dict[coordination_weight_key] = torch.zeros_like(
                self.scene_mode_coordination.weight,
            )
        coordination_bias_key = prefix + 'scene_mode_coordination.bias'
        if coordination_bias_key not in state_dict:
            state_dict[coordination_bias_key] = torch.zeros_like(
                self.scene_mode_coordination.bias,
            )
        composer_weight_key = prefix + 'joint_mode_composer.weight'
        if composer_weight_key not in state_dict:
            state_dict[composer_weight_key] = torch.zeros_like(
                self.joint_mode_composer.weight,
            )
        composer_bias_key = prefix + 'joint_mode_composer.bias'
        if composer_bias_key not in state_dict:
            state_dict[composer_bias_key] = torch.zeros_like(
                self.joint_mode_composer.bias,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _priority_trajectory_inputs(
            self,
            directed_pair_probability: torch.Tensor,
            directed_head_bias: torch.Tensor) -> tuple:
        """Select whether learned Priority outputs enter trajectory paths."""
        if self.priority_injection_mode == 'legacy':
            return directed_pair_probability, directed_head_bias
        return (
            torch.zeros_like(directed_pair_probability),
            torch.zeros_like(directed_head_bias),
        )

    def _get_lane_edge_index(self,
                             data: Batch,
                             edge_name: str,
                             device: torch.device) -> torch.Tensor:
        lane_edge_store = data['lane', 'lane']
        if edge_name in lane_edge_store:
            return lane_edge_store[edge_name]
        return torch.empty((2, 0), dtype=torch.long, device=device)

    def _build_agent_to_lane_edges(self,
                                   agent_position: torch.Tensor,
                                   agent_heading: torch.Tensor,
                                   agent_speed: torch.Tensor,
                                   agent_batch: torch.Tensor,
                                   agent_visible_mask: torch.Tensor,
                                   lane_position: torch.Tensor,
                                   lane_heading: torch.Tensor,
                                   lane_batch: torch.Tensor,
                                   lane_visible_mask: torch.Tensor) -> tuple:
        valid_mask = agent_visible_mask.unsqueeze(1) & lane_visible_mask.unsqueeze(0)
        valid_mask = valid_mask & (agent_batch.unsqueeze(1) == lane_batch.unsqueeze(0))
        distances = torch.norm(agent_position.unsqueeze(1) - lane_position.unsqueeze(0), p=2, dim=-1)
        perception_radius = self.traffic_base_radius + self.traffic_speed_scale * agent_speed * self.prediction_window
        valid_mask = valid_mask & (distances < perception_radius.unsqueeze(1))
        heading_diff = wrap_angle(agent_heading.unsqueeze(1) - lane_heading.unsqueeze(0))
        valid_mask = valid_mask & (heading_diff.abs() < torch.pi / 3)

        edge_index = dense_to_sparse(valid_mask)[0]
        if edge_index.numel() == 0:
            edge_attr_input = agent_position.new_zeros((0, 4))
            return edge_index, self.a2l_emb_layer(input=edge_attr_input)

        edge_vector = transform_point_to_local_coordinate(
            agent_position[edge_index[0]],
            lane_position[edge_index[1]],
            lane_heading[edge_index[1]],
        )
        edge_attr_length, edge_attr_theta = compute_angles_lengths_2D(edge_vector)
        edge_attr_heading = wrap_angle(agent_heading[edge_index[0]] - lane_heading[edge_index[1]])
        edge_attr_speed = agent_speed[edge_index[0]]
        edge_attr_input = torch.stack([edge_attr_length, edge_attr_theta, edge_attr_heading, edge_attr_speed], dim=-1)
        return edge_index, self.a2l_emb_layer(input=edge_attr_input)

    def _build_main_agent_to_lane_edges(
            self,
            main_lane: torch.Tensor,
            agent_position: torch.Tensor,
            agent_heading: torch.Tensor,
            agent_speed: torch.Tensor,
            agent_visible_mask: torch.Tensor,
            lane_position: torch.Tensor,
            lane_heading: torch.Tensor,
            lane_visible_mask: torch.Tensor) -> tuple:
        """Build exactly one Vehicle->Lane edge for each valid main assignment."""
        valid_agent = (
            agent_visible_mask
            & (main_lane >= 0)
        )
        agent_index = torch.nonzero(valid_agent, as_tuple=False).flatten()
        if agent_index.numel() != 0:
            assigned_lane = main_lane[agent_index]
            edge_valid = lane_visible_mask[assigned_lane]
            agent_index = agent_index[edge_valid]
            assigned_lane = assigned_lane[edge_valid]
        else:
            assigned_lane = torch.empty(
                0, dtype=torch.long, device=main_lane.device,
            )
        edge_index = torch.stack([agent_index, assigned_lane], dim=0)
        if edge_index.numel() == 0:
            return edge_index, self.a2l_emb_layer(
                input=agent_position.new_zeros((0, 4))
            )

        edge_vector = transform_point_to_local_coordinate(
            agent_position[agent_index],
            lane_position[assigned_lane],
            lane_heading[assigned_lane],
        )
        edge_length, edge_theta = compute_angles_lengths_2D(edge_vector)
        edge_heading = wrap_angle(
            agent_heading[agent_index] - lane_heading[assigned_lane]
        )
        edge_attr_input = torch.stack([
            edge_length,
            edge_theta,
            edge_heading,
            agent_speed[agent_index],
        ], dim=-1)
        return edge_index, self.a2l_emb_layer(input=edge_attr_input)

    def _compute_current_lane_average_speed(self,
                                            agent_position: torch.Tensor,
                                            agent_speed: torch.Tensor,
                                            agent_batch: torch.Tensor,
                                            agent_visible_mask: torch.Tensor,
                                            lane_position: torch.Tensor,
                                            lane_batch: torch.Tensor) -> torch.Tensor:
        lane_avg_speed = agent_speed.new_zeros(lane_position.size(0))
        if lane_position.numel() == 0 or agent_position.numel() == 0:
            return lane_avg_speed

        valid_agent_mask = agent_visible_mask
        if not valid_agent_mask.any():
            return lane_avg_speed

        assigned_lane = self._assign_nodes_to_lanes(
            agent_position[valid_agent_mask],
            agent_batch[valid_agent_mask],
            lane_position,
            lane_batch,
        )
        return self._compute_lane_average_speed(assigned_lane, agent_speed[valid_agent_mask], lane_position.size(0))

    def _compute_adjacent_lane_distance(self,
                                        lane_position: torch.Tensor,
                                        successor_edge_index: torch.Tensor,
                                        predecessor_edge_index: torch.Tensor) -> torch.Tensor:
        num_lanes = lane_position.size(0)
        adjacent_distance = lane_position.new_full((num_lanes,), 1.0)
        if num_lanes == 0:
            return adjacent_distance

        direct_edge_indices = []
        if successor_edge_index.numel() != 0:
            direct_edge_indices.append(successor_edge_index)
        if predecessor_edge_index.numel() != 0:
            direct_edge_indices.append(predecessor_edge_index)
        if len(direct_edge_indices) == 0:
            return adjacent_distance

        edge_index = torch.cat(direct_edge_indices, dim=1)
        src, dst = edge_index
        edge_distance = torch.norm(lane_position[src] - lane_position[dst], p=2, dim=-1)

        distance_sum = lane_position.new_zeros(num_lanes)
        distance_count = lane_position.new_zeros(num_lanes)
        distance_sum.scatter_add_(0, dst, edge_distance)
        distance_count.scatter_add_(0, dst, torch.ones_like(edge_distance))

        fallback_distance = edge_distance.mean().clamp_min(self.ttc_speed_epsilon)
        adjacent_distance = torch.where(
            distance_count > 0,
            distance_sum / distance_count.clamp_min(1.0),
            fallback_distance.expand_as(distance_sum),
        )
        return adjacent_distance.clamp_min(self.ttc_speed_epsilon)

    def _build_lane_adjacency(self,
                              edge_index: torch.Tensor,
                              num_lanes: int) -> dict:
        adjacency = {lane_idx: [] for lane_idx in range(num_lanes)}
        if edge_index.numel() == 0:
            return adjacency

        for src, dst in edge_index.detach().cpu().t().tolist():
            if 0 <= src < num_lanes and 0 <= dst < num_lanes:
                adjacency[dst].append(src)
        return adjacency

    def _build_weighted_lane_adjacency(self,
                                       edge_index: torch.Tensor,
                                       lane_position: torch.Tensor) -> dict:
        num_lanes = lane_position.size(0)
        adjacency = {lane_idx: [] for lane_idx in range(num_lanes)}
        if edge_index.numel() == 0:
            return adjacency

        edge_distance = torch.norm(lane_position[edge_index[0]] - lane_position[edge_index[1]], p=2, dim=-1)
        edge_distance_cpu = edge_distance.detach().cpu().tolist()
        for (src, dst), distance in zip(edge_index.detach().cpu().t().tolist(), edge_distance_cpu):
            if 0 <= src < num_lanes and 0 <= dst < num_lanes:
                adjacency[dst].append((src, max(distance, self.ttc_speed_epsilon)))
        return adjacency

    def _select_style_reference_lanes(self,
                                      lane_position: torch.Tensor,
                                      lane_heading: torch.Tensor,
                                      lane_width: torch.Tensor,
                                      lane_batch: torch.Tensor,
                                      lane_visible_mask: torch.Tensor,
                                      agent_position: torch.Tensor,
                                      agent_heading: torch.Tensor,
                                      agent_speed: torch.Tensor,
                                      agent_batch: torch.Tensor,
                                      agent_visible_mask: torch.Tensor,
                                      successor_edge_index: torch.Tensor,
                                      predecessor_edge_index: torch.Tensor) -> tuple:
        """Select causal seed/reference lanes and per-node absolute lateral distance."""
        num_agents, num_steps = agent_position.shape[:2]
        num_lanes = lane_position.size(0)
        if num_lanes == 0:
            raise ValueError('vehicle style features require at least one lane in every sample')
        if lane_width.shape != (num_lanes,):
            raise ValueError('lane.width must have shape [num_lanes]')

        flat_position = agent_position.transpose(0, 1).reshape(-1, 2)
        flat_heading = agent_heading.transpose(0, 1).reshape(-1)
        flat_speed = agent_speed.transpose(0, 1).reshape(-1)
        flat_visible = agent_visible_mask.transpose(0, 1).reshape(-1)
        flat_batch = agent_batch.repeat(num_steps)
        num_nodes = flat_position.size(0)

        relative = flat_position.unsqueeze(0) - lane_position.unsqueeze(1)
        cos_heading = torch.cos(lane_heading).unsqueeze(1)
        sin_heading = torch.sin(lane_heading).unsqueeze(1)
        lateral_distance = (
            -relative[..., 0] * sin_heading
            + relative[..., 1] * cos_heading
        ).abs()
        total_distance = torch.linalg.vector_norm(relative, dim=-1)
        heading_valid = wrap_angle(
            lane_heading.unsqueeze(1) - flat_heading.unsqueeze(0)
        ).abs() <= torch.pi / 3
        base_valid = (
            lane_visible_mask.unsqueeze(1)
            & flat_visible.unsqueeze(0)
            & (lane_batch.unsqueeze(1) == flat_batch.unsqueeze(0))
            & heading_valid
        )
        direct_seed_mask = (
            base_valid
            & (lateral_distance < self.traffic_lane_lateral_radius)
            & (total_distance < self.traffic_current_radius)
        )
        topology_candidate_mask = (
            base_valid
            & (lateral_distance < self.traffic_lane_lateral_radius)
        )

        successor_adjacency = self._build_weighted_lane_adjacency(
            successor_edge_index,
            lane_position,
        )
        predecessor_adjacency = self._build_weighted_lane_adjacency(
            predecessor_edge_index,
            lane_position,
        )
        combined_adjacency = {lane_idx: [] for lane_idx in range(num_lanes)}
        for lane_idx in range(num_lanes):
            combined_adjacency[lane_idx].extend(successor_adjacency.get(lane_idx, []))
            combined_adjacency[lane_idx].extend(predecessor_adjacency.get(lane_idx, []))

        seed_lane = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=lane_position.device,
        )
        reference_lane = torch.full_like(seed_lane, -1)
        selected_lateral_distance = lane_position.new_zeros(num_nodes)

        lane_visible_cpu = lane_visible_mask.detach().cpu().tolist()
        lane_batch_cpu = lane_batch.detach().cpu().tolist()
        flat_batch_cpu = flat_batch.detach().cpu().tolist()
        flat_visible_cpu = flat_visible.detach().cpu().tolist()
        lateral_distance_cpu = lateral_distance.detach().cpu()
        topology_candidate_cpu = topology_candidate_mask.detach().cpu()
        direct_seed_cpu = direct_seed_mask.detach().cpu()
        reachable_distance_cpu = (
            self.traffic_topology_base_distance + flat_speed * self.prediction_window
        ).detach().cpu().tolist()

        for time_idx in range(num_steps):
            for agent_idx in range(num_agents):
                node_idx = time_idx * num_agents + agent_idx
                if not flat_visible_cpu[node_idx]:
                    continue

                direct_candidates = torch.nonzero(
                    direct_seed_cpu[:, node_idx],
                    as_tuple=False,
                ).flatten()
                selected_seed = -1
                if direct_candidates.numel() != 0:
                    candidate_lateral = lateral_distance_cpu[direct_candidates, node_idx]
                    selected_seed = int(direct_candidates[candidate_lateral.argmin()].item())
                elif time_idx > 0 and bool(agent_visible_mask[agent_idx, time_idx - 1]):
                    previous_node_idx = (time_idx - 1) * num_agents + agent_idx
                    previous_seed = int(seed_lane[previous_node_idx].item())
                    if previous_seed >= 0:
                        distance_limit = max(float(reachable_distance_cpu[node_idx]), 0.0)
                        best_distance = {previous_seed: 0.0}
                        frontier = [(0.0, previous_seed)]
                        topology_candidates = []
                        while frontier:
                            path_distance, lane_idx = heapq.heappop(frontier)
                            if path_distance > best_distance.get(lane_idx, float('inf')):
                                continue
                            if path_distance > distance_limit:
                                continue
                            if (
                                lane_visible_cpu[lane_idx]
                                and lane_batch_cpu[lane_idx] == flat_batch_cpu[node_idx]
                                and bool(topology_candidate_cpu[lane_idx, node_idx])
                            ):
                                topology_candidates.append(lane_idx)
                            for next_lane, step_distance in combined_adjacency.get(lane_idx, []):
                                if not lane_visible_cpu[next_lane]:
                                    continue
                                if lane_batch_cpu[next_lane] != flat_batch_cpu[node_idx]:
                                    continue
                                next_distance = path_distance + step_distance
                                if next_distance > distance_limit:
                                    continue
                                if next_distance < best_distance.get(next_lane, float('inf')):
                                    best_distance[next_lane] = next_distance
                                    heapq.heappush(frontier, (next_distance, next_lane))
                        if topology_candidates:
                            candidate_tensor = torch.tensor(topology_candidates, dtype=torch.long)
                            candidate_lateral = lateral_distance_cpu[candidate_tensor, node_idx]
                            selected_seed = int(candidate_tensor[candidate_lateral.argmin()].item())

                if selected_seed >= 0:
                    seed_lane[node_idx] = selected_seed
                    reference_lane[node_idx] = selected_seed
                    selected_lateral_distance[node_idx] = lateral_distance[selected_seed, node_idx]
                    continue

                nearest_mask = (
                    lane_visible_mask
                    & (lane_batch == flat_batch[node_idx])
                )
                nearest_candidates = torch.nonzero(nearest_mask, as_tuple=False).flatten()
                if nearest_candidates.numel() == 0:
                    raise ValueError('sample contains no visible lane for a vehicle style feature')
                nearest_distance = total_distance[nearest_candidates, node_idx]
                nearest_lane = int(nearest_candidates[nearest_distance.argmin()].item())
                reference_lane[node_idx] = nearest_lane
                selected_lateral_distance[node_idx] = lane_width[nearest_lane] * 0.5

        seed_lane = seed_lane.reshape(num_steps, num_agents).transpose(0, 1)
        reference_lane = reference_lane.reshape(num_steps, num_agents).transpose(0, 1)
        selected_lateral_distance = selected_lateral_distance.reshape(
            num_steps,
            num_agents,
        ).transpose(0, 1)
        return seed_lane, reference_lane, selected_lateral_distance

    def _build_vehicle_style_feature(self,
                                     interaction_graph,
                                     speed: torch.Tensor,
                                     acceleration: torch.Tensor,
                                     jerk: torch.Tensor,
                                     heading: torch.Tensor,
                                     heading_change: torch.Tensor,
                                     visible_mask: torch.Tensor,
                                     lane_lateral_distance: torch.Tensor) -> torch.Tensor:
        """Build an independent per-agent, per-timestep 13D driving-style tensor."""
        num_agents, num_steps = speed.shape
        num_nodes = num_agents * num_steps

        def node_to_agent_time(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(num_steps, num_agents).transpose(0, 1)

        degree_centrality = node_to_agent_time(interaction_graph.degree_centrality)
        closeness_centrality = node_to_agent_time(interaction_graph.closeness_centrality)
        degree_centrality_change = node_to_agent_time(
            interaction_graph.degree_centrality_change
        )
        closeness_centrality_change = node_to_agent_time(
            interaction_graph.closeness_centrality_change
        )

        node_thw = speed.new_full((num_nodes,), self.style_thw_max)
        node_ttc = speed.new_full((num_nodes,), self.style_ttc_max)
        best_front_gap = speed.new_full((num_nodes,), float('inf'))
        _, edge_target = interaction_graph.E_VV_msg
        for edge_idx in torch.nonzero(
            interaction_graph.safety_feature_valid_mask,
            as_tuple=False,
        ).flatten().detach().cpu().tolist():
            target_node = int(edge_target[edge_idx].item())
            front_gap = interaction_graph.safety_longitudinal_gap[edge_idx]
            if front_gap < best_front_gap[target_node]:
                best_front_gap[target_node] = front_gap
                node_thw[target_node] = interaction_graph.THW_ij[edge_idx]
                node_ttc[target_node] = interaction_graph.TTC_ij[edge_idx]

        node_arrival_difference = speed.new_full(
            (num_nodes,),
            self.style_arrival_time_fill,
        )
        best_arrival_vehicle_distance = speed.new_full((num_nodes,), float('inf'))
        if interaction_graph.edge_attr.numel() != 0:
            arrival_edge_mask = interaction_graph.edge_attr[:, 5].bool()
            for edge_idx in torch.nonzero(
                arrival_edge_mask,
                as_tuple=False,
            ).flatten().detach().cpu().tolist():
                target_node = int(edge_target[edge_idx].item())
                vehicle_distance = interaction_graph.D_VV_msg[edge_idx]
                if vehicle_distance < best_arrival_vehicle_distance[target_node]:
                    best_arrival_vehicle_distance[target_node] = vehicle_distance
                    node_arrival_difference[target_node] = (
                        interaction_graph.same_lane_arrival_time_difference[edge_idx]
                    )

        style_feature = torch.stack([
            speed,
            torch.linalg.vector_norm(acceleration, dim=-1),
            torch.linalg.vector_norm(jerk, dim=-1),
            heading,
            heading_change,
            degree_centrality,
            closeness_centrality,
            degree_centrality_change,
            closeness_centrality_change,
            node_to_agent_time(node_thw),
            node_to_agent_time(node_ttc),
            lane_lateral_distance,
            node_to_agent_time(node_arrival_difference),
        ], dim=-1)
        if style_feature.size(-1) != len(self.STYLE_FEATURE_NAMES):
            raise RuntimeError('vehicle style feature dimension does not match its schema')
        return torch.where(
            visible_mask.unsqueeze(-1),
            style_feature,
            torch.zeros_like(style_feature),
        )

    @staticmethod
    def _masked_mean(value: torch.Tensor,
                     valid_mask: torch.Tensor) -> torch.Tensor:
        valid_count = valid_mask.sum(dim=1)
        value_sum = torch.where(
            valid_mask,
            value,
            torch.zeros_like(value),
        ).sum(dim=1)
        return value_sum / valid_count.clamp_min(1).to(value.dtype)

    @staticmethod
    def _masked_quantile(value: torch.Tensor,
                         valid_mask: torch.Tensor,
                         quantile: float) -> torch.Tensor:
        if value.shape != valid_mask.shape or value.dim() != 2:
            raise ValueError('value and valid_mask must share shape [N, T]')
        if value.size(1) == 0:
            return value.new_zeros(value.size(0))

        valid_count = valid_mask.sum(dim=1)
        ordered_value = torch.where(
            valid_mask,
            value,
            torch.full_like(value, float('inf')),
        ).sort(dim=1).values
        rank = (
            (valid_count.clamp_min(1) - 1).to(value.dtype) * quantile
        )
        lower_index = rank.floor().long()
        upper_index = rank.ceil().long()
        lower_value = ordered_value.gather(
            1, lower_index.unsqueeze(1),
        ).squeeze(1)
        upper_value = ordered_value.gather(
            1, upper_index.unsqueeze(1),
        ).squeeze(1)
        nonempty = valid_count > 0
        lower_value = torch.where(
            nonempty, lower_value, torch.zeros_like(lower_value),
        )
        upper_value = torch.where(
            nonempty, upper_value, torch.zeros_like(upper_value),
        )
        # torch.quantile's default interpolation is linear. torch.lerp also
        # preserves its floating-point evaluation order for this batched form.
        return torch.lerp(
            lower_value,
            upper_value,
            rank - lower_index.to(value.dtype),
        )

    def _build_vehicle_style_indicators(
            self,
            style_feature: torch.Tensor,
            visible_mask: torch.Tensor,
            agent_type: torch.Tensor,
            velocity_valid_mask: torch.Tensor,
            acceleration_valid_mask: torch.Tensor,
            jerk_valid_mask: torch.Tensor,
            heading_valid_mask: torch.Tensor) -> tuple:
        """Aggregate the six train-standardized indicators used for pseudo labels."""
        if style_feature.shape[:2] != visible_mask.shape:
            raise ValueError('style_feature and visible_mask must share [num_agents, num_steps]')
        if style_feature.size(-1) != len(self.STYLE_FEATURE_NAMES):
            raise ValueError('style_feature must use the documented 13D schema')

        speed = style_feature[..., 0]
        acceleration_magnitude = style_feature[..., 1].abs()
        jerk_magnitude = style_feature[..., 2].abs()
        heading_change = style_feature[..., 4].abs()
        ttc = style_feature[..., 10]
        lateral_distance = style_feature[..., 11]

        heading_change_valid_mask = torch.zeros_like(visible_mask)
        heading_change_valid_mask[:, 1:] = (
            visible_mask[:, 1:]
            & visible_mask[:, :-1]
            & heading_valid_mask[:, 1:]
            & heading_valid_mask[:, :-1]
        )
        lateral_change = torch.zeros_like(lateral_distance)
        lateral_change[:, 1:] = (
            lateral_distance[:, 1:] - lateral_distance[:, :-1]
        ).abs() / self.observation_step_time
        lateral_change_valid_mask = torch.zeros_like(visible_mask)
        lateral_change_valid_mask[:, 1:] = (
            visible_mask[:, 1:] & visible_mask[:, :-1]
        )
        ttc_risk = 1.0 - (
            ttc.clamp(min=0.0, max=self.style_ttc_max) / self.style_ttc_max
        )

        indicators = torch.stack([
            self._masked_quantile(speed, velocity_valid_mask, 0.90),
            self._masked_quantile(
                acceleration_magnitude,
                acceleration_valid_mask,
                0.90,
            ),
            self._masked_mean(jerk_magnitude, jerk_valid_mask),
            self._masked_mean(heading_change, heading_change_valid_mask),
            self._masked_mean(lateral_change, lateral_change_valid_mask),
            self._masked_mean(ttc_risk, visible_mask),
        ], dim=-1)

        reliable_mask = (
            (agent_type == 0)
            & (visible_mask.sum(dim=1) >= self.style_min_visible_steps)
        )
        return indicators, reliable_mask

    def _build_vehicle_style_prefix_indicators(
            self,
            style_feature: torch.Tensor,
            visible_mask: torch.Tensor,
            agent_type: torch.Tensor,
            velocity_valid_mask: torch.Tensor,
            acceleration_valid_mask: torch.Tensor,
            jerk_valid_mask: torch.Tensor,
            heading_valid_mask: torch.Tensor) -> tuple:
        """Build causal style indicators and supervision masks for every anchor."""
        prefix_indicators = []
        prefix_reliable_masks = []
        for historical_step in range(style_feature.size(1)):
            prefix_slice = slice(0, historical_step + 1)
            indicators, reliable_mask = self._build_vehicle_style_indicators(
                style_feature=style_feature[:, prefix_slice],
                visible_mask=visible_mask[:, prefix_slice],
                agent_type=agent_type,
                velocity_valid_mask=velocity_valid_mask[:, prefix_slice],
                acceleration_valid_mask=acceleration_valid_mask[:, prefix_slice],
                jerk_valid_mask=jerk_valid_mask[:, prefix_slice],
                heading_valid_mask=heading_valid_mask[:, prefix_slice],
            )
            prefix_indicators.append(indicators)
            prefix_reliable_masks.append(reliable_mask)
        if not prefix_indicators:
            return (
                style_feature.new_zeros((style_feature.size(0), 0, 6)),
                torch.zeros(
                    (style_feature.size(0), 0),
                    dtype=torch.bool,
                    device=style_feature.device,
                ),
            )
        return (
            torch.stack(prefix_indicators, dim=1),
            torch.stack(prefix_reliable_masks, dim=1),
        )

    def _build_vehicle_style_pseudo_label(
            self,
            indicators: torch.Tensor,
            reliable_mask: torch.Tensor) -> tuple:
        """Apply train-only robust statistics and return score and hard class label."""
        standardized = torch.zeros_like(indicators)
        style_score = indicators.new_zeros(indicators.size(0))
        pseudo_label = torch.full(
            (indicators.size(0),),
            -1,
            dtype=torch.long,
            device=indicators.device,
        )
        if not bool(self.style_statistics_ready.item()):
            return standardized, style_score, pseudo_label

        standardized = (
            (indicators - self.style_indicator_median)
            / (self.style_indicator_iqr + self.style_statistics_epsilon)
        )
        style_score = (standardized * self.style_score_weights).sum(dim=-1)
        reliable_score = style_score[reliable_mask]
        reliable_label = torch.where(
            reliable_score <= self.style_score_q33,
            torch.full_like(reliable_score, 2, dtype=torch.long),
            torch.where(
                reliable_score >= self.style_score_q66,
                torch.zeros_like(reliable_score, dtype=torch.long),
                torch.ones_like(reliable_score, dtype=torch.long),
            ),
        )
        pseudo_label[reliable_mask] = reliable_label
        return standardized, style_score, pseudo_label

    def _build_style_segment_id(self,
                                visible_mask: torch.Tensor) -> torch.Tensor:
        """Assign temporal segments, starting a new one after a long gap."""
        num_agents, num_steps = visible_mask.shape
        segment_id = torch.zeros(
            (num_agents, num_steps),
            dtype=torch.long,
            device=visible_mask.device,
        )
        visible_cpu = visible_mask.detach().cpu().tolist()
        for agent_idx in range(num_agents):
            current_segment = 0
            consecutive_invisible = 0
            for time_idx in range(num_steps):
                if visible_cpu[agent_idx][time_idx]:
                    if consecutive_invisible >= self.style_gap_reset_steps:
                        current_segment += 1
                    consecutive_invisible = 0
                else:
                    consecutive_invisible += 1
                segment_id[agent_idx, time_idx] = current_segment
        return segment_id

    def _build_style_temporal_attention_mask(self,
                                             visible_mask: torch.Tensor) -> torch.Tensor:
        """Build a finite causal mask with at least one key for every query."""
        num_agents, num_steps = visible_mask.shape
        causal_mask = torch.triu(
            torch.ones(
                (num_steps, num_steps),
                dtype=torch.bool,
                device=visible_mask.device,
            ),
            diagonal=1,
        )
        attention_mask = causal_mask.unsqueeze(0).expand(
            num_agents,
            num_steps,
            num_steps,
        ).clone()

        segment_id = self._build_style_segment_id(visible_mask)
        different_segment = segment_id.unsqueeze(2) != segment_id.unsqueeze(1)
        invisible_key = (~visible_mask).unsqueeze(1).expand_as(attention_mask)
        attention_mask = attention_mask | different_segment | invisible_key

        # An invisible query before the first observation otherwise has no
        # available causal key. Its output is discarded, but the Transformer
        # must still receive a numerically valid attention row. Use the query's
        # own zero token only for rows that would be fully masked.
        fully_masked = attention_mask.all(dim=-1)
        if fully_masked.any():
            agent_index, query_index = torch.nonzero(
                fully_masked,
                as_tuple=True,
            )
            attention_mask[agent_index, query_index, query_index] = False
        return attention_mask.repeat_interleave(self.num_heads, dim=0)

    def _encode_vehicle_style_sequence(self,
                                       style_feature: torch.Tensor,
                                       visible_mask: torch.Tensor) -> torch.Tensor:
        """Encode style features causally while retaining one output per timestep."""
        if style_feature.shape[:2] != visible_mask.shape:
            raise ValueError('style_feature and visible_mask must share [num_agents, num_steps]')
        if style_feature.size(-1) != len(self.STYLE_FEATURE_NAMES):
            raise ValueError('style_feature must use the documented 13D schema')

        num_steps = style_feature.size(1)
        time_index = torch.arange(num_steps, device=style_feature.device)
        temporal_input = (
            self.style_input_projection(style_feature)
            + self.style_time_embedding(time_index).unsqueeze(0)
        )
        temporal_input = torch.where(
            visible_mask.unsqueeze(-1),
            temporal_input,
            torch.zeros_like(temporal_input),
        )
        attention_mask = self._build_style_temporal_attention_mask(visible_mask)

        temporal_feature = self.style_temporal_encoder(
            src=temporal_input,
            mask=attention_mask,
            is_causal=False,
        )
        return torch.where(
            visible_mask.unsqueeze(-1),
            temporal_feature,
            torch.zeros_like(temporal_feature),
        )

    def _attn_pool_style_segment(self,
                                 temporal_feature: torch.Tensor,
                                 valid_mask: torch.Tensor,
                                 return_attention_weights: bool) -> tuple:
        """Pool a masked temporal segment with the shared learned style query."""
        num_agents = temporal_feature.size(0)
        query = self.style_pool_query.weight.unsqueeze(0).expand(num_agents, -1, -1)
        padding_mask = ~valid_mask
        has_valid_step = valid_mask.any(dim=1)
        if (~has_valid_step).any():
            padding_mask = padding_mask.clone()
            padding_mask[~has_valid_step, 0] = False

        pooled, attention_weight = self.style_attn_pool(
            query=query,
            key=temporal_feature,
            value=temporal_feature,
            key_padding_mask=padding_mask,
            need_weights=return_attention_weights,
            average_attn_weights=True,
        )
        pooled = pooled.squeeze(1)
        pooled = torch.where(
            has_valid_step.unsqueeze(-1),
            pooled,
            torch.zeros_like(pooled),
        )
        if attention_weight is not None:
            attention_weight = attention_weight.squeeze(1)
            attention_weight = torch.where(
                valid_mask,
                attention_weight,
                torch.zeros_like(attention_weight),
            )
        return pooled, attention_weight

    def _pool_vehicle_style_embedding(self,
                                      temporal_feature: torch.Tensor,
                                      visible_mask: torch.Tensor,
                                      return_attention_weights: bool = False) -> tuple:
        """Blend last-segment and global AttnPool outputs, then map to style embedding."""
        segment_id = self._build_style_segment_id(visible_mask)
        visible_segment_id = segment_id.masked_fill(~visible_mask, -1)
        last_segment_id = visible_segment_id.max(dim=1).values
        last_segment_mask = (
            visible_mask
            & (segment_id == last_segment_id.unsqueeze(1))
            & (last_segment_id.unsqueeze(1) >= 0)
        )

        e_global, global_weight = self._attn_pool_style_segment(
            temporal_feature,
            visible_mask,
            return_attention_weights,
        )
        e_last, last_weight = self._attn_pool_style_segment(
            temporal_feature,
            last_segment_mask,
            return_attention_weights,
        )
        last_length = last_segment_mask.sum(dim=1).to(temporal_feature.dtype)
        blend_ratio = (last_length / self.style_pool_min_length).clamp(max=1.0)
        pooled_style = (
            blend_ratio.unsqueeze(-1) * e_last
            + (1.0 - blend_ratio).unsqueeze(-1) * e_global
        )
        style_embedding = self.style_embedding_mlp(pooled_style)
        has_visible_step = visible_mask.any(dim=1)
        style_embedding = torch.where(
            has_visible_step.unsqueeze(-1),
            style_embedding,
            torch.zeros_like(style_embedding),
        )

        attention_weight = None
        if return_attention_weights:
            attention_weight = (
                blend_ratio.unsqueeze(-1) * last_weight
                + (1.0 - blend_ratio).unsqueeze(-1) * global_weight
            )
        return style_embedding, attention_weight

    def _pool_vehicle_style_prefix_embeddings(
            self,
            temporal_feature: torch.Tensor,
            visible_mask: torch.Tensor) -> torch.Tensor:
        """Pool one causal style embedding for every historical anchor."""
        prefix_embedding = []
        for historical_step in range(temporal_feature.size(1)):
            embedding, _ = self._pool_vehicle_style_embedding(
                temporal_feature=temporal_feature[:, :historical_step + 1],
                visible_mask=visible_mask[:, :historical_step + 1],
                return_attention_weights=False,
            )
            prefix_embedding.append(embedding)
        if not prefix_embedding:
            return temporal_feature.new_zeros(
                (temporal_feature.size(0), 0, temporal_feature.size(-1))
            )
        return torch.stack(prefix_embedding, dim=1)

    def _collect_lanes_by_direction(self,
                                    adjacency: dict,
                                    target: int,
                                    max_nodes: int,
                                    visible_nodes: list) -> list:
        if max_nodes <= 0:
            return []

        selected_nodes = []
        visited = {target}
        queue = [(node, 1) for node in adjacency.get(target, [])]
        cursor = 0

        while cursor < len(queue) and len(selected_nodes) < max_nodes:
            node, hop = queue[cursor]
            cursor += 1
            if node in visited:
                continue
            visited.add(node)
            if not visible_nodes[node]:
                continue

            selected_nodes.append((node, hop))
            for next_node in adjacency.get(node, []):
                if next_node not in visited:
                    queue.append((next_node, hop + 1))

        return selected_nodes

    def _build_lane_traffic_edges(self,
                                  lane_position: torch.Tensor,
                                  lane_heading: torch.Tensor,
                                  lane_batch: torch.Tensor,
                                  lane_visible_mask: torch.Tensor,
                                  lane_avg_speed: torch.Tensor,
                                  successor_edge_index: torch.Tensor,
                                  predecessor_edge_index: torch.Tensor,
                                  conflict_edge_index: torch.Tensor) -> tuple:
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=lane_position.device)
        empty_edge_attr_input = lane_position.new_zeros((0, 5))
        if lane_position.numel() == 0:
            return empty_edge_index, self.l2l_traffic_emb_layer(input=empty_edge_attr_input)

        adjacent_distance = self._compute_adjacent_lane_distance(
            lane_position=lane_position,
            successor_edge_index=successor_edge_index,
            predecessor_edge_index=predecessor_edge_index,
        )
        attention_count = torch.ceil(lane_avg_speed * self.prediction_window / adjacent_distance).long().clamp_min(0)

        successor_adjacency = self._build_lane_adjacency(successor_edge_index, lane_position.size(0))
        predecessor_adjacency = self._build_lane_adjacency(predecessor_edge_index, lane_position.size(0))
        visible_nodes = lane_visible_mask.detach().cpu().tolist()
        lane_batch_cpu = lane_batch.detach().cpu().tolist()
        attention_count_cpu = attention_count.detach().cpu().tolist()
        backward_ratio = max(self.traffic_backward_attention_ratio, 0.0)
        max_forward_attention_nodes = max(int(self.traffic_max_attention_nodes), 0)

        edge_pairs = []
        edge_direction = []
        edge_hop = []
        selected_edges = set()

        for target in range(lane_position.size(0)):
            if not visible_nodes[target]:
                continue

            raw_attention_count = int(attention_count_cpu[target])
            forward_count = min(raw_attention_count, max_forward_attention_nodes)
            backward_count = int(torch.ceil(lane_avg_speed.new_tensor(raw_attention_count * backward_ratio)).item())
            max_backward_count = int(torch.ceil(lane_avg_speed.new_tensor(max_forward_attention_nodes * backward_ratio)).item())
            backward_count = min(backward_count, max_backward_count)

            forward_nodes = self._collect_lanes_by_direction(successor_adjacency, target, forward_count, visible_nodes)
            backward_nodes = self._collect_lanes_by_direction(predecessor_adjacency, target, backward_count, visible_nodes)

            for src, hop in forward_nodes:
                edge_key = (src, target)
                if lane_batch_cpu[src] != lane_batch_cpu[target] or edge_key in selected_edges:
                    continue
                selected_edges.add(edge_key)
                edge_pairs.append([src, target])
                edge_direction.append(1.0)
                edge_hop.append(float(hop))

            for src, hop in backward_nodes:
                edge_key = (src, target)
                if lane_batch_cpu[src] != lane_batch_cpu[target] or edge_key in selected_edges:
                    continue
                selected_edges.add(edge_key)
                edge_pairs.append([src, target])
                edge_direction.append(-1.0)
                edge_hop.append(float(hop))

        if conflict_edge_index.numel() != 0:
            visible_mask = lane_visible_mask[conflict_edge_index[0]] & lane_visible_mask[conflict_edge_index[1]]
            same_batch_mask = lane_batch[conflict_edge_index[0]] == lane_batch[conflict_edge_index[1]]
            valid_conflict_edge_index = conflict_edge_index[:, visible_mask & same_batch_mask]
            for src, dst in valid_conflict_edge_index.detach().cpu().t().tolist():
                edge_key = (src, dst)
                if edge_key in selected_edges:
                    continue
                selected_edges.add(edge_key)
                edge_pairs.append([src, dst])
                edge_direction.append(0.0)
                edge_hop.append(1.0)

        if len(edge_pairs) == 0:
            return empty_edge_index, self.l2l_traffic_emb_layer(input=empty_edge_attr_input)

        edge_index = torch.tensor(edge_pairs, dtype=torch.long, device=lane_position.device).t().contiguous()
        edge_vector = transform_point_to_local_coordinate(
            lane_position[edge_index[0]],
            lane_position[edge_index[1]],
            lane_heading[edge_index[1]],
        )
        edge_attr_length, edge_attr_theta = compute_angles_lengths_2D(edge_vector)
        edge_attr_heading = wrap_angle(lane_heading[edge_index[0]] - lane_heading[edge_index[1]])
        edge_attr_direction = lane_position.new_tensor(edge_direction)
        edge_attr_hop = lane_position.new_tensor(edge_hop)
        edge_attr_input = torch.stack(
            [edge_attr_length, edge_attr_theta, edge_attr_heading, edge_attr_direction, edge_attr_hop],
            dim=-1,
        )
        return edge_index, self.l2l_traffic_emb_layer(input=edge_attr_input)

    def _build_lane_to_agent_topology_edges(self,
                                            lane_position: torch.Tensor,
                                            lane_heading: torch.Tensor,
                                            lane_batch: torch.Tensor,
                                            lane_visible_mask: torch.Tensor,
                                            agent_position: torch.Tensor,
                                            agent_heading: torch.Tensor,
                                            agent_batch: torch.Tensor,
                                            agent_visible_mask: torch.Tensor,
                                            agent_speed: torch.Tensor,
                                            agent_lateral_velocity: torch.Tensor,
                                            successor_edge_index: torch.Tensor,
                                            left_neighbor_edge_index: torch.Tensor,
                                            right_neighbor_edge_index: torch.Tensor,
                                            conflict_edge_index: torch.Tensor,
                                            potential_conflict_mask: torch.Tensor,
                                            lane_turn_direction: torch.Tensor) -> tuple:
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=lane_position.device)
        empty_edge_weight = lane_position.new_zeros(0)
        if potential_conflict_mask.numel() == lane_position.size(0):
            conflict_lane_mask = potential_conflict_mask.to(device=lane_position.device, dtype=torch.bool).clone()
        else:
            conflict_lane_mask = torch.zeros(lane_position.size(0), dtype=torch.bool, device=lane_position.device)
        if lane_position.numel() == 0 or agent_position.numel() == 0:
            return empty_edge_index, empty_edge_weight, conflict_lane_mask.float()

        valid_mask = lane_visible_mask.unsqueeze(1) & agent_visible_mask.unsqueeze(0)
        valid_mask = valid_mask & (lane_batch.unsqueeze(1) == agent_batch.unsqueeze(0))
        heading_valid_mask = wrap_angle(lane_heading.unsqueeze(1) - agent_heading.unsqueeze(0)).abs() <= torch.pi / 3
        turn_heading_valid_mask = wrap_angle(lane_heading.unsqueeze(1) - agent_heading.unsqueeze(0)).abs() <= 2 * torch.pi / 3
        valid_mask = valid_mask & heading_valid_mask
        lane_agent_local = transform_point_to_local_coordinate(
            agent_position.unsqueeze(0),
            lane_position.unsqueeze(1),
            lane_heading.unsqueeze(1),
        )
        lateral_distance = lane_agent_local[..., 1].abs()
        total_distance = torch.linalg.vector_norm(lane_agent_local, dim=-1)
        current_mask = (
            valid_mask
            & (lateral_distance < self.traffic_lane_lateral_radius)
            & (total_distance < self.traffic_current_radius)
        )
        current_edge_index = dense_to_sparse(current_mask)[0]
        if current_edge_index.numel() == 0:
            return empty_edge_index, empty_edge_weight, conflict_lane_mask.float()

        successor_adjacency = self._build_weighted_lane_adjacency(successor_edge_index, lane_position)
        left_adjacency = self._build_lane_adjacency(left_neighbor_edge_index, lane_position.size(0))
        right_adjacency = self._build_lane_adjacency(right_neighbor_edge_index, lane_position.size(0))
        conflict_adjacency = self._build_lane_adjacency(conflict_edge_index, lane_position.size(0))
        reachable_distance = self.traffic_topology_base_distance + agent_speed * self.prediction_window
        reachable_distance_cpu = reachable_distance.detach().cpu().tolist()
        lane_batch_cpu = lane_batch.detach().cpu().tolist()
        agent_batch_cpu = agent_batch.detach().cpu().tolist()
        lane_visible_cpu = lane_visible_mask.detach().cpu().tolist()
        agent_visible_cpu = agent_visible_mask.detach().cpu().tolist()
        heading_valid_cpu = heading_valid_mask.detach().cpu()
        turn_heading_valid_cpu = turn_heading_valid_mask.detach().cpu()
        lateral_distance_cpu = lateral_distance.detach().cpu()
        agent_lateral_velocity_cpu = agent_lateral_velocity.detach().cpu().tolist()
        lane_turn_direction_cpu = lane_turn_direction.detach().cpu().tolist()
        potential_conflict_indices = torch.nonzero(conflict_lane_mask.detach().cpu(), as_tuple=False).flatten().tolist()

        current_lanes_by_agent = {}
        for lane_idx, agent_idx in current_edge_index.detach().cpu().t().tolist():
            current_lanes_by_agent.setdefault(agent_idx, []).append(lane_idx)

        edge_pairs = []
        edge_weight_by_key = {}
        selected_edges = set()

        def add_candidate_edge(lane_idx: int, agent_idx: int, topology_weight: float) -> None:
            edge_key = (lane_idx, agent_idx)
            if edge_key not in selected_edges:
                selected_edges.add(edge_key)
                edge_pairs.append([lane_idx, agent_idx])
                edge_weight_by_key[edge_key] = topology_weight
            else:
                edge_weight_by_key[edge_key] = max(edge_weight_by_key[edge_key], topology_weight)

        for agent_idx, seed_lanes in current_lanes_by_agent.items():
            if not agent_visible_cpu[agent_idx]:
                continue

            distance_limit = max(float(reachable_distance_cpu[agent_idx]), 0.0)
            conflict_nodes = set()
            for conflict_lane in potential_conflict_indices:
                if lane_visible_cpu[conflict_lane] and lane_batch_cpu[conflict_lane] == agent_batch_cpu[agent_idx]:
                    conflict_nodes.add(conflict_lane)
            for lane_idx in seed_lanes:
                for conflict_lane in conflict_adjacency.get(lane_idx, []):
                    if lane_visible_cpu[conflict_lane] and lane_batch_cpu[conflict_lane] == agent_batch_cpu[agent_idx]:
                        conflict_nodes.add(conflict_lane)
                        conflict_lane_mask[conflict_lane] = True

            best_distance = {}
            frontier = []

            for lane_idx in seed_lanes:
                if lane_idx in conflict_nodes:
                    continue
                if not lane_visible_cpu[lane_idx] or lane_batch_cpu[lane_idx] != agent_batch_cpu[agent_idx]:
                    continue
                if 0.0 < best_distance.get(lane_idx, float('inf')):
                    best_distance[lane_idx] = 0.0
                    heapq.heappush(frontier, (0.0, lane_idx))

            while frontier:
                path_distance, lane_idx = heapq.heappop(frontier)
                if path_distance > best_distance.get(lane_idx, float('inf')):
                    continue
                if lane_idx in conflict_nodes:
                    continue

                for conflict_lane in conflict_adjacency.get(lane_idx, []):
                    if lane_visible_cpu[conflict_lane] and lane_batch_cpu[conflict_lane] == agent_batch_cpu[agent_idx]:
                        conflict_nodes.add(conflict_lane)
                        conflict_lane_mask[conflict_lane] = True

                add_candidate_edge(lane_idx, agent_idx, 1.0)

                for adjacent_lane in left_adjacency.get(lane_idx, []):
                    if adjacent_lane in conflict_nodes:
                        continue
                    if not lane_visible_cpu[adjacent_lane] or lane_batch_cpu[adjacent_lane] != agent_batch_cpu[agent_idx]:
                        continue
                    if not bool(heading_valid_cpu[adjacent_lane, agent_idx]):
                        continue
                    if lateral_distance_cpu[adjacent_lane, agent_idx] >= self.traffic_lane_change_lateral_radius:
                        continue
                    if agent_lateral_velocity_cpu[agent_idx] <= self.traffic_lane_change_lateral_speed:
                        continue
                    add_candidate_edge(adjacent_lane, agent_idx, 0.5)

                for adjacent_lane in right_adjacency.get(lane_idx, []):
                    if adjacent_lane in conflict_nodes:
                        continue
                    if not lane_visible_cpu[adjacent_lane] or lane_batch_cpu[adjacent_lane] != agent_batch_cpu[agent_idx]:
                        continue
                    if not bool(heading_valid_cpu[adjacent_lane, agent_idx]):
                        continue
                    if lateral_distance_cpu[adjacent_lane, agent_idx] >= self.traffic_lane_change_lateral_radius:
                        continue
                    if agent_lateral_velocity_cpu[agent_idx] >= -self.traffic_lane_change_lateral_speed:
                        continue
                    add_candidate_edge(adjacent_lane, agent_idx, 0.5)

                for next_lane, step_distance in successor_adjacency.get(lane_idx, []):
                    if next_lane in conflict_nodes:
                        continue
                    if not lane_visible_cpu[next_lane] or lane_batch_cpu[next_lane] != agent_batch_cpu[agent_idx]:
                        continue
                    at_fork = len(successor_adjacency.get(lane_idx, [])) > 1
                    is_turn_lane = lane_turn_direction_cpu[next_lane] in (1, 2)
                    is_topology_turn_lane = at_fork and is_turn_lane and bool(turn_heading_valid_cpu[next_lane, agent_idx])
                    if not bool(heading_valid_cpu[next_lane, agent_idx]) and not is_topology_turn_lane:
                        continue
                    next_distance = path_distance + step_distance
                    if next_distance > distance_limit:
                        continue
                    if next_distance < best_distance.get(next_lane, float('inf')):
                        best_distance[next_lane] = next_distance
                        heapq.heappush(frontier, (next_distance, next_lane))

        if len(edge_pairs) == 0:
            return empty_edge_index, empty_edge_weight, conflict_lane_mask.float()

        edge_weight = lane_position.new_tensor([edge_weight_by_key[tuple(edge_pair)] for edge_pair in edge_pairs])
        edge_index = torch.tensor(edge_pairs, dtype=torch.long, device=lane_position.device).t().contiguous()
        return edge_index, edge_weight, conflict_lane_mask.float()

    def _assign_nodes_to_lanes(self,
                               position: torch.Tensor,
                               batch: torch.Tensor,
                               lane_position: torch.Tensor,
                               lane_batch: torch.Tensor) -> torch.Tensor:
        assigned_lane = torch.empty(position.size(0), dtype=torch.long, device=position.device)
        if lane_position.numel() == 0:
            return assigned_lane.fill_(0)

        for start in range(0, position.size(0), 4096):
            end = min(start + 4096, position.size(0))
            distances = torch.norm(position[start:end].unsqueeze(1) - lane_position.unsqueeze(0), p=2, dim=-1)
            same_batch = batch[start:end].unsqueeze(1) == lane_batch.unsqueeze(0)
            masked_distances = distances.masked_fill(~same_batch, float('inf'))
            min_distances, min_indices = masked_distances.min(dim=1)

            no_lane_in_batch = torch.isinf(min_distances)
            if no_lane_in_batch.any():
                fallback_indices = distances[no_lane_in_batch].min(dim=1).indices
                min_indices[no_lane_in_batch] = fallback_indices

            assigned_lane[start:end] = min_indices

        return assigned_lane

    def _compute_lane_average_speed(self,
                                    assigned_lane: torch.Tensor,
                                    speed: torch.Tensor,
                                    num_lanes: int) -> torch.Tensor:
        lane_speed_sum = speed.new_zeros(num_lanes)
        lane_count = speed.new_zeros(num_lanes)
        lane_speed_sum.scatter_add_(0, assigned_lane, speed)
        lane_count.scatter_add_(0, assigned_lane, torch.ones_like(speed))
        return lane_speed_sum / lane_count.clamp_min(1.0)

    def _mark_conflict_lanes(self,
                             lane_position: torch.Tensor,
                             lane_batch: torch.Tensor,
                             E_VV_msg: torch.Tensor,
                             interaction_position: torch.Tensor,
                             interaction_batch: torch.Tensor) -> torch.Tensor:
        lane_conflict = torch.zeros(lane_position.size(0), dtype=torch.bool, device=lane_position.device)
        if E_VV_msg.numel() == 0 or lane_position.numel() == 0:
            return lane_conflict.float()

        src, dst = E_VV_msg
        conflict_points = (interaction_position[src] + interaction_position[dst]) * 0.5
        conflict_batch = interaction_batch[src]

        for start in range(0, conflict_points.size(0), 4096):
            end = min(start + 4096, conflict_points.size(0))
            points = conflict_points[start:end]
            point_batch = conflict_batch[start:end]
            same_batch = lane_batch.unsqueeze(1) == point_batch.unsqueeze(0)
            distances = torch.norm(lane_position.unsqueeze(1) - points.unsqueeze(0), p=2, dim=-1)
            lane_conflict = lane_conflict | ((distances < self.conflict_lane_radius) & same_batch).any(dim=1)

        return lane_conflict.float()

    def forward(self,
                data: Batch,
                l_embs: torch.Tensor,
                return_module_contexts: bool = False):
        # initialization
        historical_position = data['agent']['position'][:, :self.num_historical_steps]
        historical_visible_mask = data['agent']['visible_mask'][:, :self.num_historical_steps]
        causal_kinematics = compute_historical_kinematics(
            position=historical_position,
            visible_mask=historical_visible_mask,
            step_time=self.observation_step_time,
        )
        (
            a_velocity,
            a_velocity_length,
            a_acceleration,
            a_jerk,
            a_heading,
            a_heading_change,
            a_velocity_theta,
        ) = self._select_main_motion(
            data=data,
            causal_kinematics=causal_kinematics,
        )
        a_length = data['agent']['length'].unsqueeze(-1).repeat_interleave(self.num_historical_steps,-1)      #[(N1,...,Nb),H]
        a_width = data['agent']['width'].unsqueeze(-1).repeat_interleave(self.num_historical_steps,-1)      #[(N1,...,Nb),H]
        a_type = data['agent']['type'].to(historical_position.dtype).unsqueeze(-1).repeat_interleave(self.num_historical_steps,-1)      #[(N1,...,Nb),H]
        a_input = torch.cat([
            a_velocity,
            a_velocity_length.unsqueeze(-1),
            a_acceleration,
            a_jerk,
            torch.stack([torch.sin(a_heading), torch.cos(a_heading)], dim=-1),
            a_heading_change.unsqueeze(-1),
            a_length.unsqueeze(-1),
            a_width.unsqueeze(-1),
            a_type.unsqueeze(-1),
        ], dim=-1)
        a_embs = self.a_emb_layer(input=a_input)    #[(N1,...,Nb),H,D]

        current_step = self.num_historical_steps - 1
        proposal_module_contexts = {}
        refinement_module_contexts = {}
        l2m_position_l = data['lane']['position']                       #[(M1,...,Mb),2]
        l2m_heading_l = data['lane']['heading']                         #[(M1,...,Mb)]
        l2m_batch_l = data['lane']['batch']                             #[(M1,...,Mb)]
        left_neighbor_edge_index = self._get_lane_edge_index(data, 'left_neighbor_edge_index', l2m_position_l.device)
        right_neighbor_edge_index = self._get_lane_edge_index(data, 'right_neighbor_edge_index', l2m_position_l.device)
        predecessor_edge_index = self._get_lane_edge_index(data, 'predecessor_edge_index', l2m_position_l.device)
        successor_edge_index = self._get_lane_edge_index(data, 'successor_edge_index', l2m_position_l.device)
        conflict_edge_index = self._get_lane_edge_index(data, 'conflict_edge_index', l2m_position_l.device)
        adjacent_left_edge_index = self._get_lane_edge_index(data, 'adjacent_left_edge_index', l2m_position_l.device)
        adjacent_right_edge_index = self._get_lane_edge_index(data, 'adjacent_right_edge_index', l2m_position_l.device)
        lane_change_pair_index = self._get_lane_edge_index(data, 'lane_change_pair_index', l2m_position_l.device)
        if conflict_edge_index.numel() != 0:
            lane_visible_mask = data['lane']['visible_mask']
            conflict_valid_mask = lane_visible_mask[conflict_edge_index[0]] & lane_visible_mask[conflict_edge_index[1]]
            conflict_valid_mask = conflict_valid_mask & (conflict_edge_index[0] != conflict_edge_index[1])
            conflict_edge_index = conflict_edge_index[:, conflict_valid_mask]
        if 'width' not in data['lane']:
            raise ValueError(
                'lane.width is required for vehicle style features; regenerate INTERACTION processed data'
            )
        required_lane_fields = (
            'drivable',
            'start_position',
            'end_position',
            'centerline_spacing',
            'potential_conflict_mask',
            'crossing_conflict_mask',
            'merge_region_mask',
            'split_region_mask',
            'lane_change_critical_mask',
        )
        missing_lane_fields = [
            field_name for field_name in required_lane_fields
            if field_name not in data['lane']
        ]
        if missing_lane_fields:
            raise ValueError(
                'processed lane data is missing conflict-aware topology fields '
                f'{missing_lane_fields}; regenerate INTERACTION processed data'
            )
        p2l_edge_index = data['polyline', 'lane']['polyline_to_lane_edge_index']
        lane_association = load_lane_association_from_data(
            data=data,
            num_historical_steps=self.num_historical_steps,
        )
        if lane_association is None:
            lane_association = self.lane_reachability_builder.associate(
                agent_position=historical_position,
                agent_heading=causal_kinematics.heading,
                agent_speed=causal_kinematics.speed,
                agent_heading_valid_mask=causal_kinematics.heading_valid_mask,
                agent_visible_mask=historical_visible_mask,
                agent_batch=data['agent']['batch'],
                lane_heading=l2m_heading_l,
                lane_width=data['lane']['width'],
                lane_drivable=data['lane']['drivable'],
                lane_visible_mask=data['lane']['visible_mask'],
                lane_batch=l2m_batch_l,
                lane_centerline_spacing=data['lane']['centerline_spacing'],
                predecessor_edge_index=predecessor_edge_index,
                successor_edge_index=successor_edge_index,
                left_change_edge_index=left_neighbor_edge_index,
                right_change_edge_index=right_neighbor_edge_index,
                adjacent_left_edge_index=adjacent_left_edge_index,
                adjacent_right_edge_index=adjacent_right_edge_index,
                polyline_position=data['polyline']['position'],
                polyline_heading=data['polyline']['heading'],
                polyline_length=data['polyline']['length'],
                polyline_side=data['polyline']['side'],
                polyline_to_lane_edge_index=p2l_edge_index,
            )
        lane_reachability = self.lane_reachability_builder.build_reachability(
            association=lane_association,
            agent_velocity=causal_kinematics.velocity,
            agent_acceleration=causal_kinematics.acceleration,
            agent_visible_mask=historical_visible_mask,
            lane_heading=l2m_heading_l,
            lane_length=data['lane']['length'],
            lane_start_position=data['lane']['start_position'],
            lane_end_position=data['lane']['end_position'],
            lane_drivable=data['lane']['drivable'],
            lane_visible_mask=data['lane']['visible_mask'],
            successor_edge_index=successor_edge_index,
            lane_change_pair_index=lane_change_pair_index,
        )
        conflict_instances = self.lane_reachability_builder.build_conflict_instances(
            reachability=lane_reachability,
            potential_conflict_mask=data['lane']['potential_conflict_mask'],
            agent_batch=data['agent']['batch'],
            num_historical_steps=self.num_historical_steps,
            agent_valid_mask=data['agent']['type'] == 0,
        )
        style_reference_lane = lane_association.main_lane
        style_lane_lateral_distance = lane_association.lateral_distance
        data['agent']['main_lane'] = lane_association.main_lane
        data['agent']['main_lane_projection_s'] = lane_association.projection_s
        data['agent']['main_lane_lateral_distance'] = lane_association.lateral_distance
        data['agent']['main_lane_heading_difference'] = lane_association.heading_difference
        data['agent']['auxiliary_lane_seed_edge_index'] = lane_association.auxiliary_seed_edge_index
        data['agent']['reachable_lane_edge_index'] = lane_reachability.edge_index
        data['agent']['reachable_lane_path_distance'] = lane_reachability.path_distance
        data['agent']['reachable_lane_change_count'] = lane_reachability.lane_change_count
        data['agent']['reachable_distance_budget'] = lane_reachability.distance_budget
        data['agent']['longitudinal_speed'] = lane_reachability.longitudinal_speed
        data['agent']['conflict_instance_pair_index'] = conflict_instances.pair_index
        data['agent']['conflict_instance_lane'] = conflict_instances.conflict_lane
        data['agent']['conflict_instance_valid_mask'] = conflict_instances.valid_mask
        data['agent']['conflict_instance_tau_raw_i'] = conflict_instances.tau_raw_i
        data['agent']['conflict_instance_tau_raw_j'] = conflict_instances.tau_raw_j
        data['agent']['conflict_instance_tau_clip_i'] = conflict_instances.tau_clip_i
        data['agent']['conflict_instance_tau_clip_j'] = conflict_instances.tau_clip_j
        data['agent']['conflict_instance_urgency'] = conflict_instances.urgency_cost
        data['agent']['E_VV_conf'] = conflict_instances.E_VV_conf
        data['agent']['E_VV_conf_attr'] = conflict_instances.directed_edge_attr
        static_l_embs = l_embs
        num_all_agent = a_length.size(0)                # N1+...+Nb
        historical_vehicle_message_graph = self.vehicle_interaction_graph(
            position=historical_position,
            velocity=causal_kinematics.velocity,
            visible_mask=historical_visible_mask,
            batch=data['agent']['batch'],
            lane_position=l2m_position_l,
            lane_batch=l2m_batch_l,
            conflict_edge_index=conflict_edge_index,
            lane_heading=l2m_heading_l,
            reference_lane=style_reference_lane,
            assigned_lane=lane_association.main_lane,
            left_neighbor_edge_index=left_neighbor_edge_index,
            right_neighbor_edge_index=right_neighbor_edge_index,
        )
        data['agent']['E_VV_msg'] = historical_vehicle_message_graph.E_VV_msg
        data['agent']['D_VV_msg'] = historical_vehicle_message_graph.D_VV_msg
        data['agent']['A_VV_msg'] = historical_vehicle_message_graph.A_VV_msg
        style_feature = self._build_vehicle_style_feature(
            interaction_graph=historical_vehicle_message_graph,
            speed=causal_kinematics.speed,
            acceleration=causal_kinematics.acceleration,
            jerk=causal_kinematics.jerk,
            heading=causal_kinematics.heading,
            heading_change=causal_kinematics.heading_change,
            visible_mask=historical_visible_mask,
            lane_lateral_distance=style_lane_lateral_distance,
        )
        style_temporal_feature = self._encode_vehicle_style_sequence(
            style_feature=style_feature,
            visible_mask=historical_visible_mask,
        )
        style_prefix_embedding = self._pool_vehicle_style_prefix_embeddings(
            temporal_feature=style_temporal_feature,
            visible_mask=historical_visible_mask,
        )
        style_embedding = style_prefix_embedding[:, current_step]
        style_attention_weight = None
        if self.return_style_attention_weights:
            _, style_attention_weight = self._pool_vehicle_style_embedding(
                temporal_feature=style_temporal_feature,
                visible_mask=historical_visible_mask,
                return_attention_weights=True,
            )
        style_prefix_logits = self.style_classifier(style_prefix_embedding)
        style_prefix_probability = self.style_probability_norm(
            style_prefix_logits,
        )
        style_logits = style_prefix_logits[:, current_step]
        style_probability = style_prefix_probability[:, current_step]
        (
            style_prefix_indicator,
            style_prefix_reliable_mask,
        ) = self._build_vehicle_style_prefix_indicators(
            style_feature=style_feature,
            visible_mask=historical_visible_mask,
            agent_type=data['agent']['type'],
            velocity_valid_mask=causal_kinematics.velocity_valid_mask,
            acceleration_valid_mask=causal_kinematics.acceleration_valid_mask,
            jerk_valid_mask=causal_kinematics.jerk_valid_mask,
            heading_valid_mask=causal_kinematics.heading_valid_mask,
        )
        (
            flat_style_standardized_indicator,
            flat_style_score,
            flat_style_pseudo_label,
        ) = self._build_vehicle_style_pseudo_label(
            indicators=style_prefix_indicator.reshape(
                -1, style_prefix_indicator.size(-1),
            ),
            reliable_mask=style_prefix_reliable_mask.reshape(-1),
        )
        style_prefix_standardized_indicator = (
            flat_style_standardized_indicator.reshape_as(style_prefix_indicator)
        )
        style_prefix_score = flat_style_score.reshape(
            style_prefix_reliable_mask.shape,
        )
        style_prefix_pseudo_label = flat_style_pseudo_label.reshape(
            style_prefix_reliable_mask.shape,
        )
        style_indicator = style_prefix_indicator[:, current_step]
        style_standardized_indicator = (
            style_prefix_standardized_indicator[:, current_step]
        )
        style_score = style_prefix_score[:, current_step]
        style_reliable_mask = style_prefix_reliable_mask[:, current_step]
        style_pseudo_label = style_prefix_pseudo_label[:, current_step]
        data['agent']['style_feature'] = style_feature
        data['agent']['style_visible_mask'] = historical_visible_mask
        data['agent']['style_temporal_feature'] = style_temporal_feature
        data['agent']['style_temporal_visible_mask'] = historical_visible_mask
        data['agent']['style_embedding'] = style_embedding
        data['agent']['style_prefix_embedding'] = style_prefix_embedding
        data['agent']['style_prefix_logits'] = style_prefix_logits
        data['agent']['style_prefix_probability'] = style_prefix_probability
        data['agent']['style_prefix_score_indicator'] = style_prefix_indicator
        data['agent']['style_prefix_standardized_score_indicator'] = (
            style_prefix_standardized_indicator
        )
        data['agent']['style_prefix_score'] = style_prefix_score
        data['agent']['style_prefix_reliable_mask'] = style_prefix_reliable_mask
        data['agent']['style_prefix_pseudo_label'] = style_prefix_pseudo_label
        data['agent']['style_logits'] = style_logits
        data['agent']['style_probability'] = style_probability
        data['agent']['style_score_indicator'] = style_indicator
        data['agent']['style_standardized_score_indicator'] = style_standardized_indicator
        data['agent']['style_score'] = style_score
        data['agent']['style_reliable_mask'] = style_reliable_mask
        data['agent']['style_pseudo_label'] = style_pseudo_label
        if style_attention_weight is not None:
            data['agent']['style_attention_weight'] = style_attention_weight
        elif 'style_attention_weight' in data['agent']:
            del data['agent']['style_attention_weight']

        final_main_lane = lane_association.main_lane[:, current_step]
        valid_main_lane = final_main_lane >= 0
        safe_main_lane = final_main_lane.clamp(min=0)
        final_lane_direction = torch.stack([
            torch.cos(l2m_heading_l[safe_main_lane]),
            torch.sin(l2m_heading_l[safe_main_lane]),
        ], dim=-1)
        longitudinal_acceleration = (
            causal_kinematics.acceleration[:, current_step]
            * final_lane_direction
        ).sum(dim=-1).clamp_min(0.0)
        longitudinal_acceleration = torch.where(
            valid_main_lane,
            longitudinal_acceleration,
            torch.zeros_like(longitudinal_acceleration),
        )
        instance_lane = conflict_instances.conflict_lane
        conflict_type_feature = torch.stack([
            data['lane']['crossing_conflict_mask'][instance_lane],
            data['lane']['merge_region_mask'][instance_lane],
            data['lane']['split_region_mask'][instance_lane],
            data['lane']['lane_change_critical_mask'][instance_lane],
        ], dim=-1).to(historical_position.dtype)
        # F_rule^{ijc} = [r_i_priority, r_j_priority, r_i_yield,
        #                 r_j_yield, r_i_controlled, r_j_controlled].
        # The interface remains zero until these map attributes are reliable;
        # Lanelet2 canPass is a passability constraint, not a priority signal.
        rule_feature = historical_position.new_zeros(
            (instance_lane.numel(), 6),
        )
        priority_output = self.priority_module(
            conflict_instances=conflict_instances,
            style_embedding=style_embedding,
            longitudinal_speed=lane_reachability.longitudinal_speed[:, current_step],
            longitudinal_acceleration=longitudinal_acceleration,
            conflict_type_feature=conflict_type_feature,
            rule_feature=rule_feature,
            historical_position=historical_position,
            future_position=data['agent']['position'][
                :, self.num_historical_steps:
                self.num_historical_steps + self.num_future_steps
            ],
            historical_visible_mask=historical_visible_mask,
            future_visible_mask=data['agent']['visible_mask'][
                :, self.num_historical_steps:
                self.num_historical_steps + self.num_future_steps
            ],
            polyline_position=data['polyline']['position'],
            polyline_heading=data['polyline']['heading'],
            polyline_length=data['polyline']['length'],
            polyline_side=data['polyline']['side'],
            polyline_to_lane_edge_index=p2l_edge_index,
            step_time=self.observation_step_time,
        )
        data['agent']['priority_instance_logits'] = priority_output.instance_logits
        data['agent']['priority_instance_probability'] = priority_output.instance_probability
        data['agent']['priority_rule_feature'] = rule_feature
        data['agent']['priority_pair_probability'] = priority_output.pair_probability
        data['agent']['priority_directed_pair_probability'] = priority_output.directed_pair_probability
        data['agent']['priority_pseudo_label'] = priority_output.pseudo_label
        data['agent']['priority_pseudo_label_valid_mask'] = priority_output.pseudo_label_valid_mask
        data['agent']['priority_order_target'] = priority_output.order_target
        data['agent']['priority_future_entry_time_i'] = priority_output.future_entry_time_i
        data['agent']['priority_future_entry_time_j'] = priority_output.future_entry_time_j

        (
            trajectory_priority_probability,
            trajectory_priority_head_bias,
        ) = self._priority_trajectory_inputs(
            priority_output.directed_pair_probability,
            priority_output.directed_head_bias,
        )

        directed_conflict_edge_input = torch.cat([
            conflict_instances.directed_edge_attr,
            trajectory_priority_probability,
        ], dim=-1)
        directed_conflict_edge_embs = self.conflict_edge_emb_layer(
            directed_conflict_edge_input,
        )
        if conflict_instances.E_VV_conf.numel() == 0:
            vehicle_priority_context = torch.zeros_like(
                a_embs[:, current_step],
            )
        else:
            priority_updated_agent = self.vehicle_priority_attn_layer(
                x=a_embs[:, current_step],
                edge_index=conflict_instances.E_VV_conf,
                edge_attr=directed_conflict_edge_embs,
                attention_bias=trajectory_priority_head_bias,
            )
            involved_agent = torch.zeros(
                num_all_agent,
                dtype=torch.bool,
                device=historical_position.device,
            )
            involved_agent[conflict_instances.E_VV_conf.reshape(-1)] = True
            vehicle_priority_context = torch.where(
                involved_agent.unsqueeze(-1),
                priority_updated_agent - a_embs[:, current_step],
                torch.zeros_like(priority_updated_agent),
            )
        data['agent']['vehicle_priority_representation'] = vehicle_priority_context

        vehicle_to_lane_source = self.vehicle_to_lane_source_mlp(torch.cat([
            a_embs[:, current_step],
            style_embedding,
            vehicle_priority_context,
        ], dim=-1))
        a2l_edge_index, a2l_edge_attr_embs = self._build_main_agent_to_lane_edges(
            main_lane=final_main_lane,
            agent_position=data['agent']['position'][:, current_step],
            agent_heading=causal_kinematics.heading[:, current_step],
            agent_speed=causal_kinematics.speed[:, current_step],
            agent_visible_mask=data['agent']['visible_mask'][:, current_step],
            lane_position=l2m_position_l,
            lane_heading=l2m_heading_l,
            lane_visible_mask=data['lane']['visible_mask'],
        )
        dynamic_l_embs = self.a2l_attn_layer(
            x=[vehicle_to_lane_source, static_l_embs],
            edge_index=a2l_edge_index,
            edge_attr=a2l_edge_attr_embs,
        )
        lane_speed_valid = (
            historical_visible_mask[:, current_step]
            & valid_main_lane
        )
        lane_avg_speed = self._compute_lane_average_speed(
            assigned_lane=final_main_lane[lane_speed_valid],
            speed=causal_kinematics.speed[lane_speed_valid, current_step],
            num_lanes=l2m_position_l.size(0),
        )
        l2l_traffic_edge_index, l2l_traffic_edge_attr_embs = self._build_lane_traffic_edges(
            lane_position=l2m_position_l,
            lane_heading=l2m_heading_l,
            lane_batch=l2m_batch_l,
            lane_visible_mask=data['lane']['visible_mask'],
            lane_avg_speed=lane_avg_speed,
            successor_edge_index=successor_edge_index,
            predecessor_edge_index=predecessor_edge_index,
            conflict_edge_index=conflict_edge_index,
        )
        dynamic_l_embs = self.l2l_traffic_attn_layer(
            x=dynamic_l_embs,
            edge_index=l2l_traffic_edge_index,
            edge_attr=l2l_traffic_edge_attr_embs,
        )

        (
            E_VV_conf_mode,
            E_VV_conf_mode_attr,
        ) = self.lane_reachability_builder.expand_conflict_edges_across_modes(
            conflict_instances=conflict_instances,
            num_agents=num_all_agent,
            num_modes=self.num_modes,
            historical_step=current_step,
        )
        priority_probability_mode = (
            trajectory_priority_probability.repeat_interleave(
                self.num_modes, dim=0,
            )
        )
        priority_head_bias_mode = (
            trajectory_priority_head_bias.repeat_interleave(
                self.num_modes, dim=0,
            )
        )
        E_VV_conf_mode_embs = self.conflict_edge_emb_layer(torch.cat([
            E_VV_conf_mode_attr,
            priority_probability_mode,
        ], dim=-1))
        proposal_E_VV_conf_mode_embs = (
            self._condition_proposal_conflict_edges(
                E_VV_conf_mode_embs,
            )
        )

        if self.preserve_base_edges:
            base_vehicle_edge_index, base_vehicle_edge_attr = (
                self._build_base_vehicle_edges(
                    position=historical_position,
                    visible_mask=historical_visible_mask,
                    batch=data['agent']['batch'],
                    sparse_edge_index=historical_vehicle_message_graph.E_VV_msg,
                    sparse_edge_attr=historical_vehicle_message_graph.edge_attr,
                )
            )
            data['agent']['E_VV_base'] = base_vehicle_edge_index
            E_VV_msg, E_VV_msg_attr = (
                self._expand_vehicle_edges_across_modes(
                    edge_index=base_vehicle_edge_index,
                    edge_attr=base_vehicle_edge_attr,
                    num_agents=num_all_agent,
                    num_modes=self.num_modes,
                )
            )
        else:
            E_VV_msg, E_VV_msg_attr = (
                self.vehicle_interaction_graph.expand_across_modes(
                    graph=historical_vehicle_message_graph,
                    num_agents=num_all_agent,
                    num_modes=self.num_modes,
                )
            )
        m_embs = self.mode_tokens.weight.unsqueeze(0).repeat_interleave(self.num_historical_steps,0)            #[H,K,D]
        m_embs = m_embs.unsqueeze(1).repeat_interleave(num_all_agent,1).reshape(-1, self.hidden_dim)            #[H*(N1,...,Nb)*K,D]

        m_batch = data['agent']['batch'].unsqueeze(1).repeat_interleave(self.num_modes,1)                       # [(N1,...,Nb),K]
        m_position = historical_position.unsqueeze(2).repeat_interleave(self.num_modes,2)  #[(N1,...,Nb),H,K,2]
        m_heading = a_heading.unsqueeze(2).repeat_interleave(self.num_modes,2)    #[(N1,...,Nb),H,K]
        m_valid_mask = historical_visible_mask.unsqueeze(2).repeat_interleave(self.num_modes,2)  #[(N1,...,Nb),H,K]

        #ALL EDGE
        #t2m edge
        t2m_position_t = historical_position.reshape(-1,2)      #[(N1,...,Nb)*H,2]
        t2m_position_m = m_position.reshape(-1,2)                                                   #[(N1,...,Nb)*H*K,2]
        t2m_heading_t = a_heading.reshape(-1)                                        #[(N1,...,Nb)*H]
        t2m_heading_m = m_heading.reshape(-1)                                                       #[(N1,...,Nb)*H*K]
        t2m_valid_mask_t = historical_visible_mask              #[(N1,...,Nb),H]
        t2m_valid_mask_m = m_valid_mask.reshape(num_all_agent,-1)                                   #[(N1,...,Nb),H*K]
        t2m_valid_mask = t2m_valid_mask_t.unsqueeze(2) & t2m_valid_mask_m.unsqueeze(1)              #[(N1,...,Nb),H,H*K]
        t2m_edge_index = dense_to_sparse(t2m_valid_mask)[0]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1]/self.num_modes) >= t2m_edge_index[0]]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1]/self.num_modes) - t2m_edge_index[0] <= self.duration]
        t2m_edge_vector = transform_point_to_local_coordinate(t2m_position_t[t2m_edge_index[0]], t2m_position_m[t2m_edge_index[1]], t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_length, t2m_edge_attr_theta = compute_angles_lengths_2D(t2m_edge_vector)
        t2m_edge_attr_heading = wrap_angle(t2m_heading_t[t2m_edge_index[0]] - t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_interval = t2m_edge_index[0] - torch.floor(t2m_edge_index[1]/self.num_modes)
        t2m_edge_attr_input = torch.stack([t2m_edge_attr_length, t2m_edge_attr_theta, t2m_edge_attr_heading, t2m_edge_attr_interval], dim=-1)
        t2m_edge_attr_embs = self.t2m_emb_layer(input=t2m_edge_attr_input)

        #m2m_a_edge
        m2m_a_position = m_position.permute(1,2,0,3).reshape(-1, 2)    #[H*K*(N1,...,Nb),2]
        m2m_a_heading = m_heading.permute(1,2,0).reshape(-1)           #[H*K*(N1,...,Nb)]
        m2m_a_vel_length = a_velocity_length.transpose(0,1).unsqueeze(1).repeat_interleave(self.num_modes,1).reshape(-1)  #[H*K*(N1,...,Nb)]
        m2m_a_vel_theta = a_velocity_theta.transpose(0,1).unsqueeze(1).repeat_interleave(self.num_modes,1).reshape(-1)    #[H*K*(N1,...,Nb)]
        m2m_a_batch = data['agent']['batch']                           #[(N1,...,Nb)]
        m2m_a_node_batch = m2m_a_batch.unsqueeze(0).unsqueeze(0).repeat(self.num_historical_steps, self.num_modes, 1).reshape(-1)
        m2m_a_edge_attr_embs = self._embed_vehicle_interaction_edges(
            edge_index=E_VV_msg,
            graph_edge_attr=E_VV_msg_attr,
            position=m2m_a_position,
            heading=m2m_a_heading,
            velocity_length=m2m_a_vel_length,
            velocity_theta=m2m_a_vel_theta,
        )

        #l2m edge
        l2m_position_m = m_position.reshape(-1,2)                       #[(N1,...,Nb)*H*K,2]
        l2m_heading_m = m_heading.reshape(-1)                           #[(N1,...,Nb)]
        (
            reachable_l2m_edge_index,
            _l2m_path_distance,
            reachable_l2m_lane_change_count,
        ) = self.lane_reachability_builder.expand_reachability_across_modes(
            reachability=lane_reachability,
            num_agents=num_all_agent,
            num_historical_steps=self.num_historical_steps,
            num_modes=self.num_modes,
        )
        reachable_l2m_weight = torch.where(
            reachable_l2m_lane_change_count > 0,
            l2m_position_l.new_full(
                reachable_l2m_lane_change_count.shape, 0.5,
            ),
            l2m_position_l.new_ones(
                reachable_l2m_lane_change_count.shape,
            ),
        )
        if self.preserve_base_edges:
            l2m_batch_m = m_batch.unsqueeze(1).repeat_interleave(
                self.num_historical_steps, 1,
            ).reshape(-1)
            l2m_edge_index = self._build_base_lane_edges(
                lane_position=l2m_position_l,
                lane_batch=l2m_batch_l,
                lane_visible_mask=data['lane']['visible_mask'],
                target_position=l2m_position_m,
                target_batch=l2m_batch_m,
                target_visible_mask=m_valid_mask.reshape(-1),
            )
            l2m_edge_topology_weight = self._match_reachability_weight(
                base_edge_index=l2m_edge_index,
                reachable_edge_index=reachable_l2m_edge_index,
                reachable_weight=reachable_l2m_weight,
                num_targets=l2m_position_m.size(0),
                dtype=l2m_position_l.dtype,
            )
            data['agent']['base_l2m_edge_index'] = l2m_edge_index
        else:
            l2m_edge_index = reachable_l2m_edge_index
            l2m_edge_topology_weight = reachable_l2m_weight
        l2m_edge_vector = transform_point_to_local_coordinate(l2m_position_l[l2m_edge_index[0]], l2m_position_m[l2m_edge_index[1]], l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_length, l2m_edge_attr_theta = compute_angles_lengths_2D(l2m_edge_vector)
        l2m_edge_attr_heading = wrap_angle(l2m_heading_l[l2m_edge_index[0]] - l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_conflict = data['lane']['potential_conflict_mask'][l2m_edge_index[0]].to(l2m_position_l.dtype)
        l2m_edge_attr_input = torch.stack([l2m_edge_attr_length, l2m_edge_attr_theta, l2m_edge_attr_heading, l2m_edge_attr_conflict, l2m_edge_topology_weight], dim=-1)
        l2m_edge_attr_embs = self.l2m_emb_layer(input=l2m_edge_attr_input)

        #mode edge
        #m2m_h                        
        m2m_h_position = m_position.permute(2,0,1,3).reshape(-1, 2)    #[K*(N1,...,Nb)*H,2]
        m2m_h_heading = m_heading.permute(2,0,1).reshape(-1)           #[K*(N1,...,Nb)*H]
        m2m_h_valid_mask = m_valid_mask.permute(2,0,1).reshape(-1, self.num_historical_steps)   #[K*(N1,...,Nb),H]
        m2m_h_valid_mask = m2m_h_valid_mask.unsqueeze(2) & m2m_h_valid_mask.unsqueeze(1)        #[K*(N1,...,Nb),H,H]     
        m2m_h_edge_index = dense_to_sparse(m2m_h_valid_mask)[0]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] > m2m_h_edge_index[0]]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] - m2m_h_edge_index[0] <= self.duration]
        m2m_h_edge_vector = transform_point_to_local_coordinate(m2m_h_position[m2m_h_edge_index[0]], m2m_h_position[m2m_h_edge_index[1]], m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_length, m2m_h_edge_attr_theta = compute_angles_lengths_2D(m2m_h_edge_vector)
        m2m_h_edge_attr_heading = wrap_angle(m2m_h_heading[m2m_h_edge_index[0]] - m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_interval = m2m_h_edge_index[0] - m2m_h_edge_index[1]
        m2m_h_edge_attr_input = torch.stack([m2m_h_edge_attr_length, m2m_h_edge_attr_theta, m2m_h_edge_attr_heading, m2m_h_edge_attr_interval], dim=-1)
        m2m_h_edge_attr_embs = self.m2m_h_emb_layer(input=m2m_h_edge_attr_input)

        #m2m_s edge
        m2m_s_valid_mask = m_valid_mask.transpose(0,1).reshape(-1, self.num_modes)              #[H*(N1,...,Nb),K]
        m2m_s_valid_mask = m2m_s_valid_mask.unsqueeze(2) & m2m_s_valid_mask.unsqueeze(1)        #[H*(N1,...,Nb),K,K]
        m2m_s_edge_index = dense_to_sparse(m2m_s_valid_mask)[0]
        m2m_s_edge_index = m2m_s_edge_index[:, m2m_s_edge_index[0] != m2m_s_edge_index[1]]
        m2m_s_src_mode = m2m_s_edge_index[0] % self.num_modes
        m2m_s_dst_mode = m2m_s_edge_index[1] % self.num_modes
        m2m_s_pair_id = m2m_s_src_mode * self.num_modes + m2m_s_dst_mode
        m2m_s_edge_attr_embs = self.mode_pair_emb(m2m_s_pair_id)

        #ALL ATTENTION
        #t2m attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)  #[(N1,...,Nb)*H,D]
        m_embs_t = self.t2m_attn_layer(x = [t_embs, m_embs], edge_index = t2m_edge_index, edge_attr = t2m_edge_attr_embs)         #[(N1,...,Nb)*H*K,D]

        #l2m attention
        final_mode_target = torch.zeros(
            (num_all_agent, self.num_historical_steps, self.num_modes),
            dtype=torch.bool,
            device=m_embs.device,
        )
        final_mode_target[:, current_step] = True
        if self.preserve_base_edges:
            m_embs_l = self.l2m_attn_layer(
                x=[static_l_embs, m_embs],
                edge_index=l2m_edge_index,
                edge_attr=l2m_edge_attr_embs,
            )
            if (
                (self.reachability_residual_scale > 0
                 or return_module_contexts)
                and reachable_l2m_edge_index.numel() != 0
            ):
                reachable_l2m_target_time = (
                    torch.div(
                        reachable_l2m_edge_index[1],
                        self.num_modes,
                        rounding_mode='floor',
                    ) % self.num_historical_steps
                )
                final_reachable_mask = (
                    reachable_l2m_target_time == current_step
                )
                final_reachable_edge = reachable_l2m_edge_index[
                    :, final_reachable_mask
                ]
                reachable_edge_vector = transform_point_to_local_coordinate(
                    l2m_position_l[final_reachable_edge[0]],
                    l2m_position_m[final_reachable_edge[1]],
                    l2m_heading_m[final_reachable_edge[1]],
                )
                (
                    reachable_attr_length,
                    reachable_attr_theta,
                ) = compute_angles_lengths_2D(reachable_edge_vector)
                reachable_attr_heading = wrap_angle(
                    l2m_heading_l[final_reachable_edge[0]]
                    - l2m_heading_m[final_reachable_edge[1]]
                )
                reachable_attr_conflict = (
                    data['lane']['potential_conflict_mask'][
                        final_reachable_edge[0]
                    ].to(l2m_position_l.dtype)
                )
                reachable_l2m_attr_input = torch.stack([
                    reachable_attr_length,
                    reachable_attr_theta,
                    reachable_attr_heading,
                    reachable_attr_conflict,
                    reachable_l2m_weight[final_reachable_mask],
                ], dim=-1)
                reachable_l2m_attr_embs = self.l2m_emb_layer(
                    input=reachable_l2m_attr_input,
                )
                reachable_output = self.l2m_attn_layer(
                    x=[dynamic_l_embs, m_embs],
                    edge_index=final_reachable_edge,
                    edge_attr=reachable_l2m_attr_embs,
                )
                involved_target = torch.zeros(
                    m_embs.size(0), dtype=torch.bool, device=m_embs.device,
                )
                involved_target[final_reachable_edge[1]] = True
                reachability_delta = torch.where(
                    involved_target.unsqueeze(-1),
                    reachable_output - m_embs,
                    torch.zeros_like(m_embs),
                )
                if return_module_contexts:
                    proposal_module_contexts['reachability'] = (
                        reachability_delta.reshape(
                            num_all_agent,
                            self.num_historical_steps,
                            self.num_modes,
                            self.hidden_dim,
                        ).mean(dim=2)
                    )
                m_embs_l = (
                    m_embs_l
                    + self.reachability_residual_scale * reachability_delta
                )
        else:
            l2m_target_time = (
                torch.div(
                    l2m_edge_index[1], self.num_modes,
                    rounding_mode='floor',
                ) % self.num_historical_steps
            )
            historical_l2m_mask = l2m_target_time < current_step
            final_l2m_mask = l2m_target_time == current_step
            m_embs_l_static = self.l2m_attn_layer(
                x=[static_l_embs, m_embs],
                edge_index=l2m_edge_index[:, historical_l2m_mask],
                edge_attr=l2m_edge_attr_embs[historical_l2m_mask],
            )
            m_embs_l_dynamic = self.l2m_attn_layer(
                x=[dynamic_l_embs, m_embs],
                edge_index=l2m_edge_index[:, final_l2m_mask],
                edge_attr=l2m_edge_attr_embs[final_l2m_mask],
            )
            m_embs_l = torch.where(
                final_mode_target.reshape(-1, 1),
                m_embs_l_dynamic,
                m_embs_l_static,
            )
            if return_module_contexts:
                reachability_delta = torch.where(
                    final_mode_target.reshape(-1, 1),
                    m_embs_l_dynamic - m_embs,
                    torch.zeros_like(m_embs),
                )
                proposal_module_contexts['reachability'] = (
                    reachability_delta.reshape(
                        num_all_agent,
                        self.num_historical_steps,
                        self.num_modes,
                        self.hidden_dim,
                    ).mean(dim=2)
                )
        
        m_embs = m_embs_t + m_embs_l
        m_embs = m_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1,self.hidden_dim)       #[H*(N1,...,Nb)*K,D]
        #moda attention  
        for i in range(self.num_attn_layers):
            #m2m_a
            m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(1,2).reshape(-1, self.hidden_dim)  #[H*K*(N1,...,Nb),D]
            m_embs = self.m2m_a_attn_layers[i](x = m_embs, edge_index = E_VV_msg, edge_attr = m2m_a_edge_attr_embs)
            #m2m_h
            m_embs = m_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(1,2,0,3).reshape(-1, self.hidden_dim)  #[K*(N1,...,Nb)*H,D]
            m_embs = self.m2m_h_attn_layers[i](x = m_embs, edge_index = m2m_h_edge_index, edge_attr = m2m_h_edge_attr_embs)
            #m2m_s
            m_embs = m_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim).transpose(0,2).reshape(-1, self.hidden_dim)  #[H*(N1,...,Nb)*K,D]
            m_embs = self.m2m_s_attn_layers[i](x = m_embs, edge_index = m2m_s_edge_index, edge_attr = m2m_s_edge_attr_embs)
        proposal_conflict_input = m_embs.reshape(
            self.num_historical_steps,
            num_all_agent,
            self.num_modes,
            self.hidden_dim,
        ).transpose(1, 2).reshape(-1, self.hidden_dim)  # [H,K,N,D]
        proposal_conflict_output = self._apply_conflict_attention(
            node_embedding=proposal_conflict_input,
            conflict_edge_index=E_VV_conf_mode,
            conflict_edge_embedding=proposal_E_VV_conf_mode_embs,
            priority_head_bias=priority_head_bias_mode,
            attention_layer=self.proposal_conflict_attn_layer,
            num_agents=num_all_agent,
        )
        proposal_conflict_output = self._gate_proposal_conflict_residual(
            conflict_input=proposal_conflict_input,
            conflict_output=proposal_conflict_output,
            conflict_edge_index=E_VV_conf_mode,
            num_agents=num_all_agent,
        )
        if return_module_contexts:
            proposal_module_contexts['priority'] = (
                proposal_conflict_output - proposal_conflict_input
            ).reshape(
                self.num_historical_steps,
                self.num_modes,
                num_all_agent,
                self.hidden_dim,
            ).permute(2, 0, 1, 3).mean(dim=2)
        m_embs = proposal_conflict_output.reshape(
            self.num_historical_steps,
            self.num_modes,
            num_all_agent,
            self.hidden_dim,
        ).transpose(1, 2).transpose(0, 1).reshape(-1, self.hidden_dim)
        proposal_pre_style = m_embs
        m_embs = self._apply_style_film(
            node_embedding=m_embs,
            style_prefix_embedding=style_prefix_embedding,
            film_generator=self.proposal_style_film,
        )
        if return_module_contexts:
            proposal_module_contexts['style'] = (
                m_embs - proposal_pre_style
            ).reshape(
                num_all_agent,
                self.num_historical_steps,
                self.num_modes,
                self.hidden_dim,
            ).mean(dim=2)
        m_embs = self._coordinate_scene_modes(
            node_embedding=m_embs,
            agent_batch=data['agent']['batch'],
            valid_mask=m_valid_mask,
            num_scenes=data.num_graphs,
        )

        #generate traj
        traj_propose = self.traj_propose(m_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 2)         #[(N1,...,Nb),H,K,F,2]
        traj_propose = transform_traj_to_global_coordinate(traj_propose, m_position, m_heading)        #[(N1,...,Nb),H,K,F,2]

        #generate anchor
        proposal = traj_propose.detach()        #[(N1,...,Nb),H,K,F,2]
        
        n_batch = m_batch                                                                                                   #[(N1,...,Nb),K]
        n_position = proposal[:,:,:, self.num_future_steps // 2,:]                                                     #[(N1,...,Nb),H,K,2]
        _, n_heading = compute_angles_lengths_2D(proposal[:,:,:, self.num_future_steps // 2,:] - proposal[:,:,:, self.num_future_steps // 2 - 1,:])  #[(N1,...,Nb),H,K]
        n_valid_mask = m_valid_mask                                                                                         #[(N1,...,Nb),H,K]
        
        proposal = transform_traj_to_local_coordinate(proposal, n_position, n_heading)                                      #[(N1,...,Nb),H,K,F,2]
        anchor = self.proposal_to_anchor(proposal.reshape(-1, self.num_future_steps*2))                                     #[(N1,...,Nb)*H*K,D]
        n_embs = anchor                                                                                                     #[(N1,...,Nb)*H*K,D]

        #t2n edge
        t2n_position_t = historical_position.reshape(-1,2)      #[(N1,...,Nb)*H,2]
        t2n_position_n = n_position.reshape(-1,2)                                                   #[(N1,...,Nb)*H*K,2]
        t2n_heading_t = a_heading.reshape(-1)                                        #[(N1,...,Nb)*H]
        t2n_heading_n = n_heading.reshape(-1)                                                       #[(N1,...,Nb)*H*K]
        t2n_valid_mask_t = historical_visible_mask              #[(N1,...,Nb),H]
        t2n_valid_mask_n = n_valid_mask.reshape(num_all_agent,-1)                                   #[(N1,...,Nb),H*K]
        t2n_valid_mask = t2n_valid_mask_t.unsqueeze(2) & t2n_valid_mask_n.unsqueeze(1)              #[(N1,...,Nb),H,H*K]
        t2n_edge_index = dense_to_sparse(t2n_valid_mask)[0]
        t2n_edge_index = t2n_edge_index[:, torch.floor(t2n_edge_index[1]/self.num_modes) >= t2n_edge_index[0]]
        t2n_edge_index = t2n_edge_index[:, torch.floor(t2n_edge_index[1]/self.num_modes) - t2n_edge_index[0] <= self.duration]
        t2n_edge_vector = transform_point_to_local_coordinate(t2n_position_t[t2n_edge_index[0]], t2n_position_n[t2n_edge_index[1]], t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_length, t2n_edge_attr_theta = compute_angles_lengths_2D(t2n_edge_vector)
        t2n_edge_attr_heading = wrap_angle(t2n_heading_t[t2n_edge_index[0]] - t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_interval = t2n_edge_index[0] - torch.floor(t2n_edge_index[1]/self.num_modes) - self.num_future_steps//2
        t2n_edge_attr_input = torch.stack([t2n_edge_attr_length, t2n_edge_attr_theta, t2n_edge_attr_heading, t2n_edge_attr_interval], dim=-1)
        t2n_edge_attr_embs = self.t2n_emb_layer(input=t2n_edge_attr_input)

        #l2n edge
        l2n_position_l = data['lane']['position']                       #[(M1,...,Mb),2]
        l2n_position_n = n_position.reshape(-1,2)                       #[(N1,...,Nb)*H*K,2]
        l2n_heading_l = data['lane']['heading']                         #[(M1,...,Mb)]
        l2n_heading_n = n_heading.reshape(-1)                           #[(N1,...,Nb)*H*K]
        l2n_batch_l = data['lane']['batch']                             #[(M1,...,Mb)]
        l2n_batch_n = n_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps,1).reshape(-1)       #[(N1,...,Nb)*H*K]
        l2n_valid_mask_l = data['lane']['visible_mask']                                                     #[(M1,...,Mb)]
        l2n_valid_mask_n = n_valid_mask.reshape(-1)                                                         #[(N1,...,Nb)*H*K]
        l2n_speed_n = causal_kinematics.speed.unsqueeze(2).repeat_interleave(
            self.num_modes, 2,
        ).reshape(-1)
        midpoint_edge_index, _, _ = self._build_lane_to_agent_topology_edges(
            lane_position=l2n_position_l,
            lane_heading=l2n_heading_l,
            lane_batch=l2n_batch_l,
            lane_visible_mask=l2n_valid_mask_l,
            agent_position=l2n_position_n,
            agent_heading=l2n_heading_n,
            agent_batch=l2n_batch_n,
            agent_visible_mask=l2n_valid_mask_n,
            agent_speed=l2n_speed_n,
            agent_lateral_velocity=torch.zeros_like(l2n_speed_n),
            successor_edge_index=successor_edge_index,
            left_neighbor_edge_index=left_neighbor_edge_index,
            right_neighbor_edge_index=right_neighbor_edge_index,
            conflict_edge_index=torch.empty(
                (2, 0), dtype=torch.long, device=l2n_position_l.device,
            ),
            potential_conflict_mask=torch.zeros_like(
                data['lane']['potential_conflict_mask'],
            ),
            lane_turn_direction=data['lane']['turn_direction'],
        )
        num_refinement_targets = l2n_position_n.size(0)
        reachable_key = (
            reachable_l2m_edge_index[0] * num_refinement_targets
            + reachable_l2m_edge_index[1]
        )
        midpoint_key = (
            midpoint_edge_index[0] * num_refinement_targets
            + midpoint_edge_index[1]
        )
        if reachable_key.numel() == 0 or midpoint_key.numel() == 0:
            reachable_l2n_edge_index = midpoint_edge_index[:, :0]
            reachable_l2n_weight = l2n_position_l.new_zeros(0)
        else:
            sorted_reachable_key, reachable_permutation = reachable_key.sort()
            reachable_position = torch.searchsorted(
                sorted_reachable_key, midpoint_key,
            )
            safe_position = reachable_position.clamp(
                max=sorted_reachable_key.numel() - 1,
            )
            intersection_mask = (
                (reachable_position < sorted_reachable_key.numel())
                & (sorted_reachable_key[safe_position] == midpoint_key)
            )
            reachable_l2n_edge_index = midpoint_edge_index[:, intersection_mask]
            matched_reachable_index = reachable_permutation[
                safe_position[intersection_mask]
            ]
            reachable_l2n_weight = reachable_l2m_weight[
                matched_reachable_index
            ]
        if self.preserve_base_edges:
            l2n_edge_index = self._build_base_lane_edges(
                lane_position=l2n_position_l,
                lane_batch=l2n_batch_l,
                lane_visible_mask=l2n_valid_mask_l,
                target_position=l2n_position_n,
                target_batch=l2n_batch_n,
                target_visible_mask=l2n_valid_mask_n,
            )
            l2n_edge_topology_weight = self._match_reachability_weight(
                base_edge_index=l2n_edge_index,
                reachable_edge_index=reachable_l2n_edge_index,
                reachable_weight=reachable_l2n_weight,
                num_targets=num_refinement_targets,
                dtype=l2n_position_l.dtype,
            )
            data['agent']['base_l2n_edge_index'] = l2n_edge_index
        else:
            l2n_edge_index = reachable_l2n_edge_index
            l2n_edge_topology_weight = reachable_l2n_weight
        l2n_edge_vector = transform_point_to_local_coordinate(l2n_position_l[l2n_edge_index[0]], l2n_position_n[l2n_edge_index[1]], l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_length, l2n_edge_attr_theta = compute_angles_lengths_2D(l2n_edge_vector)
        l2n_edge_attr_heading = wrap_angle(l2n_heading_l[l2n_edge_index[0]] - l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_conflict = data['lane']['potential_conflict_mask'][l2n_edge_index[0]].to(l2n_position_l.dtype)
        l2n_edge_attr_input = torch.stack([l2n_edge_attr_length, l2n_edge_attr_theta, l2n_edge_attr_heading, l2n_edge_attr_conflict, l2n_edge_topology_weight], dim=-1)
        l2n_edge_attr_embs = self.l2n_emb_layer(input=l2n_edge_attr_input)

        #mode edge
        #n2n_a_edge
        n2n_a_position = n_position.permute(1,2,0,3).reshape(-1, 2)    #[H*K*(N1,...,Nb),2]
        n2n_a_heading = n_heading.permute(1,2,0).reshape(-1)           #[H*K*(N1,...,Nb)]
        n2n_a_vel_length = a_velocity_length.transpose(0,1).unsqueeze(1).repeat_interleave(self.num_modes,1).reshape(-1)  #[H*K*(N1,...,Nb)]
        n2n_a_vel_theta = a_velocity_theta.transpose(0,1).unsqueeze(1).repeat_interleave(self.num_modes,1).reshape(-1)    #[H*K*(N1,...,Nb)]
        n2n_a_edge_index = E_VV_msg
        n2n_a_edge_attr_embs = self._embed_vehicle_interaction_edges(
            edge_index=n2n_a_edge_index,
            graph_edge_attr=E_VV_msg_attr,
            position=n2n_a_position,
            heading=n2n_a_heading,
            velocity_length=n2n_a_vel_length,
            velocity_theta=n2n_a_vel_theta,
        )

        #n2n_h edge                        
        n2n_h_position = n_position.permute(2,0,1,3).reshape(-1, 2)    #[K*(N1,...,Nb)*H,2]
        n2n_h_heading = n_heading.permute(2,0,1).reshape(-1)           #[K*(N1,...,Nb)*H]
        n2n_h_valid_mask = n_valid_mask.permute(2,0,1).reshape(-1, self.num_historical_steps)   #[K*(N1,...,Nb),H]
        n2n_h_valid_mask = n2n_h_valid_mask.unsqueeze(2) & n2n_h_valid_mask.unsqueeze(1)        #[K*(N1,...,Nb),H,H]        
        n2n_h_edge_index = dense_to_sparse(n2n_h_valid_mask)[0]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] > n2n_h_edge_index[0]]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] - n2n_h_edge_index[0] <= self.duration]   
        n2n_h_edge_vector = transform_point_to_local_coordinate(n2n_h_position[n2n_h_edge_index[0]], n2n_h_position[n2n_h_edge_index[1]], n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_length, n2n_h_edge_attr_theta = compute_angles_lengths_2D(n2n_h_edge_vector)
        n2n_h_edge_attr_heading = wrap_angle(n2n_h_heading[n2n_h_edge_index[0]] - n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_interval = n2n_h_edge_index[0] - n2n_h_edge_index[1]
        n2n_h_edge_attr_input = torch.stack([n2n_h_edge_attr_length, n2n_h_edge_attr_theta, n2n_h_edge_attr_heading, n2n_h_edge_attr_interval], dim=-1)
        n2n_h_edge_attr_embs = self.m2m_h_emb_layer(input=n2n_h_edge_attr_input)

        #n2n_s edge
        n2n_s_position = n_position.transpose(0,1).reshape(-1,2)                                #[H*(N1,...,Nb)*K,2]
        n2n_s_heading = n_heading.transpose(0,1).reshape(-1)                                    #[H*(N1,...,Nb)*K]
        n2n_s_valid_mask = n_valid_mask.transpose(0,1).reshape(-1, self.num_modes)              #[H*(N1,...,Nb),K]
        n2n_s_valid_mask = n2n_s_valid_mask.unsqueeze(2) & n2n_s_valid_mask.unsqueeze(1)        #[H*(N1,...,Nb),K,K]
        n2n_s_edge_index = dense_to_sparse(n2n_s_valid_mask)[0]
        n2n_s_edge_index = n2n_s_edge_index[:, n2n_s_edge_index[0] != n2n_s_edge_index[1]]
        n2n_s_edge_vector = transform_point_to_local_coordinate(n2n_s_position[n2n_s_edge_index[0]], n2n_s_position[n2n_s_edge_index[1]], n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_length, n2n_s_edge_attr_theta = compute_angles_lengths_2D(n2n_s_edge_vector)
        n2n_s_edge_attr_heading = wrap_angle(n2n_s_heading[n2n_s_edge_index[0]] - n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_input = torch.stack([n2n_s_edge_attr_length, n2n_s_edge_attr_theta, n2n_s_edge_attr_heading], dim=-1)
        n2n_s_edge_attr_embs = self.m2m_s_emb_layer(input=n2n_s_edge_attr_input)

        #t2n attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)  #[(N1,...,Nb)*H,D]
        n_embs_t = self.t2n_attn_layer(x = [t_embs, n_embs], edge_index = t2n_edge_index, edge_attr = t2n_edge_attr_embs)         #[(N1,...,Nb)*H*K,D]

        #l2n attention
        if self.preserve_base_edges:
            n_embs_l = self.l2n_attn_layer(
                x=[static_l_embs, n_embs],
                edge_index=l2n_edge_index,
                edge_attr=l2n_edge_attr_embs,
            )
            if (
                (self.reachability_residual_scale > 0
                 or return_module_contexts)
                and reachable_l2n_edge_index.numel() != 0
            ):
                reachable_l2n_target_time = (
                    torch.div(
                        reachable_l2n_edge_index[1],
                        self.num_modes,
                        rounding_mode='floor',
                    ) % self.num_historical_steps
                )
                final_reachable_mask = (
                    reachable_l2n_target_time == current_step
                )
                final_reachable_edge = reachable_l2n_edge_index[
                    :, final_reachable_mask
                ]
                reachable_edge_vector = transform_point_to_local_coordinate(
                    l2n_position_l[final_reachable_edge[0]],
                    l2n_position_n[final_reachable_edge[1]],
                    l2n_heading_n[final_reachable_edge[1]],
                )
                (
                    reachable_attr_length,
                    reachable_attr_theta,
                ) = compute_angles_lengths_2D(reachable_edge_vector)
                reachable_attr_heading = wrap_angle(
                    l2n_heading_l[final_reachable_edge[0]]
                    - l2n_heading_n[final_reachable_edge[1]]
                )
                reachable_attr_conflict = (
                    data['lane']['potential_conflict_mask'][
                        final_reachable_edge[0]
                    ].to(l2n_position_l.dtype)
                )
                reachable_attr_input = torch.stack([
                    reachable_attr_length,
                    reachable_attr_theta,
                    reachable_attr_heading,
                    reachable_attr_conflict,
                    reachable_l2n_weight[final_reachable_mask],
                ], dim=-1)
                reachable_attr_embs = self.l2n_emb_layer(
                    input=reachable_attr_input,
                )
                reachable_output = self.l2n_attn_layer(
                    x=[dynamic_l_embs, n_embs],
                    edge_index=final_reachable_edge,
                    edge_attr=reachable_attr_embs,
                )
                involved_target = torch.zeros(
                    n_embs.size(0), dtype=torch.bool, device=n_embs.device,
                )
                involved_target[final_reachable_edge[1]] = True
                reachability_delta = torch.where(
                    involved_target.unsqueeze(-1),
                    reachable_output - n_embs,
                    torch.zeros_like(n_embs),
                )
                if return_module_contexts:
                    refinement_module_contexts['reachability'] = (
                        reachability_delta.reshape(
                            num_all_agent,
                            self.num_historical_steps,
                            self.num_modes,
                            self.hidden_dim,
                        ).mean(dim=2)
                    )
                n_embs_l = (
                    n_embs_l
                    + self.reachability_residual_scale * reachability_delta
                )
        else:
            l2n_target_time = (
                torch.div(
                    l2n_edge_index[1], self.num_modes,
                    rounding_mode='floor',
                ) % self.num_historical_steps
            )
            historical_l2n_mask = l2n_target_time < current_step
            final_l2n_mask = l2n_target_time == current_step
            n_embs_l_static = self.l2n_attn_layer(
                x=[static_l_embs, n_embs],
                edge_index=l2n_edge_index[:, historical_l2n_mask],
                edge_attr=l2n_edge_attr_embs[historical_l2n_mask],
            )
            n_embs_l_dynamic = self.l2n_attn_layer(
                x=[dynamic_l_embs, n_embs],
                edge_index=l2n_edge_index[:, final_l2n_mask],
                edge_attr=l2n_edge_attr_embs[final_l2n_mask],
            )
            n_embs_l = torch.where(
                final_mode_target.reshape(-1, 1),
                n_embs_l_dynamic,
                n_embs_l_static,
            )
            if return_module_contexts:
                reachability_delta = torch.where(
                    final_mode_target.reshape(-1, 1),
                    n_embs_l_dynamic - n_embs,
                    torch.zeros_like(n_embs),
                )
                refinement_module_contexts['reachability'] = (
                    reachability_delta.reshape(
                        num_all_agent,
                        self.num_historical_steps,
                        self.num_modes,
                        self.hidden_dim,
                    ).mean(dim=2)
                )

        n_embs = n_embs_t + n_embs_l
        refinement_conflict_input = n_embs.reshape(
            num_all_agent,
            self.num_historical_steps,
            self.num_modes,
            self.hidden_dim,
        ).permute(1, 2, 0, 3).reshape(-1, self.hidden_dim)  # [H,K,N,D]
        refinement_conflict_output = self._apply_conflict_attention(
            node_embedding=refinement_conflict_input,
            conflict_edge_index=E_VV_conf_mode,
            conflict_edge_embedding=E_VV_conf_mode_embs,
            priority_head_bias=priority_head_bias_mode,
            attention_layer=self.refinement_conflict_attn_layer,
            num_agents=num_all_agent,
        )
        if return_module_contexts:
            refinement_module_contexts['priority'] = (
                refinement_conflict_output - refinement_conflict_input
            ).reshape(
                self.num_historical_steps,
                self.num_modes,
                num_all_agent,
                self.hidden_dim,
            ).permute(2, 0, 1, 3).mean(dim=2)
        n_embs = refinement_conflict_output.reshape(
            self.num_historical_steps,
            self.num_modes,
            num_all_agent,
            self.hidden_dim,
        ).transpose(1, 2).reshape(-1, self.hidden_dim)  # [H,N,K,D]
        #moda attention  
        for i in range(self.num_attn_layers):
            #m2m_a
            n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(1,2).reshape(-1, self.hidden_dim)  #[H*K*(N1,...,Nb),D]
            n_embs = self.n2n_a_attn_layers[i](x = n_embs, edge_index = n2n_a_edge_index, edge_attr = n2n_a_edge_attr_embs)
            #m2m_h
            n_embs = n_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(1,2,0,3).reshape(-1, self.hidden_dim)  #[K*(N1,...,Nb)*H,D]
            n_embs = self.n2n_h_attn_layers[i](x = n_embs, edge_index = n2n_h_edge_index, edge_attr = n2n_h_edge_attr_embs)
            #m2m_s
            n_embs = n_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim).transpose(0,2).reshape(-1, self.hidden_dim)  #[H*(N1,...,Nb)*K,D]
            n_embs = self.n2n_s_attn_layers[i](x = n_embs, edge_index = n2n_s_edge_index, edge_attr = n2n_s_edge_attr_embs)
        n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1, self.hidden_dim)      #[(N1,...,Nb)*H*K,D
        refinement_pre_style = n_embs
        n_embs = self._apply_style_film(
            node_embedding=n_embs,
            style_prefix_embedding=style_prefix_embedding,
            film_generator=self.refinement_style_film,
        )
        if return_module_contexts:
            refinement_module_contexts['style'] = (
                n_embs - refinement_pre_style
            ).reshape(
                num_all_agent,
                self.num_historical_steps,
                self.num_modes,
                self.hidden_dim,
            ).mean(dim=2)

        #generate refinement
        traj_refine = self.traj_refine(n_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 2)                  #[(N1,...,Nb),H,K,F,2]         
        traj_output = transform_traj_to_global_coordinate(proposal + traj_refine, n_position, n_heading)                                               #[(N1,...,Nb),H,K,F,2] 

        traj_propose, traj_output = self._compose_joint_modes(
            node_embedding=n_embs,
            trajectory_propose=traj_propose,
            trajectory_refine=traj_output,
            agent_batch=data['agent']['batch'],
            valid_mask=n_valid_mask,
            num_scenes=data.num_graphs,
            exception_agent_mask=(data['agent']['category'] == 1),
        )

        #generate prob
        prob_output = self.prob_decoder(n_embs.detach()).reshape(-1, self.num_historical_steps, self.num_modes)       #[(N1,...,Nb),H,K]
        prob_output = self.prob_norm(prob_output)                                       #[(N1,...,Nb),H,K]

        outputs = (traj_propose, traj_output, prob_output)
        if not return_module_contexts:
            return outputs

        zero_context = a_embs.new_zeros(
            num_all_agent,
            self.num_historical_steps,
            self.hidden_dim,
        )
        for contexts in (
                proposal_module_contexts, refinement_module_contexts):
            for module_name in ('style', 'reachability', 'priority'):
                contexts.setdefault(module_name, zero_context)
        return outputs, {
            'proposal': proposal_module_contexts,
            'refinement': refinement_module_contexts,
        }
