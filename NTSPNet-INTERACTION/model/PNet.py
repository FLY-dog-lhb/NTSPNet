import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
from itertools import permutations
import math
import pandas as pd

from losses import HuberTrajLoss
from losses import CELoss
from metrics import minJointADE
from metrics import minJointFDE
from metrics import minJointMR
from modules import Backbone
from modules import MapEncoder
from modules import OriginalBackbone
from modules import OriginalResidualBackbone

from utils import generate_target
from utils import generate_predict_mask
from utils import compute_angles_lengths_2D

#torch.set_float32_matmul_precision('high')

class PNet(pl.LightningModule):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_visible_steps: int,
                 num_modes: int,
                 num_attn_layers: int,
                 num_hops: int,
                 num_heads: int,
                 dropout: float,
                 lr: float,
                 weight_decay: float,
                 warmup_epochs: int,
                 T_max: int,
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
                 style_statistics_epsilon: float = None,
                 style_statistics_path: str = None,
                 lambda_hist_aux: float = 0.3,
                 lambda_priority: float = 0.3,
                 lambda_style: float = 0.05,
                 lambda_endpoint: float = 0.0,
                 lambda_late_horizon: float = 0.0,
                 lambda_teacher_distillation: float = 0.0,
                 teacher_distillation_mode: str = 'all_modes_permutation',
                 teacher_advantage_margin: float = 0.05,
                 late_horizon_steps: int = 7,
                 late_horizon_reduction: str = 'agent_mean',
                 priority_order_weight: float = 0.2,
                 style_balance_weight: float = 0.05,
                 motion_source: str = 'causal_difference',
                 csv_motion_cache_dir: str = None,
                 teacher_trajectory_cache_dir: str = None,
                 preserve_base_edges: bool = False,
                 reachability_residual_scale: float = 0.0,
                 trunk_variant: str = 'extended',
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
                 joint_mode_max_exception_agents: int = 2,
                 proposal_conflict_training_scope: str = 'all',
                 baseline_residual_max_scale: float = 0.1,
                 baseline_residual_training_scope: str = 'fusion_only',
                 baseline_residual_provider_lr_scale: float = 0.1,
                 baseline_residual_modules: str = 'all',
                 **kwargs) -> None:
        super(PNet, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.duration = duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_visible_steps = num_visible_steps
        self.num_modes = num_modes
        self.num_attn_layers = num_attn_layers
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.interaction_distance_margin = interaction_distance_margin
        self.observation_step_time = observation_step_time
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
        self.style_statistics_epsilon = (
            ttc_speed_epsilon
            if style_statistics_epsilon is None
            else style_statistics_epsilon
        )
        self.style_statistics_path = style_statistics_path
        self.lambda_hist_aux = lambda_hist_aux
        self.lambda_priority = lambda_priority
        self.lambda_style = lambda_style
        self.lambda_endpoint = lambda_endpoint
        self.lambda_late_horizon = lambda_late_horizon
        self.lambda_teacher_distillation = lambda_teacher_distillation
        self.teacher_distillation_mode = teacher_distillation_mode
        self.teacher_advantage_margin = teacher_advantage_margin
        self.late_horizon_steps = late_horizon_steps
        self.late_horizon_reduction = late_horizon_reduction
        self.priority_order_weight = priority_order_weight
        self.style_balance_weight = style_balance_weight
        self.motion_source = motion_source
        self.csv_motion_cache_dir = csv_motion_cache_dir
        self.teacher_trajectory_cache_dir = teacher_trajectory_cache_dir
        self.preserve_base_edges = preserve_base_edges
        self.reachability_residual_scale = reachability_residual_scale
        self.trunk_variant = trunk_variant
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
        self.proposal_conflict_training_scope = (
            proposal_conflict_training_scope
        )
        self.baseline_residual_max_scale = baseline_residual_max_scale
        self.baseline_residual_training_scope = (
            baseline_residual_training_scope
        )
        self.baseline_residual_provider_lr_scale = (
            baseline_residual_provider_lr_scale
        )
        self.baseline_residual_modules = baseline_residual_modules
        if motion_source not in ('causal_difference', 'csv'):
            raise ValueError(
                "motion_source must be 'causal_difference' or 'csv'"
            )
        if reachability_residual_scale < 0:
            raise ValueError('reachability_residual_scale must be non-negative')
        if trunk_variant not in ('extended', 'original', 'original_residual'):
            raise ValueError(
                "trunk_variant must be 'extended', 'original', or "
                "'original_residual'"
            )
        if not 0.0 < baseline_residual_max_scale <= 1.0:
            raise ValueError(
                'baseline_residual_max_scale must be in (0, 1]'
            )
        if baseline_residual_training_scope not in (
                'fusion_only', 'residual_branch'):
            raise ValueError(
                "baseline_residual_training_scope must be 'fusion_only' "
                "or 'residual_branch'"
            )
        if not 0.0 < baseline_residual_provider_lr_scale <= 1.0:
            raise ValueError(
                'baseline_residual_provider_lr_scale must be in (0, 1]'
            )
        if baseline_residual_modules not in (
                'all', 'style', 'reachability', 'priority',
                'style_reachability', 'style_priority',
                'reachability_priority'):
            raise ValueError(
                "baseline_residual_modules must be 'all', 'style', "
                "'reachability', 'priority', 'style_reachability', "
                "'style_priority', or 'reachability_priority'"
            )
        if (
            baseline_residual_modules != 'all'
            and baseline_residual_training_scope != 'fusion_only'
        ):
            raise ValueError(
                'module-specific residual ablations require '
                "baseline_residual_training_scope='fusion_only'"
            )
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
        if joint_mode_max_exception_agents <= 0:
            raise ValueError(
                'joint_mode_max_exception_agents must be positive'
            )
        if proposal_conflict_training_scope not in (
                'all', 'adapter_only', 'residual_gate_only',
                'scene_coordination_only', 'joint_composition_only'):
            raise ValueError(
                "proposal_conflict_training_scope must be 'all', "
                "'adapter_only', 'residual_gate_only', or "
                "'scene_coordination_only', or 'joint_composition_only'"
            )
        if proposal_conflict_training_scope == 'adapter_only':
            if trunk_variant != 'extended':
                raise ValueError(
                    'adapter_only training requires the extended trunk'
                )
            if proposal_conflict_mode_adapter_scale == 0.0:
                raise ValueError(
                    'adapter_only training requires a non-zero '
                    'proposal_conflict_mode_adapter_scale'
                )
            if proposal_conflict_mode_scale != 0.0:
                raise ValueError(
                    'adapter_only training requires the legacy '
                    'proposal_conflict_mode_scale to remain zero'
                )
            if proposal_conflict_residual_gate_scale != 0.0:
                raise ValueError(
                    'adapter_only training requires the proposal conflict '
                    'residual gate to remain disabled'
                )
            if scene_mode_coordination_scale != 0.0:
                raise ValueError(
                    'adapter_only training requires scene mode coordination '
                    'to remain disabled'
                )
            if joint_mode_composition_scale != 0.0:
                raise ValueError(
                    'adapter_only training requires joint mode composition '
                    'to remain disabled'
                )
        if proposal_conflict_training_scope == 'residual_gate_only':
            if trunk_variant != 'extended':
                raise ValueError(
                    'residual_gate_only training requires the extended trunk'
                )
            if proposal_conflict_residual_gate_scale == 0.0:
                raise ValueError(
                    'residual_gate_only training requires a non-zero '
                    'proposal_conflict_residual_gate_scale'
                )
            if proposal_conflict_mode_scale != 0.0:
                raise ValueError(
                    'residual_gate_only training requires the legacy '
                    'proposal_conflict_mode_scale to remain zero'
                )
            if proposal_conflict_mode_adapter_scale != 0.0:
                raise ValueError(
                    'residual_gate_only training requires the proposal '
                    'conflict mode adapter to remain disabled'
                )
            if scene_mode_coordination_scale != 0.0:
                raise ValueError(
                    'residual_gate_only training requires scene mode '
                    'coordination to remain disabled'
                )
            if joint_mode_composition_scale != 0.0:
                raise ValueError(
                    'residual_gate_only training requires joint mode '
                    'composition to remain disabled'
                )
        if proposal_conflict_training_scope == 'scene_coordination_only':
            if trunk_variant != 'extended':
                raise ValueError(
                    'scene_coordination_only training requires the extended '
                    'trunk'
                )
            if scene_mode_coordination_scale == 0.0:
                raise ValueError(
                    'scene_coordination_only training requires a non-zero '
                    'scene_mode_coordination_scale'
                )
            if any((
                    proposal_conflict_mode_scale != 0.0,
                    proposal_conflict_mode_adapter_scale != 0.0,
                    proposal_conflict_residual_gate_scale != 0.0,
                    joint_mode_composition_scale != 0.0)):
                raise ValueError(
                    'scene_coordination_only training requires all proposal '
                    'conflict and joint-composition experimental branches '
                    'to remain disabled'
                )
        if proposal_conflict_training_scope == 'joint_composition_only':
            if trunk_variant != 'extended':
                raise ValueError(
                    'joint_composition_only training requires the extended '
                    'trunk'
                )
            if joint_mode_composition_scale == 0.0:
                raise ValueError(
                    'joint_composition_only training requires a non-zero '
                    'joint_mode_composition_scale'
                )
            if any((
                    proposal_conflict_mode_scale != 0.0,
                    proposal_conflict_mode_adapter_scale != 0.0,
                    proposal_conflict_residual_gate_scale != 0.0,
                    scene_mode_coordination_scale != 0.0)):
                raise ValueError(
                    'joint_composition_only training requires all earlier '
                    'experimental branches to remain disabled'
                )
        for weight_name, weight_value in (
            ('lambda_hist_aux', lambda_hist_aux),
            ('lambda_priority', lambda_priority),
            ('lambda_style', lambda_style),
            ('lambda_endpoint', lambda_endpoint),
            ('lambda_late_horizon', lambda_late_horizon),
            ('lambda_teacher_distillation', lambda_teacher_distillation),
            ('priority_order_weight', priority_order_weight),
            ('style_balance_weight', style_balance_weight),
        ):
            if weight_value < 0:
                raise ValueError(f'{weight_name} must be non-negative')
        if lambda_late_horizon > 0 and not (
                1 <= late_horizon_steps <= num_future_steps):
            raise ValueError(
                'late_horizon_steps must be between 1 and num_future_steps'
            )
        if late_horizon_reduction not in ('agent_mean', 'scene_mean'):
            raise ValueError(
                "late_horizon_reduction must be 'agent_mean' or 'scene_mean'"
            )
        if lambda_endpoint > 0 and lambda_late_horizon > 0:
            raise ValueError(
                'lambda_endpoint and lambda_late_horizon are mutually exclusive'
            )
        if trunk_variant in ('original', 'original_residual') and (
                lambda_endpoint != 0 or lambda_late_horizon != 0
                or lambda_teacher_distillation != 0):
            raise ValueError(
                'auxiliary trajectory losses must remain zero for the '
                'original control'
            )
        if lambda_teacher_distillation > 0:
            if not teacher_trajectory_cache_dir:
                raise ValueError(
                    'teacher_trajectory_cache_dir is required when '
                    'lambda_teacher_distillation is positive'
                )
            if (
                teacher_distillation_mode == 'all_modes_permutation'
                and num_modes > 8
            ):
                raise ValueError(
                    'teacher trajectory permutation matching requires '
                    'num_modes <= 8'
                )
        if teacher_distillation_mode not in (
                'all_modes_permutation',
                'advantage_gated_single_mode',
                'advantage_gated_scene_mode'):
            raise ValueError(
                "teacher_distillation_mode must be 'all_modes_permutation' "
                "or 'advantage_gated_single_mode', or "
                "'advantage_gated_scene_mode'"
            )
        if teacher_advantage_margin < 0:
            raise ValueError('teacher_advantage_margin must be non-negative')
        if trunk_variant in ('original', 'original_residual') and (
                joint_mode_composition_scale != 0):
            raise ValueError(
                'joint mode composition requires the extended trunk'
            )

        if trunk_variant == 'original_residual':
            if proposal_conflict_training_scope != 'all':
                raise ValueError(
                    'original_residual requires '
                    "proposal_conflict_training_scope='all'"
                )
            if any((
                    proposal_conflict_mode_scale != 0.0,
                    proposal_conflict_mode_adapter_scale != 0.0,
                    proposal_conflict_residual_gate_scale != 0.0,
                    scene_mode_coordination_scale != 0.0,
                    joint_mode_composition_scale != 0.0)):
                raise ValueError(
                    'original_residual requires all historical experimental '
                    'output branches to remain disabled'
                )

        if trunk_variant == 'original':
            self.Backbone = OriginalBackbone(
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                duration=duration,
                a2a_radius=a2a_radius,
                l2a_radius=l2a_radius,
                num_attn_layers=num_attn_layers,
                num_modes=num_modes,
                num_heads=num_heads,
                dropout=dropout,
                motion_source=motion_source,
            )
        else:
            extended_backbone = Backbone(
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                duration=duration,
                a2a_radius=a2a_radius,
                l2a_radius=l2a_radius,
                num_attn_layers=num_attn_layers,
                num_modes=num_modes,
                num_heads=num_heads,
                dropout=dropout,
                interaction_distance_margin=interaction_distance_margin,
                observation_step_time=observation_step_time,
                lane_arrival_time_threshold=lane_arrival_time_threshold,
                ttc_speed_epsilon=ttc_speed_epsilon,
                interaction_ttc_threshold=interaction_ttc_threshold,
                interaction_collision_distance=interaction_collision_distance,
                conflict_lane_radius=conflict_lane_radius,
                traffic_base_radius=traffic_base_radius,
                traffic_speed_scale=traffic_speed_scale,
                traffic_backward_attention_ratio=traffic_backward_attention_ratio,
                traffic_max_attention_nodes=traffic_max_attention_nodes,
                traffic_current_radius=traffic_current_radius,
                traffic_lane_lateral_radius=traffic_lane_lateral_radius,
                traffic_topology_base_distance=traffic_topology_base_distance,
                traffic_lane_change_lateral_radius=traffic_lane_change_lateral_radius,
                traffic_lane_change_lateral_speed=traffic_lane_change_lateral_speed,
                style_thw_max=style_thw_max,
                style_ttc_max=style_ttc_max,
                style_arrival_time_fill=style_arrival_time_fill,
                style_transformer_layers=style_transformer_layers,
                style_gap_reset_steps=style_gap_reset_steps,
                style_pool_min_length=style_pool_min_length,
                style_min_visible_steps=style_min_visible_steps,
                style_min_last_segment_steps=style_min_last_segment_steps,
                style_statistics_epsilon=self.style_statistics_epsilon,
                style_statistics_path=style_statistics_path,
                motion_source=motion_source,
                preserve_base_edges=preserve_base_edges,
                reachability_residual_scale=reachability_residual_scale,
                style_film_mode=style_film_mode,
                style_film_bound=style_film_bound,
                conflict_update_scope=conflict_update_scope,
                priority_injection_mode=priority_injection_mode,
                proposal_conflict_mode_scale=proposal_conflict_mode_scale,
                proposal_conflict_mode_adapter_scale=(
                    proposal_conflict_mode_adapter_scale
                ),
                proposal_conflict_residual_gate_scale=(
                    proposal_conflict_residual_gate_scale
                ),
                scene_mode_coordination_scale=(
                    scene_mode_coordination_scale
                ),
                joint_mode_composition_scale=(
                    joint_mode_composition_scale
                ),
                joint_mode_assignment_mode=joint_mode_assignment_mode,
                joint_mode_max_exception_agents=(
                    joint_mode_max_exception_agents
                ),
            )
            if trunk_variant == 'original_residual':
                original_backbone = OriginalBackbone(
                    hidden_dim=hidden_dim,
                    num_historical_steps=num_historical_steps,
                    num_future_steps=num_future_steps,
                    duration=duration,
                    a2a_radius=a2a_radius,
                    l2a_radius=l2a_radius,
                    num_attn_layers=num_attn_layers,
                    num_modes=num_modes,
                    num_heads=num_heads,
                    dropout=dropout,
                    motion_source=motion_source,
                )
                self.Backbone = OriginalResidualBackbone(
                    base=original_backbone,
                    residual=extended_backbone,
                    hidden_dim=hidden_dim,
                    num_modes=num_modes,
                    max_scale=baseline_residual_max_scale,
                    training_scope=baseline_residual_training_scope,
                    active_modules=baseline_residual_modules,
                )
            else:
                self.Backbone = extended_backbone
        self.MapEncoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )

        self.reg_loss_traj = HuberTrajLoss()
        self.cls_loss = CELoss()
        teacher_permutations = (
            torch.tensor(
                list(permutations(range(num_modes))),
                dtype=torch.long,
            )
            if num_modes <= 8
            else torch.empty((0, num_modes), dtype=torch.long)
        )
        self.register_buffer(
            '_teacher_mode_permutations',
            teacher_permutations,
            persistent=False,
        )

        self.min_joint_ade = minJointADE()
        self.min_joint_fde = minJointFDE()
        self.min_joint_mr = minJointMR(threshold=2.0)

        self._columns = ['case_id', 'track_id', 'frame_id', 'timestamp_ms', 'interesting_agent',
                         'x1', 'y1', 'psi_rad1',
                         'x2', 'y2', 'psi_rad2',
                         'x3', 'y3', 'psi_rad3',
                         'x4', 'y4', 'psi_rad4',
                         'x5', 'y5', 'psi_rad5',
                         'x6', 'y6', 'psi_rad6']
        self.test_output = dict()

        # validation prediction saving (enabled by val.py via --save_predictions)
        self.save_val_predictions = False
        self.val_output = dict()
        self._val_columns = (
            ['case_id', 'track_id', 'frame_id', 'timestamp_ms', 'x_gt', 'y_gt'] +
            [f'{c}{k+1}' for k in range(6) for c in ('x', 'y')]
        )
        self._configure_trainable_scope()

    def _configure_trainable_scope(self) -> None:
        """Freeze the base model for an isolated proposal conflict branch."""
        if self.trunk_variant == 'original_residual':
            self.MapEncoder.requires_grad_(False)
            return
        if self.proposal_conflict_training_scope == 'all':
            return
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if self.proposal_conflict_training_scope == 'adapter_only':
            parameter = self.Backbone.proposal_conflict_mode_tokens.weight
            parameters = (parameter,)
        elif self.proposal_conflict_training_scope == 'residual_gate_only':
            parameter = (
                self.Backbone.proposal_conflict_residual_gate_tokens.weight
            )
            parameters = (parameter,)
        elif self.proposal_conflict_training_scope == 'scene_coordination_only':
            parameters = tuple(self.Backbone.scene_mode_coordination.parameters())
        else:
            if self.joint_mode_assignment_mode in (
                    'permutation', 'limited_permutation'):
                parameters = (self.Backbone.joint_mode_composer.weight,)
            else:
                parameters = tuple(
                    self.Backbone.joint_mode_composer.parameters()
                )
        for parameter in parameters:
            parameter.requires_grad_(True)

    def _proposal_conflict_adapter_anchor(self) -> torch.Tensor:
        """Keep isolated zero-conflict minibatches differentiable."""
        if self.proposal_conflict_training_scope == 'adapter_only':
            parameters = (self.Backbone.proposal_conflict_mode_tokens.weight,)
        elif self.proposal_conflict_training_scope == 'residual_gate_only':
            parameters = ((
                self.Backbone.proposal_conflict_residual_gate_tokens.weight
            ),)
        elif self.proposal_conflict_training_scope == 'scene_coordination_only':
            parameters = tuple(self.Backbone.scene_mode_coordination.parameters())
        else:
            if self.joint_mode_assignment_mode in (
                    'permutation', 'limited_permutation'):
                parameters = (self.Backbone.joint_mode_composer.weight,)
            else:
                parameters = tuple(
                    self.Backbone.joint_mode_composer.parameters()
                )
        return sum(parameter.sum() for parameter in parameters) * 0.0

    def forward(self, 
                data: Batch):
        lane_embs = self.MapEncoder(data=data)
        pred = self.Backbone(data=data, l_embs=lane_embs)
        return pred

    def train(self, mode: bool = True):
        super().train(mode)
        if self.trunk_variant == 'original_residual':
            self.MapEncoder.eval()
            self.Backbone.train(mode)
        return self

    @staticmethod
    def _checkpoint_state(checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False,
        )
        state_dict = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError(
                f'checkpoint has no state dictionary: {checkpoint_path}'
            )
        return state_dict

    @staticmethod
    def _extract_prefixed_state(state_dict, prefix: str):
        extracted = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not extracted:
            raise ValueError(
                f'checkpoint contains no parameters with prefix {prefix!r}'
            )
        return extracted

    def load_original_residual_weights(self,
                                       base_checkpoint: str,
                                       residual_checkpoint: str) -> None:
        if self.trunk_variant != 'original_residual':
            raise RuntimeError(
                'dual checkpoint loading requires trunk_variant='
                "'original_residual'"
            )
        base_state = self._checkpoint_state(base_checkpoint)
        residual_state = self._checkpoint_state(residual_checkpoint)
        self.Backbone.base.load_state_dict(
            self._extract_prefixed_state(base_state, 'Backbone.'),
            strict=True,
        )
        self.MapEncoder.load_state_dict(
            self._extract_prefixed_state(base_state, 'MapEncoder.'),
            strict=True,
        )
        self.Backbone.residual.load_state_dict(
            self._extract_prefixed_state(residual_state, 'Backbone.'),
            strict=True,
        )
        for gate in self.Backbone.gates.values():
            if gate.detach().item() != 0.0:
                raise RuntimeError(
                    'baseline-residual module gates must initialize to zero'
                )

    def _compute_style_losses(self, data: Batch):
        """Return pseudo-label CE and batch marginal-to-uniform KL losses."""
        self.Backbone.require_style_statistics()
        if 'style_prefix_reliable_mask' in data['agent']:
            reliable_mask = data['agent']['style_prefix_reliable_mask']
            style_logits = data['agent']['style_prefix_logits']
            style_probability = data['agent']['style_prefix_probability']
            pseudo_label = data['agent']['style_prefix_pseudo_label']
        else:
            reliable_mask = data['agent']['style_reliable_mask']
            style_logits = data['agent']['style_logits']
            style_probability = data['agent']['style_probability']
            pseudo_label = data['agent']['style_pseudo_label']

        if not reliable_mask.any():
            zero_loss = style_logits.sum() * 0.0
            return zero_loss, zero_loss
        if (pseudo_label[reliable_mask] < 0).any():
            raise RuntimeError('reliable style anchors must have driving-style pseudo labels')

        style_cls_loss = F.cross_entropy(
            style_logits[reliable_mask],
            pseudo_label[reliable_mask],
        )
        mean_probability = style_probability[reliable_mask].mean(dim=0)
        mean_probability = mean_probability.clamp_min(torch.finfo(mean_probability.dtype).tiny)
        uniform_probability = 1.0 / mean_probability.numel()
        style_balance_loss = (
            mean_probability
            * (mean_probability.log() - math.log(uniform_probability))
        ).sum()
        return style_cls_loss, style_balance_loss

    def _compute_priority_losses(self, data: Batch):
        logits = data['agent']['priority_instance_logits']
        probability = data['agent']['priority_instance_probability']
        pseudo_label = data['agent']['priority_pseudo_label']
        valid_mask = data['agent']['priority_pseudo_label_valid_mask']
        order_target = data['agent']['priority_order_target']
        if not valid_mask.any():
            zero_loss = logits.sum() * 0.0
            return zero_loss, zero_loss, zero_loss

        priority_ce_loss = F.cross_entropy(
            logits[valid_mask],
            pseudo_label[valid_mask],
        )
        priority_score = probability[:, 0] - probability[:, 1]
        priority_order_loss = F.smooth_l1_loss(
            priority_score[valid_mask] - order_target[valid_mask],
            torch.zeros_like(priority_score[valid_mask]),
            reduction='sum',
        ) / valid_mask.sum().to(priority_score.dtype).clamp_min(1.0)
        priority_loss = (
            priority_ce_loss
            + self.priority_order_weight * priority_order_loss
        )
        return priority_loss, priority_ce_loss, priority_order_loss

    def _compute_endpoint_auxiliary(self,
                                    traj_propose: torch.Tensor,
                                    traj_output: torch.Tensor,
                                    target_traj: torch.Tensor,
                                    target_mask: torch.Tensor,
                                    predict_mask: torch.Tensor,
                                    agent_batch: torch.Tensor,
                                    zero_loss: torch.Tensor) -> torch.Tensor:
        """Regress the endpoint-oracle mode without replacing the ADE mode."""
        endpoint_valid = (
            predict_mask[:, -1] & target_mask[:, -1, -1]
        )
        if not endpoint_valid.any():
            return zero_loss

        proposal_endpoint = traj_propose[:, -1, :, -1]
        output_endpoint = traj_output[:, -1, :, -1]
        target_endpoint = target_traj[:, -1, -1]
        selected_proposal = []
        selected_output = []
        selected_target = []
        for scene_index in agent_batch.unique(sorted=True):
            scene_valid = endpoint_valid & (agent_batch == scene_index)
            if not scene_valid.any():
                continue
            joint_endpoint_error = torch.linalg.vector_norm(
                proposal_endpoint[scene_valid]
                - target_endpoint[scene_valid].unsqueeze(1),
                dim=-1,
            ).sum(dim=0)
            endpoint_best_mode = joint_endpoint_error.argmin()
            selected_proposal.append(
                proposal_endpoint[scene_valid, endpoint_best_mode]
            )
            selected_output.append(
                output_endpoint[scene_valid, endpoint_best_mode]
            )
            selected_target.append(target_endpoint[scene_valid])
        if not selected_target:
            return zero_loss

        selected_proposal = torch.cat(selected_proposal, dim=0)
        selected_output = torch.cat(selected_output, dim=0)
        selected_target = torch.cat(selected_target, dim=0)
        endpoint_propose_loss = self.reg_loss_traj(
            selected_proposal, selected_target,
        )
        endpoint_refine_loss = self.reg_loss_traj(
            selected_output, selected_target,
        )
        return endpoint_refine_loss + 0.5 * endpoint_propose_loss

    def _compute_late_horizon_auxiliary(
            self,
            traj_propose: torch.Tensor,
            traj_output: torch.Tensor,
            target_traj: torch.Tensor,
            target_mask: torch.Tensor,
            predict_mask: torch.Tensor,
            agent_batch: torch.Tensor,
            zero_loss: torch.Tensor) -> torch.Tensor:
        """Regress one scene-level mode across the complete late horizon."""
        late_slice = slice(-self.late_horizon_steps, None)
        late_valid = (
            predict_mask[:, -1]
            & target_mask[:, -1, late_slice].all(dim=-1)
        )
        if not late_valid.any():
            return zero_loss

        proposal_late = traj_propose[:, -1, :, late_slice]
        output_late = traj_output[:, -1, :, late_slice]
        target_late = target_traj[:, -1, late_slice]
        selected_proposal = []
        selected_output = []
        selected_target = []
        scene_losses = []
        for scene_index in agent_batch.unique(sorted=True):
            scene_valid = late_valid & (agent_batch == scene_index)
            if not scene_valid.any():
                continue
            joint_late_error = torch.linalg.vector_norm(
                proposal_late[scene_valid]
                - target_late[scene_valid].unsqueeze(1),
                dim=-1,
            ).sum(dim=(0, 2))
            late_best_mode = joint_late_error.argmin()
            selected_proposal.append(
                proposal_late[scene_valid, late_best_mode]
            )
            selected_output.append(
                output_late[scene_valid, late_best_mode]
            )
            selected_target.append(target_late[scene_valid])
            if self.late_horizon_reduction == 'scene_mean':
                scene_propose_loss = self.reg_loss_traj(
                    selected_proposal[-1], selected_target[-1],
                )
                scene_refine_loss = self.reg_loss_traj(
                    selected_output[-1], selected_target[-1],
                )
                scene_losses.append(
                    scene_refine_loss + 0.5 * scene_propose_loss
                )
        if not selected_target:
            return zero_loss

        if self.late_horizon_reduction == 'scene_mean':
            return torch.stack(scene_losses).mean()

        selected_proposal = torch.cat(selected_proposal, dim=0)
        selected_output = torch.cat(selected_output, dim=0)
        selected_target = torch.cat(selected_target, dim=0)
        late_propose_loss = self.reg_loss_traj(
            selected_proposal, selected_target,
        )
        late_refine_loss = self.reg_loss_traj(
            selected_output, selected_target,
        )
        return late_refine_loss + 0.5 * late_propose_loss

    def _compute_trajectory_objective(self,
                                      data: Batch,
                                      traj_propose: torch.Tensor,
                                      traj_output: torch.Tensor,
                                      prob_output: torch.Tensor):
        agent_mask = data['agent']['category'] == 1
        traj_propose = traj_propose[agent_mask]
        traj_output = traj_output[agent_mask]
        prob_output = prob_output[agent_mask]
        target_traj, target_mask = generate_target(
            position=data['agent']['position'],
            mask=data['agent']['visible_mask'],
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
        )
        target_traj = target_traj[agent_mask]
        target_mask = target_mask[agent_mask]
        agent_batch = data['agent']['batch'][agent_mask]

        errors = (
            torch.norm(
                traj_propose - target_traj.unsqueeze(2),
                p=2,
                dim=-1,
            ) * target_mask.unsqueeze(2)
        ).sum(dim=-1)
        joint_errors = [
            error.sum(dim=0, keepdim=True)
            for error in unbatch(errors, agent_batch)
        ]
        joint_errors = torch.cat(joint_errors, dim=0)
        num_agent_per_batch = torch.bincount(agent_batch)
        best_mode_index = joint_errors.argmin(dim=-1).repeat_interleave(
            num_agent_per_batch, 0,
        )
        agent_index = torch.arange(traj_propose.size(0), device=traj_propose.device)
        time_index = torch.arange(
            self.num_historical_steps, device=traj_propose.device,
        )
        traj_best_propose = traj_propose[
            agent_index[:, None], time_index[None, :], best_mode_index,
        ]
        traj_best_output = traj_output[
            agent_index[:, None], time_index[None, :], best_mode_index,
        ]
        predict_mask = generate_predict_mask(
            data['agent']['visible_mask'][agent_mask, :self.num_historical_steps],
            self.num_visible_steps,
        )

        zero_loss = traj_propose.sum() * 0.0

        def anchor_losses(historical_step: int):
            valid_agent = predict_mask[:, historical_step]
            if not valid_agent.any():
                return zero_loss, zero_loss, zero_loss
            point_mask = target_mask[valid_agent, historical_step]
            if not point_mask.any():
                regression_propose = zero_loss
                regression_refine = zero_loss
            else:
                regression_propose = self.reg_loss_traj(
                    traj_best_propose[valid_agent, historical_step][point_mask],
                    target_traj[valid_agent, historical_step][point_mask],
                )
                regression_refine = self.reg_loss_traj(
                    traj_best_output[valid_agent, historical_step][point_mask],
                    target_traj[valid_agent, historical_step][point_mask],
                )
            mode_loss = self.cls_loss(
                prob_output[valid_agent, historical_step],
                best_mode_index[valid_agent, historical_step],
            )
            return regression_propose, regression_refine, mode_loss

        main_propose, main_refine, main_mode = anchor_losses(
            self.num_historical_steps - 1,
        )
        main_loss = main_refine + 0.5 * main_propose + 0.2 * main_mode
        endpoint_aux_loss = zero_loss
        if self.lambda_endpoint > 0:
            endpoint_aux_loss = self._compute_endpoint_auxiliary(
                traj_propose=traj_propose,
                traj_output=traj_output,
                target_traj=target_traj,
                target_mask=target_mask,
                predict_mask=predict_mask,
                agent_batch=agent_batch,
                zero_loss=zero_loss,
            )
        late_horizon_aux_loss = zero_loss
        if self.lambda_late_horizon > 0:
            late_horizon_aux_loss = self._compute_late_horizon_auxiliary(
                traj_propose=traj_propose,
                traj_output=traj_output,
                target_traj=target_traj,
                target_mask=target_mask,
                predict_mask=predict_mask,
                agent_batch=agent_batch,
                zero_loss=zero_loss,
            )

        historical_anchor_losses = []
        for historical_step in range(self.num_historical_steps - 1):
            hist_propose, hist_refine, hist_mode = anchor_losses(historical_step)
            if predict_mask[:, historical_step].any():
                historical_anchor_losses.append(
                    hist_refine + 0.5 * hist_propose + 0.1 * hist_mode
                )
        if historical_anchor_losses:
            hist_aux_loss = torch.stack(historical_anchor_losses).mean()
        else:
            hist_aux_loss = zero_loss

        return {
            'main_loss': main_loss,
            'endpoint_aux_loss': endpoint_aux_loss,
            'late_horizon_aux_loss': late_horizon_aux_loss,
            'hist_aux_loss': hist_aux_loss,
            'main_propose': main_propose,
            'main_refine': main_refine,
            'main_mode': main_mode,
            'agent_mask': agent_mask,
            'agent_batch': agent_batch,
            'target_traj': target_traj,
            'target_mask': target_mask,
            'predict_mask': predict_mask,
            'traj_output': traj_output,
        }

    def _compute_teacher_trajectory_distillation(
            self,
            data: Batch,
            traj_output: torch.Tensor,
            agent_mask: torch.Tensor,
            agent_batch: torch.Tensor) -> torch.Tensor:
        """Match six teacher modes one-to-one and average agents per scene."""
        if 'teacher_trajectory' not in data['agent']:
            raise RuntimeError(
                'teacher trajectories are missing from the training batch; '
                'check teacher_trajectory_cache_dir'
            )
        if 'teacher_trajectory_valid_mask' not in data['agent']:
            raise RuntimeError(
                'teacher trajectory validity mask is missing from the batch'
            )

        teacher_valid = data['agent'][
            'teacher_trajectory_valid_mask'
        ][agent_mask].bool()
        if not teacher_valid.all():
            raise RuntimeError(
                'every predicted agent must have a cached teacher trajectory'
            )
        student = traj_output[:, -1]
        teacher = data['agent']['teacher_trajectory'][agent_mask].detach()
        if teacher.shape != student.shape:
            raise RuntimeError(
                'cached teacher trajectory shape does not match student '
                f'output: teacher={tuple(teacher.shape)}, '
                f'student={tuple(student.shape)}'
            )
        if not torch.isfinite(teacher).all():
            raise RuntimeError('cached teacher trajectory contains non-finite values')

        pair_cost = F.smooth_l1_loss(
            student[:, :, None].expand(
                -1, -1, self.num_modes, -1, -1,
            ),
            teacher[:, None, :].expand(
                -1, self.num_modes, -1, -1, -1,
            ),
            reduction='none',
        ).sum(dim=-1).mean(dim=-1)
        mode_index = torch.arange(
            self.num_modes,
            device=student.device,
        )
        permutation_cost = pair_cost[
            :, mode_index, self._teacher_mode_permutations
        ].mean(dim=-1)
        best_permutation = self._teacher_mode_permutations[
            permutation_cost.argmin(dim=-1)
        ]
        matched_teacher = teacher[
            torch.arange(student.size(0), device=student.device)[:, None],
            best_permutation,
        ]
        agent_loss = F.smooth_l1_loss(
            student,
            matched_teacher,
            reduction='none',
        ).sum(dim=-1).mean(dim=(1, 2))
        scene_losses = [
            agent_loss[agent_batch == scene_index].mean()
            for scene_index in agent_batch.unique(sorted=True)
        ]
        return torch.stack(scene_losses).mean()

    def _compute_teacher_advantage_distillation(
            self,
            data: Batch,
            traj_output: torch.Tensor,
            agent_mask: torch.Tensor,
            agent_batch: torch.Tensor,
            target_traj: torch.Tensor,
            target_mask: torch.Tensor,
            predict_mask: torch.Tensor):
        """Distill one mode only when the teacher has a real FDE advantage."""
        if 'teacher_trajectory' not in data['agent']:
            raise RuntimeError(
                'teacher trajectories are missing from the training batch; '
                'check teacher_trajectory_cache_dir'
            )
        if 'teacher_trajectory_valid_mask' not in data['agent']:
            raise RuntimeError(
                'teacher trajectory validity mask is missing from the batch'
            )
        teacher_valid = data['agent'][
            'teacher_trajectory_valid_mask'
        ][agent_mask].bool()
        if not teacher_valid.all():
            raise RuntimeError(
                'every predicted agent must have a cached teacher trajectory'
            )

        student = traj_output[:, -1]
        teacher = data['agent']['teacher_trajectory'][agent_mask].detach()
        if teacher.shape != student.shape:
            raise RuntimeError(
                'cached teacher trajectory shape does not match student '
                f'output: teacher={tuple(teacher.shape)}, '
                f'student={tuple(student.shape)}'
            )
        if not torch.isfinite(teacher).all():
            raise RuntimeError(
                'cached teacher trajectory contains non-finite values'
            )

        final_target = target_traj[:, -1]
        final_target_mask = target_mask[:, -1]
        official_valid = predict_mask[:, -1] & final_target_mask.all(dim=-1)
        zero_loss = student.sum() * 0.0
        if not official_valid.any():
            return zero_loss, zero_loss

        target_endpoint = final_target[:, -1]
        student_fde = torch.linalg.vector_norm(
            student[:, :, -1] - target_endpoint[:, None],
            dim=-1,
        )
        teacher_fde = torch.linalg.vector_norm(
            teacher[:, :, -1] - target_endpoint[:, None],
            dim=-1,
        )
        teacher_best_fde, teacher_best_mode = teacher_fde.min(dim=-1)
        student_best_fde = student_fde.min(dim=-1).values
        eligible = official_valid & (
            student_best_fde.detach() - teacher_best_fde
            > self.teacher_advantage_margin
        )
        eligible_rate = (
            eligible.sum().to(student.dtype)
            / official_valid.sum().to(student.dtype)
        )
        if not eligible.any():
            return zero_loss, eligible_rate

        agent_index = torch.arange(student.size(0), device=student.device)
        selected_teacher = teacher[agent_index, teacher_best_mode][eligible]
        eligible_student = student[eligible]
        expanded_teacher = selected_teacher[:, None].expand_as(
            eligible_student
        )
        trajectory_cost = F.smooth_l1_loss(
            eligible_student,
            expanded_teacher,
            reduction='none',
        ).sum(dim=-1).mean(dim=-1)
        endpoint_cost = F.smooth_l1_loss(
            eligible_student[:, :, -1],
            expanded_teacher[:, :, -1],
            reduction='none',
        ).sum(dim=-1)
        mode_cost = trajectory_cost + endpoint_cost
        selected_student_mode = mode_cost.argmin(dim=-1)
        selected_mode_loss = mode_cost[
            torch.arange(mode_cost.size(0), device=student.device),
            selected_student_mode,
        ]
        eligible_batch = agent_batch[eligible]
        scene_losses = [
            selected_mode_loss[eligible_batch == scene_index].mean()
            for scene_index in eligible_batch.unique(sorted=True)
        ]
        return torch.stack(scene_losses).mean(), eligible_rate

    def _compute_teacher_scene_mode_distillation(
            self,
            data: Batch,
            traj_output: torch.Tensor,
            agent_mask: torch.Tensor,
            agent_batch: torch.Tensor,
            target_traj: torch.Tensor,
            target_mask: torch.Tensor,
            predict_mask: torch.Tensor):
        """Distill one shared joint mode in teacher-advantage scenes."""
        if 'teacher_trajectory' not in data['agent']:
            raise RuntimeError(
                'teacher trajectories are missing from the training batch; '
                'check teacher_trajectory_cache_dir'
            )
        if 'teacher_trajectory_valid_mask' not in data['agent']:
            raise RuntimeError(
                'teacher trajectory validity mask is missing from the batch'
            )
        teacher_valid = data['agent'][
            'teacher_trajectory_valid_mask'
        ][agent_mask].bool()
        if not teacher_valid.all():
            raise RuntimeError(
                'every predicted agent must have a cached teacher trajectory'
            )

        student = traj_output[:, -1]
        teacher = data['agent']['teacher_trajectory'][agent_mask].detach()
        if teacher.shape != student.shape:
            raise RuntimeError(
                'cached teacher trajectory shape does not match student '
                f'output: teacher={tuple(teacher.shape)}, '
                f'student={tuple(student.shape)}'
            )
        if not torch.isfinite(teacher).all():
            raise RuntimeError(
                'cached teacher trajectory contains non-finite values'
            )

        final_target = target_traj[:, -1]
        final_target_mask = target_mask[:, -1]
        official_valid = predict_mask[:, -1] & final_target_mask.all(dim=-1)
        zero_loss = student.sum() * 0.0
        scene_losses = []
        official_scene_count = 0
        eligible_scene_count = 0
        target_endpoint = final_target[:, -1]
        for scene_index in agent_batch.unique(sorted=True):
            scene_valid = official_valid & (agent_batch == scene_index)
            if not scene_valid.any():
                continue
            official_scene_count += 1
            scene_student = student[scene_valid]
            scene_teacher = teacher[scene_valid]
            scene_target_endpoint = target_endpoint[scene_valid]
            student_joint_fde = torch.linalg.vector_norm(
                scene_student[:, :, -1]
                - scene_target_endpoint[:, None],
                dim=-1,
            ).mean(dim=0)
            teacher_joint_fde = torch.linalg.vector_norm(
                scene_teacher[:, :, -1]
                - scene_target_endpoint[:, None],
                dim=-1,
            ).mean(dim=0)
            teacher_best_fde, teacher_best_mode = (
                teacher_joint_fde.min(dim=0)
            )
            student_best_fde = student_joint_fde.min(dim=0).values
            if (
                student_best_fde.detach() - teacher_best_fde
                <= self.teacher_advantage_margin
            ):
                continue

            eligible_scene_count += 1
            selected_teacher = scene_teacher[:, teacher_best_mode]
            expanded_teacher = selected_teacher[:, None].expand_as(
                scene_student
            )
            trajectory_cost = F.smooth_l1_loss(
                scene_student,
                expanded_teacher,
                reduction='none',
            ).sum(dim=-1).mean(dim=-1)
            endpoint_cost = F.smooth_l1_loss(
                scene_student[:, :, -1],
                expanded_teacher[:, :, -1],
                reduction='none',
            ).sum(dim=-1)
            agent_mode_cost = trajectory_cost + endpoint_cost
            student_best_mode = agent_mode_cost.mean(dim=0).argmin()
            scene_losses.append(
                agent_mode_cost[:, student_best_mode].mean()
            )

        if official_scene_count == 0:
            return zero_loss, zero_loss
        eligible_rate = student.new_tensor(
            eligible_scene_count / official_scene_count
        )
        if not scene_losses:
            return zero_loss, eligible_rate
        return torch.stack(scene_losses).mean(), eligible_rate

    def _compute_original_trajectory_objective(self,
                                               data: Batch,
                                               traj_propose: torch.Tensor,
                                               traj_output: torch.Tensor,
                                               prob_output: torch.Tensor):
        """Reproduce the version-22 proposal + refinement + CE objective."""
        agent_mask = data['agent']['category'] == 1
        traj_propose = traj_propose[agent_mask]
        traj_output = traj_output[agent_mask]
        prob_output = prob_output[agent_mask]
        target_traj, target_mask = generate_target(
            position=data['agent']['position'],
            mask=data['agent']['visible_mask'],
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
        )
        target_traj = target_traj[agent_mask]
        target_mask = target_mask[agent_mask]
        agent_batch = data['agent']['batch'][agent_mask]

        errors = (
            torch.norm(
                traj_propose - target_traj.unsqueeze(2), p=2, dim=-1,
            ) * target_mask.unsqueeze(2)
        ).sum(dim=-1)
        joint_errors = torch.cat([
            error.sum(dim=0, keepdim=True)
            for error in unbatch(errors, agent_batch)
        ], dim=0)
        num_agent_per_batch = torch.bincount(agent_batch)
        best_mode_index = joint_errors.argmin(dim=-1).repeat_interleave(
            num_agent_per_batch, 0,
        )
        agent_index = torch.arange(
            traj_propose.size(0), device=traj_propose.device,
        )
        time_index = torch.arange(
            self.num_historical_steps, device=traj_propose.device,
        )
        traj_best_propose = traj_propose[
            agent_index[:, None], time_index[None, :], best_mode_index,
        ]
        traj_best_output = traj_output[
            agent_index[:, None], time_index[None, :], best_mode_index,
        ]
        predict_mask = generate_predict_mask(
            data['agent']['visible_mask'][
                agent_mask, :self.num_historical_steps
            ],
            self.num_visible_steps,
        )
        point_mask = target_mask[predict_mask]
        zero_loss = traj_propose.sum() * 0.0
        if point_mask.any():
            reg_propose = self.reg_loss_traj(
                traj_best_propose[predict_mask][point_mask],
                target_traj[predict_mask][point_mask],
            )
            reg_refine = self.reg_loss_traj(
                traj_best_output[predict_mask][point_mask],
                target_traj[predict_mask][point_mask],
            )
        else:
            reg_propose = zero_loss
            reg_refine = zero_loss
        if predict_mask.any():
            cls_loss = self.cls_loss(
                prob_output[predict_mask], best_mode_index[predict_mask],
            )
        else:
            cls_loss = zero_loss
        return {
            'loss': reg_propose + reg_refine + cls_loss,
            'reg_propose': reg_propose,
            'reg_refine': reg_refine,
            'cls_loss': cls_loss,
            'agent_mask': agent_mask,
            'agent_batch': agent_batch,
            'target_traj': target_traj,
            'target_mask': target_mask,
            'traj_output': traj_output,
        }

    def on_fit_start(self) -> None:
        if self.trunk_variant in ('extended', 'original_residual'):
            self.Backbone.require_style_statistics()

    def training_step(self,data,batch_idx):
        traj_propose, traj_output, prob_output = self(data)
        if self.trunk_variant in ('original', 'original_residual'):
            objective = self._compute_original_trajectory_objective(
                data, traj_propose, traj_output, prob_output,
            )
            self.log('train_reg_loss_traj_propose', objective['reg_propose'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_reg_loss_traj_refine', objective['reg_refine'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_cls_loss', objective['cls_loss'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_loss', objective['loss'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            if self.trunk_variant == 'original_residual':
                scales = self.Backbone.fusion_scales()
                for module_name, scale in scales.items():
                    self.log(f'train_baseline_{module_name}_scale', scale, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            return objective['loss']
        trajectory_loss = self._compute_trajectory_objective(
            data, traj_propose, traj_output, prob_output,
        )
        style_cls_loss, style_balance_loss = self._compute_style_losses(data)
        style_loss = (
            style_cls_loss
            + self.style_balance_weight * style_balance_loss
        )
        priority_loss, priority_ce_loss, priority_order_loss = (
            self._compute_priority_losses(data)
        )
        loss = (
            trajectory_loss['main_loss']
            + self.lambda_endpoint * trajectory_loss['endpoint_aux_loss']
            + self.lambda_late_horizon
            * trajectory_loss['late_horizon_aux_loss']
            + self.lambda_hist_aux * trajectory_loss['hist_aux_loss']
            + self.lambda_priority * priority_loss
            + self.lambda_style * style_loss
        )
        teacher_distillation_loss = trajectory_loss['main_loss'] * 0.0
        teacher_distillation_eligible_rate = (
            trajectory_loss['main_loss'] * 0.0
        )
        if self.lambda_teacher_distillation > 0:
            if self.teacher_distillation_mode == 'all_modes_permutation':
                teacher_distillation_loss = (
                    self._compute_teacher_trajectory_distillation(
                        data=data,
                        traj_output=trajectory_loss['traj_output'],
                        agent_mask=trajectory_loss['agent_mask'],
                        agent_batch=trajectory_loss['agent_batch'],
                    )
                )
                teacher_distillation_eligible_rate = (
                    torch.ones_like(teacher_distillation_loss)
                )
            elif (
                self.teacher_distillation_mode
                == 'advantage_gated_single_mode'
            ):
                (
                    teacher_distillation_loss,
                    teacher_distillation_eligible_rate,
                ) = self._compute_teacher_advantage_distillation(
                    data=data,
                    traj_output=trajectory_loss['traj_output'],
                    agent_mask=trajectory_loss['agent_mask'],
                    agent_batch=trajectory_loss['agent_batch'],
                    target_traj=trajectory_loss['target_traj'],
                    target_mask=trajectory_loss['target_mask'],
                    predict_mask=trajectory_loss['predict_mask'],
                )
            else:
                (
                    teacher_distillation_loss,
                    teacher_distillation_eligible_rate,
                ) = self._compute_teacher_scene_mode_distillation(
                    data=data,
                    traj_output=trajectory_loss['traj_output'],
                    agent_mask=trajectory_loss['agent_mask'],
                    agent_batch=trajectory_loss['agent_batch'],
                    target_traj=trajectory_loss['target_traj'],
                    target_mask=trajectory_loss['target_mask'],
                    predict_mask=trajectory_loss['predict_mask'],
                )
            loss = (
                loss
                + self.lambda_teacher_distillation
                * teacher_distillation_loss
            )
        if self.proposal_conflict_training_scope != 'all':
            loss = loss + self._proposal_conflict_adapter_anchor()
        self.log('train_main_loss', trajectory_loss['main_loss'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_endpoint_aux_loss', trajectory_loss['endpoint_aux_loss'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_late_horizon_aux_loss', trajectory_loss['late_horizon_aux_loss'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_hist_aux_loss', trajectory_loss['hist_aux_loss'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_priority_loss', priority_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_priority_ce_loss', priority_ce_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_priority_order_loss', priority_order_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_style_cls_loss', style_cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_style_balance_loss', style_balance_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_teacher_distillation_loss', teacher_distillation_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_teacher_distillation_eligible_rate', teacher_distillation_eligible_rate, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return loss

    def validation_step(self,data,batch_idx):
        traj_propose, traj_output_all, prob_output = self(data)
        batch_size = len(data['case_id'])
        if self.trunk_variant in ('original', 'original_residual'):
            objective = self._compute_original_trajectory_objective(
                data, traj_propose, traj_output_all, prob_output,
            )
            agent_mask = objective['agent_mask']
            agent_batch = objective['agent_batch']
            target_traj = objective['target_traj']
            target_mask = objective['target_mask']
            traj_output = objective['traj_output']
            self.log('val_reg_loss_traj_propose', objective['reg_propose'], prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_reg_loss_traj_refine', objective['reg_refine'], prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_cls_loss', objective['cls_loss'], prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_loss', objective['loss'], prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            if self.trunk_variant == 'original_residual':
                scales = self.Backbone.fusion_scales()
                for module_name, scale in scales.items():
                    self.log(f'val_baseline_{module_name}_scale', scale, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        else:
            trajectory_loss = self._compute_trajectory_objective(
                data, traj_propose, traj_output_all, prob_output,
            )
            agent_mask = trajectory_loss['agent_mask']
            agent_batch = trajectory_loss['agent_batch']
            target_traj = trajectory_loss['target_traj']
            target_mask = trajectory_loss['target_mask']
            traj_output = trajectory_loss['traj_output']
            style_cls_loss, style_balance_loss = self._compute_style_losses(data)
            style_loss = (
                style_cls_loss
                + self.style_balance_weight * style_balance_loss
            )
            priority_loss, priority_ce_loss, priority_order_loss = (
                self._compute_priority_losses(data)
            )
            loss = (
                trajectory_loss['main_loss']
                + self.lambda_endpoint * trajectory_loss['endpoint_aux_loss']
                + self.lambda_late_horizon
                * trajectory_loss['late_horizon_aux_loss']
                + self.lambda_hist_aux * trajectory_loss['hist_aux_loss']
                + self.lambda_priority * priority_loss
                + self.lambda_style * style_loss
            )
            self.log('val_main_loss', trajectory_loss['main_loss'], prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_endpoint_aux_loss', trajectory_loss['endpoint_aux_loss'], prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_late_horizon_aux_loss', trajectory_loss['late_horizon_aux_loss'], prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_hist_aux_loss', trajectory_loss['hist_aux_loss'], prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_priority_loss', priority_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_priority_ce_loss', priority_ce_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_priority_order_loss', priority_order_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_style_cls_loss', style_cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_style_balance_loss', style_balance_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        visible_mask = data['agent']['visible_mask'][agent_mask]                      #[(n1,...nb),H+F]
        visible_num = visible_mask.sum(dim=-1)                                        #[(n1,...nb)]
        scored_mask = visible_num == self.num_historical_steps + self.num_future_steps
        scored_predict_traj = unbatch(traj_output[scored_mask,-1], agent_batch[scored_mask])                   #[(n1,K,F,2),...,(nb,K,F,2)]
        scored_target_traj = unbatch(target_traj[scored_mask,-1], agent_batch[scored_mask])                    #[(n1,F,2),...,(nb,F,2)]
        scored_target_mask = unbatch(target_mask[scored_mask,-1], agent_batch[scored_mask])                    #[(n1,F),...,(nb,F)]

        self.min_joint_ade.update(scored_predict_traj, scored_target_traj, scored_target_mask)
        self.min_joint_fde.update(scored_predict_traj, scored_target_traj, scored_target_mask)
        self.log('val_minJointADE', self.min_joint_ade, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val_minJointFDE', self.min_joint_fde, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.min_joint_mr.update(scored_predict_traj, scored_target_traj, scored_target_mask)
        self.log('val_minJointMR', self.min_joint_mr, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        # save predictions for visualization
        if self.save_val_predictions:
            scored_agent_ids = data['agent']['id'][agent_mask][scored_mask]
            scored_agent_ids = unbatch(scored_agent_ids, agent_batch[scored_mask])
            case_ids = data['case_id'].cpu().numpy()
            scenario_names = data['scenario_name']
            future_frame_ids = list(range(self.num_future_steps))
            future_frame_ids = [fid + self.num_historical_steps + 1 for fid in future_frame_ids]
            future_timestamps = [fid * 100 for fid in future_frame_ids]

            for i in range(batch_size):
                pred_i = scored_predict_traj[i].cpu().numpy()    # [ni, K, F, 2]
                gt_i   = scored_target_traj[i].cpu().numpy()     # [ni, F, 2]
                aids_i = scored_agent_ids[i].cpu().numpy()        # [ni]
                key = scenario_names[i]
                rows = []
                for j in range(len(aids_i)):
                    for f_idx in range(self.num_future_steps):
                        row = [int(case_ids[i]), int(aids_i[j]),
                               future_frame_ids[f_idx], future_timestamps[f_idx],
                               gt_i[j, f_idx, 0], gt_i[j, f_idx, 1]]
                        for k in range(6):
                            row += [pred_i[j, k, f_idx, 0], pred_i[j, k, f_idx, 1]]
                        rows.append(row)
                if key in self.val_output:
                    self.val_output[key].extend(rows)
                else:
                    self.val_output[key] = rows

    def on_validation_end(self):
        if not self.save_val_predictions or not self.val_output:
            return
        import os
        os.makedirs('./val_output', exist_ok=True)
        for key, rows in self.val_output.items():
            df = pd.DataFrame(rows, columns=self._val_columns)
            df.to_csv(f'./val_output/{key}_val_pred.csv', index=False)
        self.val_output.clear()

    def test_step(self,data,batch_idx):
        traj_propose, traj_output, prob_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K]

        agent_mask = data['agent']['category'] == 1
        agent_batch = data['agent']['batch'][agent_mask]
        batch_size = len(data['case_id'])
        num_agent_pre_batch = torch.bincount(agent_batch)

        scenario_name = data['scenario_name']   #[b]
        case_id = data['case_id']               #[b]
        agent_id = data['agent']['id'][agent_mask]
        agent_interset = data['agent']['interest'][agent_mask]
        traj_output = traj_output[agent_mask, -1]
        tep = torch.cat([data['agent']['position'][agent_mask, -1:].unsqueeze(1).repeat_interleave(self.num_modes,1), traj_output], dim=-2) #[(n1+...+nb),K,F+1,2]
        _, yaw_output = compute_angles_lengths_2D(tep[:,:,1:] - tep[:,:,:-1])   #[(n1+...+nb),K,F]

        scored_agent_id = unbatch(agent_id, agent_batch)                        #[n1,...nb]
        scored_agent_interset = unbatch(agent_interset, agent_batch)            #[n1,...nb]
        scored_predict_traj = unbatch(traj_output, agent_batch)           #[(n1,K,F,2),...,(nb,K,F,2)]
        scored_predict_yaw = unbatch(yaw_output, agent_batch)             #[(n1,K,F),...,(nb,K,F)]
        
        case_id = case_id.cpu().numpy()
        scored_agent_id = [agent_id.cpu().numpy() for agent_id in scored_agent_id]
        scored_agent_interset = [agent_interset.cpu().numpy() for agent_interset in scored_agent_interset]
        scored_predict_traj = [predict_traj.cpu().numpy() for predict_traj in scored_predict_traj]
        scored_predict_yaw = [predict_yaw.cpu().numpy() for predict_yaw in scored_predict_yaw]
        
        scored_frame_id = list(range(30))
        scored_frame_id = [id + 11 for id in scored_frame_id]
        scored_timestamp_ms = [frame_id * 100 for frame_id in scored_frame_id]

        for i in range(batch_size):
            rows = []
            for j in range(num_agent_pre_batch[i]):
                for k in range(self.num_future_steps):
                    row = [case_id[i], scored_agent_id[i][j], scored_frame_id[k], scored_timestamp_ms[k], scored_agent_interset[i][j],
                        scored_predict_traj[i][j,0,k,0], scored_predict_traj[i][j,0,k,1], scored_predict_yaw[i][j,0,k],
                        scored_predict_traj[i][j,1,k,0], scored_predict_traj[i][j,1,k,1], scored_predict_yaw[i][j,1,k],
                        scored_predict_traj[i][j,2,k,0], scored_predict_traj[i][j,2,k,1], scored_predict_yaw[i][j,2,k],
                        scored_predict_traj[i][j,3,k,0], scored_predict_traj[i][j,3,k,1], scored_predict_yaw[i][j,3,k],
                        scored_predict_traj[i][j,4,k,0], scored_predict_traj[i][j,4,k,1], scored_predict_yaw[i][j,4,k],
                        scored_predict_traj[i][j,5,k,0], scored_predict_traj[i][j,5,k,1], scored_predict_yaw[i][j,5,k]]
                    rows.append(row)

            if scenario_name[i] in self.test_output:
                self.test_output[scenario_name[i]] = self.test_output[scenario_name[i]] + rows
            else:
                self.test_output[scenario_name[i]] = rows

    def on_test_end(self):
        for key, value in self.test_output.items():
            df = pd.DataFrame(value, columns=self._columns)
            df['track_to_predict'] = 1
            df.to_csv('./test_output/' + key + '_sub.csv', index=False)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if param_name.endswith('bias'):
                    no_decay.add(full_param_name)
                elif param_name.endswith('weight'):
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                else:
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param
            for param_name, param in self.named_parameters()
            if param.requires_grad
        }
        decay.intersection_update(param_dict)
        no_decay.intersection_update(param_dict)
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        provider_prefix = 'Backbone.residual.'
        use_provider_lr = (
            self.trunk_variant == 'original_residual'
            and self.baseline_residual_training_scope == 'residual_branch'
        )
        provider_names = {
            param_name
            for param_name in param_dict
            if use_provider_lr and param_name.startswith(provider_prefix)
        }
        fusion_names = set(param_dict) - provider_names

        optim_groups = []
        for names, learning_rate in (
            (fusion_names, self.lr),
            (
                provider_names,
                self.lr * self.baseline_residual_provider_lr_scale,
            ),
        ):
            decay_names = decay & names
            no_decay_names = no_decay & names
            if decay_names:
                optim_groups.append({
                    "params": [
                        param_dict[param_name]
                        for param_name in sorted(decay_names)
                    ],
                    "weight_decay": self.weight_decay,
                    "lr": learning_rate,
                })
            if no_decay_names:
                optim_groups.append({
                    "params": [
                        param_dict[param_name]
                        for param_name in sorted(no_decay_names)
                    ],
                    "weight_decay": 0.0,
                    "lr": learning_rate,
                })
        if not optim_groups:
            raise RuntimeError('model has no trainable parameters')

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        
        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('PNet')
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_historical_steps', type=int, default=10)
        parser.add_argument('--num_future_steps', type=int, default=30)
        parser.add_argument('--duration', type=int, default=10)
        parser.add_argument('--a2a_radius', type=float, default=80)
        parser.add_argument('--l2a_radius', type=float, default=80)
        parser.add_argument('--interaction_distance_margin', type=float, default=5.0)
        parser.add_argument('--observation_step_time', type=float, default=0.1)
        parser.add_argument('--lane_arrival_time_threshold', type=float, default=1.0)
        parser.add_argument('--ttc_speed_epsilon', type=float, default=1e-3)
        parser.add_argument('--interaction_ttc_threshold', type=float, default=3.0)
        parser.add_argument('--interaction_collision_distance', type=float, default=3.0)
        parser.add_argument('--conflict_lane_radius', type=float, default=10.0)
        parser.add_argument('--traffic_base_radius', type=float, default=10.0)
        parser.add_argument('--traffic_speed_scale', type=float, default=1.0)
        parser.add_argument('--traffic_backward_attention_ratio', type=float, default=0.5)
        parser.add_argument('--traffic_max_attention_nodes', type=int, default=5)
        parser.add_argument('--traffic_current_radius', type=float, default=8.0)
        parser.add_argument('--traffic_lane_lateral_radius', type=float, default=4.0)
        parser.add_argument('--traffic_topology_base_distance', type=float, default=5.0)
        parser.add_argument('--traffic_lane_change_lateral_radius', type=float, default=1.0)
        parser.add_argument('--traffic_lane_change_lateral_speed', type=float, default=0.3)
        parser.add_argument('--style_thw_max', type=float, default=5.0)
        parser.add_argument('--style_ttc_max', type=float, default=10.0)
        parser.add_argument('--style_arrival_time_fill', type=float, default=3.0)
        parser.add_argument('--style_transformer_layers', type=int, default=2)
        parser.add_argument('--style_gap_reset_steps', type=int, default=3)
        parser.add_argument('--style_pool_min_length', type=int, default=3)
        parser.add_argument('--style_min_visible_steps', type=int, default=5)
        parser.add_argument('--style_min_last_segment_steps', type=int, default=5)
        parser.add_argument('--style_statistics_epsilon', type=float, default=None)
        parser.add_argument('--style_statistics_path', type=str, default=None)
        parser.add_argument('--lambda_hist_aux', type=float, default=0.3)
        parser.add_argument('--lambda_priority', type=float, default=0.3)
        parser.add_argument('--lambda_style', type=float, default=0.05)
        parser.add_argument('--lambda_endpoint', type=float, default=0.0)
        parser.add_argument('--lambda_late_horizon', type=float, default=0.0)
        parser.add_argument(
            '--lambda_teacher_distillation',
            type=float,
            default=0.0,
            help=(
                'Weight for scene-balanced, permutation-invariant final '
                'trajectory distillation from a cached teacher.'
            ),
        )
        parser.add_argument(
            '--teacher_distillation_mode',
            choices=(
                'all_modes_permutation',
                'advantage_gated_single_mode',
                'advantage_gated_scene_mode',
            ),
            default='all_modes_permutation',
            help=(
                'Cached-teacher objective. The default restores version_28; '
                'advantage_gated_single_mode only transfers a teacher mode '
                'that improves per-agent endpoint oracle error; '
                'advantage_gated_scene_mode transfers one shared joint mode '
                'only when the teacher improves scene endpoint oracle error.'
            ),
        )
        parser.add_argument(
            '--teacher_advantage_margin',
            type=float,
            default=0.05,
            help=(
                'Minimum teacher endpoint advantage in metres required by '
                'the selected advantage-gated distillation mode.'
            ),
        )
        parser.add_argument('--late_horizon_steps', type=int, default=7)
        parser.add_argument(
            '--late_horizon_reduction',
            choices=('agent_mean', 'scene_mean'),
            default='agent_mean',
        )
        parser.add_argument('--priority_order_weight', type=float, default=0.2)
        parser.add_argument('--style_balance_weight', type=float, default=0.05)
        parser.add_argument(
            '--motion_source',
            choices=('causal_difference', 'csv'),
            default='csv',
            help='New training defaults to original INTERACTION CSV motion.',
        )
        parser.add_argument(
            '--csv_motion_cache_dir',
            type=str,
            default=None,
            help='Defaults to <root>/csv_motion_v1.',
        )
        parser.add_argument(
            '--preserve_base_edges',
            action='store_true',
            default=True,
            help='Use original radius-based vehicle and lane edges as the trunk.',
        )
        parser.add_argument(
            '--no_preserve_base_edges',
            action='store_false',
            dest='preserve_base_edges',
        )
        parser.add_argument(
            '--reachability_residual_scale',
            type=float,
            default=0.0,
            help='Scale for dynamic Reachability lane residual on top of base edges.',
        )
        parser.add_argument(
            '--trunk_variant',
            choices=('extended', 'original', 'original_residual'),
            default='extended',
            help=(
                "'original' restores the version-22 dense graph, 7/3/5-D "
                'features, and original trajectory objective.'
            ),
        )
        parser.add_argument(
            '--baseline_residual_max_scale',
            type=float,
            default=0.1,
            help=(
                'Maximum Style/Reachability/Priority representation-residual '
                'scale used by original_residual.'
            ),
        )
        parser.add_argument(
            '--baseline_residual_training_scope',
            choices=('fusion_only', 'residual_branch'),
            default='fusion_only',
            help=(
                "'fusion_only' freezes both pretrained branches and trains "
                'the selected module gates and adapters; residual_branch '
                'also fine-tunes the extended feature provider while keeping '
                'v22 frozen.'
            ),
        )
        parser.add_argument(
            '--baseline_residual_provider_lr_scale',
            type=float,
            default=0.1,
            help=(
                'Learning-rate multiplier for the extended feature provider '
                "when baseline_residual_training_scope='residual_branch'."
            ),
        )
        parser.add_argument(
            '--baseline_residual_modules',
            choices=(
                'all', 'style', 'reachability', 'priority',
                'style_reachability', 'style_priority',
                'reachability_priority',
            ),
            default='all',
            help=(
                'Select which original_residual gate/adapters can modify '
                "the frozen v22 representation. 'all' preserves existing "
                'checkpoint semantics; single-module and explicit two-module '
                'values are for retrained causal ablations.'
            ),
        )
        parser.add_argument(
            '--style_film_mode',
            choices=('legacy', 'bounded_identity'),
            default='legacy',
            help=(
                "'bounded_identity' starts Style FiLM as an exact identity "
                'and bounds both modulation terms with tanh.'
            ),
        )
        parser.add_argument(
            '--style_film_bound',
            type=float,
            default=0.1,
            help='Absolute bound for Style FiLM gamma and beta.',
        )
        parser.add_argument(
            '--conflict_update_scope',
            choices=('legacy', 'participants_only'),
            default='legacy',
            help=(
                "'participants_only' updates only agent-mode nodes that are "
                'endpoints of E_VV_conf.'
            ),
        )
        parser.add_argument(
            '--priority_injection_mode',
            choices=('legacy', 'auxiliary_only'),
            default='legacy',
            help=(
                "'auxiliary_only' trains Priority supervision but replaces "
                'its trajectory probability and head bias with zeros.'
            ),
        )
        parser.add_argument(
            '--proposal_conflict_mode_scale',
            type=float,
            default=0.0,
            help=(
                'Bounded mode-token modulation strength for proposal-stage '
                'conflict edge embeddings; 0 preserves legacy behavior.'
            ),
        )
        parser.add_argument(
            '--proposal_conflict_mode_adapter_scale',
            type=float,
            default=0.0,
            help=(
                'Bounded modulation strength for the zero-initialized, '
                'conflict-specific proposal mode tokens.'
            ),
        )
        parser.add_argument(
            '--proposal_conflict_residual_gate_scale',
            type=float,
            default=0.0,
            help=(
                'Bounded strength for the zero-initialized per-mode gate on '
                'proposal conflict-attention participant residuals.'
            ),
        )
        parser.add_argument(
            '--scene_mode_coordination_scale',
            type=float,
            default=0.0,
            help=(
                'Bounded strength for the zero-initialized scene-level '
                'same-mode coordination residual before proposal decoding.'
            ),
        )
        parser.add_argument(
            '--joint_mode_composition_scale',
            type=float,
            default=0.0,
            help=(
                'Strength of the hard scene-level joint-mode assignment; '
                '0 preserves the existing output ordering.'
            ),
        )
        parser.add_argument(
            '--joint_mode_assignment_mode',
            choices=(
                'independent', 'permutation', 'limited_permutation',
            ),
            default='independent',
            help=(
                "'independent' selects each joint slot separately; "
                "'permutation' enforces one-to-one local modes; "
                "'limited_permutation' additionally caps per-scene agent "
                'exceptions.'
            ),
        )
        parser.add_argument(
            '--joint_mode_max_exception_agents',
            type=int,
            default=2,
            help=(
                'Maximum predicted agents per scene and historical anchor '
                'allowed to leave identity in limited_permutation mode.'
            ),
        )
        parser.add_argument(
            '--proposal_conflict_training_scope',
            choices=(
                'all', 'adapter_only', 'residual_gate_only',
                'scene_coordination_only', 'joint_composition_only',
            ),
            default='all',
            help=(
                "'adapter_only' freezes the base model and trains only the "
                "conflict-edge mode tokens; 'residual_gate_only' trains only "
                "the proposal conflict residual-gate tokens; "
                "'scene_coordination_only' trains only the scene-level "
                "same-mode residual projection; 'joint_composition_only' "
                'trains only the scene-level hard mode composer.'
            ),
        )
        parser.add_argument('--num_visible_steps', type=int, default=3)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_attn_layers', type=int, default=3)
        parser.add_argument('--num_hops', type=int, default=4)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--warmup_epochs', type=int, default=4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
