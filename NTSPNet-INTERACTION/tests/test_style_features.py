import types

import torch
import torch.nn as nn

from modules.backbone import Backbone


def make_backbone(num_historical_steps=1, **kwargs):
    return Backbone(
        hidden_dim=8,
        num_historical_steps=num_historical_steps,
        num_future_steps=4,
        duration=1,
        a2a_radius=50.0,
        l2a_radius=50.0,
        num_attn_layers=1,
        num_modes=2,
        num_heads=2,
        dropout=0.0,
        interaction_distance_margin=25.0,
        **kwargs,
    )


def test_bounded_style_film_starts_as_exact_identity():
    backbone = make_backbone(
        num_historical_steps=3,
        style_film_mode='bounded_identity',
    )
    node_embedding = torch.randn((2 * 3 * backbone.num_modes, 8))
    style_prefix_embedding = torch.randn((2, 3, 8))

    for film_generator in (
        backbone.proposal_style_film,
        backbone.refinement_style_film,
    ):
        output_layer = film_generator.mlp[-1]
        output = backbone._apply_style_film(
            node_embedding,
            style_prefix_embedding,
            film_generator,
        )

        assert torch.equal(output_layer.weight, torch.zeros_like(output_layer.weight))
        assert torch.equal(output_layer.bias, torch.zeros_like(output_layer.bias))
        assert torch.equal(output, node_embedding)


def test_bounded_style_film_output_layer_receives_gradient_at_identity():
    backbone = make_backbone(
        num_historical_steps=3,
        style_film_mode='bounded_identity',
    )
    node_embedding = torch.randn((2 * 3 * backbone.num_modes, 8))
    style_prefix_embedding = torch.randn((2, 3, 8))
    output = backbone._apply_style_film(
        node_embedding,
        style_prefix_embedding,
        backbone.proposal_style_film,
    )

    output.square().mean().backward()

    gradient = backbone.proposal_style_film.mlp[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.any(gradient != 0)


def test_bounded_style_film_limits_scale_and_bias_modulation():
    backbone = make_backbone(
        style_film_mode='bounded_identity',
        style_film_bound=0.1,
    )
    output_layer = backbone.proposal_style_film.mlp[-1]
    with torch.no_grad():
        output_layer.bias[:backbone.hidden_dim].fill_(100.0)
        output_layer.bias[backbone.hidden_dim:].fill_(100.0)

    output = backbone._apply_style_film(
        torch.ones((backbone.num_modes, 8)),
        torch.zeros((1, 1, 8)),
        backbone.proposal_style_film,
    )

    assert torch.all(output <= 1.2)
    assert torch.all(output >= 1.2 - 1e-6)


def test_legacy_style_film_keeps_unbounded_formula():
    backbone = make_backbone(style_film_mode='legacy')
    film_parameter = torch.cat((torch.full((1, 1, 8), 2.0),
                                torch.full((1, 1, 8), 3.0)), dim=-1)

    class ConstantFilm(nn.Module):
        def forward(self, _):
            return film_parameter

    output = backbone._apply_style_film(
        torch.ones((backbone.num_modes, 8)),
        torch.zeros((1, 1, 8)),
        ConstantFilm(),
    )

    assert torch.equal(output, torch.full_like(output, 6.0))


def test_style_feature_has_documented_13d_order_and_front_vehicle_aggregation():
    backbone = make_backbone()
    position = torch.tensor([[[0.0, 0.0]], [[20.0, 0.0]]])
    velocity = torch.tensor([[[10.0, 0.0]], [[5.0, 0.0]]])
    visible = torch.ones((2, 1), dtype=torch.bool)
    reference_lane = torch.zeros((2, 1), dtype=torch.long)
    graph = backbone.vehicle_interaction_graph(
        position=position,
        velocity=velocity,
        visible_mask=visible,
        batch=torch.zeros(2, dtype=torch.long),
        lane_position=torch.tensor([[10.0, 0.0]]),
        lane_batch=torch.zeros(1, dtype=torch.long),
        conflict_edge_index=torch.empty((2, 0), dtype=torch.long),
        lane_heading=torch.zeros(1),
        reference_lane=reference_lane,
    )

    style_feature = backbone._build_vehicle_style_feature(
        interaction_graph=graph,
        speed=torch.tensor([[10.0], [5.0]]),
        acceleration=torch.tensor([[[3.0, 4.0]], [[0.0, 0.0]]]),
        jerk=torch.tensor([[[0.0, 6.0]], [[0.0, 0.0]]]),
        heading=torch.tensor([[0.1], [0.2]]),
        heading_change=torch.tensor([[0.01], [0.02]]),
        visible_mask=visible,
        lane_lateral_distance=torch.tensor([[1.0], [2.0]]),
    )

    assert style_feature.shape == (2, 1, 13)
    assert backbone.STYLE_FEATURE_NAMES[7:9] == (
        'degree_centrality_change',
        'closeness_centrality_change',
    )
    assert torch.allclose(style_feature[0, 0, :5], torch.tensor([10.0, 5.0, 6.0, 0.1, 0.01]))
    assert torch.allclose(style_feature[0, 0, 9:], torch.tensor([2.0, 4.0, 1.0, 1.0]), atol=1e-3)
    assert torch.allclose(style_feature[1, 0, 9:], torch.tensor([5.0, 10.0, 2.0, 1.0]), atol=1e-3)


def test_invisible_style_timestep_is_zeroed_and_mask_is_kept_separately():
    backbone = make_backbone()
    position = torch.tensor([[[0.0, 0.0]]])
    velocity = torch.zeros_like(position)
    graph = backbone.vehicle_interaction_graph(
        position=position,
        velocity=velocity,
        visible_mask=torch.zeros((1, 1), dtype=torch.bool),
        batch=torch.zeros(1, dtype=torch.long),
        lane_position=torch.tensor([[0.0, 0.0]]),
        lane_batch=torch.zeros(1, dtype=torch.long),
        conflict_edge_index=torch.empty((2, 0), dtype=torch.long),
        lane_heading=torch.zeros(1),
        reference_lane=torch.full((1, 1), -1, dtype=torch.long),
    )

    style_feature = backbone._build_vehicle_style_feature(
        interaction_graph=graph,
        speed=torch.zeros((1, 1)),
        acceleration=torch.zeros((1, 1, 2)),
        jerk=torch.zeros((1, 1, 2)),
        heading=torch.zeros((1, 1)),
        heading_change=torch.zeros((1, 1)),
        visible_mask=torch.zeros((1, 1), dtype=torch.bool),
        lane_lateral_distance=torch.zeros((1, 1)),
    )

    assert torch.equal(style_feature, torch.zeros((1, 1, 13)))


def test_style_attention_is_causal_and_resets_only_after_three_invisible_steps():
    backbone = make_backbone(num_historical_steps=5)
    short_gap_mask = backbone._build_style_temporal_attention_mask(
        torch.tensor([[True, False, False, True, True]])
    )[0]
    long_gap_mask = backbone._build_style_temporal_attention_mask(
        torch.tensor([[True, False, False, False, True]])
    )[0]

    assert short_gap_mask[3, 0] == 0
    assert long_gap_mask[4, 0] == 1
    assert short_gap_mask[0, 4] == 1
    assert (~short_gap_mask).any(dim=-1).all()
    assert (~long_gap_mask).any(dim=-1).all()


def test_style_transformer_handles_leading_invisible_prefix_without_nan():
    backbone = make_backbone(num_historical_steps=10).eval()
    visible = torch.tensor([[False, False, False] + [True] * 7])
    style_feature = torch.randn((1, 10, 13))

    with torch.no_grad():
        temporal = backbone._encode_vehicle_style_sequence(
            style_feature,
            visible,
        )
        prefix = backbone._pool_vehicle_style_prefix_embeddings(
            temporal,
            visible,
        )

    assert torch.isfinite(temporal).all()
    assert torch.isfinite(prefix).all()
    assert torch.equal(temporal[:, :3], torch.zeros((1, 3, 8)))
    assert torch.equal(prefix[:, :3], torch.zeros((1, 3, 8)))
    assert torch.any(prefix[:, 3:] != 0)


def test_style_transformer_retains_timestep_outputs_without_future_leakage():
    torch.manual_seed(7)
    backbone = make_backbone(num_historical_steps=4).eval()
    visible = torch.ones((1, 4), dtype=torch.bool)
    base_style = torch.randn((1, 4, 13))
    changed_future_style = base_style.clone()
    changed_future_style[:, 3] = changed_future_style[:, 3] + 100.0

    with torch.no_grad():
        base_output = backbone._encode_vehicle_style_sequence(base_style, visible)
        changed_output = backbone._encode_vehicle_style_sequence(
            changed_future_style,
            visible,
        )

    assert base_output.shape == (1, 4, 8)
    assert torch.allclose(base_output[:, :3], changed_output[:, :3], atol=1e-6)


def test_style_transformer_all_invisible_sequence_is_finite_and_zero():
    backbone = make_backbone(num_historical_steps=4).eval()
    visible = torch.zeros((2, 4), dtype=torch.bool)

    with torch.no_grad():
        output = backbone._encode_vehicle_style_sequence(
            torch.zeros((2, 4, 13)),
            visible,
        )

    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros((2, 4, 8)))


def test_style_pool_blends_short_last_segment_with_global_history():
    backbone = make_backbone(num_historical_steps=7).eval()
    backbone.style_embedding_mlp = nn.Identity()

    def masked_mean_pool(self, temporal_feature, valid_mask, return_attention_weights):
        count = valid_mask.sum(dim=1).clamp_min(1).to(temporal_feature.dtype)
        weight = valid_mask.to(temporal_feature.dtype) / count.unsqueeze(-1)
        pooled = (temporal_feature * weight.unsqueeze(-1)).sum(dim=1)
        pooled = torch.where(
            valid_mask.any(dim=1).unsqueeze(-1),
            pooled,
            torch.zeros_like(pooled),
        )
        return pooled, weight if return_attention_weights else None

    backbone._attn_pool_style_segment = types.MethodType(masked_mean_pool, backbone)
    temporal_feature = torch.zeros((1, 7, 8))
    temporal_feature[:, :3] = 2.0
    temporal_feature[:, 6] = 10.0
    visible = torch.tensor([[True, True, True, False, False, False, True]])

    embedding, attention_weight = backbone._pool_vehicle_style_embedding(
        temporal_feature,
        visible,
        return_attention_weights=True,
    )

    assert torch.allclose(embedding, torch.full((1, 8), 6.0))
    assert torch.allclose(
        attention_weight,
        torch.tensor([[1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.5]]),
    )


def test_style_attn_pool_returns_optional_fused_attention_weights():
    backbone = make_backbone(num_historical_steps=5).eval()
    temporal_feature = torch.randn((2, 5, 8))
    visible = torch.tensor([
        [True, True, False, True, True],
        [False, False, False, False, False],
    ])

    with torch.no_grad():
        embedding, attention_weight = backbone._pool_vehicle_style_embedding(
            temporal_feature,
            visible,
            return_attention_weights=True,
        )
        _, default_weight = backbone._pool_vehicle_style_embedding(
            temporal_feature,
            visible,
            return_attention_weights=False,
        )

    assert embedding.shape == (2, 8)
    assert attention_weight.shape == (2, 5)
    assert torch.all(attention_weight[0, ~visible[0]] == 0)
    assert torch.allclose(attention_weight[0].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.equal(embedding[1], torch.zeros(8))
    assert torch.equal(attention_weight[1], torch.zeros(5))
    assert default_weight is None


def test_style_prefix_embeddings_never_pool_future_temporal_features():
    torch.manual_seed(11)
    backbone = make_backbone(num_historical_steps=4).eval()
    temporal = torch.randn((1, 4, 8))
    changed = temporal.clone()
    changed[:, 3] = changed[:, 3] + 100.0
    visible = torch.ones((1, 4), dtype=torch.bool)

    with torch.no_grad():
        base_prefix = backbone._pool_vehicle_style_prefix_embeddings(
            temporal,
            visible,
        )
        changed_prefix = backbone._pool_vehicle_style_prefix_embeddings(
            changed,
            visible,
        )

    assert base_prefix.shape == (1, 4, 8)
    assert torch.allclose(base_prefix[:, :3], changed_prefix[:, :3], atol=1e-6)


def test_style_score_indicators_follow_the_documented_temporal_reductions():
    backbone = make_backbone(num_historical_steps=10)
    style_feature = torch.zeros((1, 10, 13))
    style_feature[0, :, 0] = torch.arange(1.0, 11.0)
    style_feature[0, :, 1] = torch.arange(2.0, 22.0, 2.0)
    style_feature[0, :, 2] = 3.0
    style_feature[0, :, 4] = torch.arange(10.0) * 0.1
    style_feature[0, :, 10] = torch.arange(10.0)
    style_feature[0, :, 11] = torch.arange(10.0) * 0.2
    visible = torch.ones((1, 10), dtype=torch.bool)
    valid = torch.ones_like(visible)

    indicators, reliable = backbone._build_vehicle_style_indicators(
        style_feature=style_feature,
        visible_mask=visible,
        agent_type=torch.tensor([0]),
        velocity_valid_mask=valid,
        acceleration_valid_mask=valid,
        jerk_valid_mask=valid,
        heading_valid_mask=valid,
    )

    expected_ttc_risk = (1.0 - torch.arange(10.0) / 10.0).mean()
    assert torch.allclose(indicators[0, 0], torch.quantile(torch.arange(1.0, 11.0), 0.9))
    assert torch.allclose(indicators[0, 1], torch.quantile(torch.arange(2.0, 22.0, 2.0), 0.9))
    assert torch.allclose(indicators[0, 2], torch.tensor(3.0))
    assert torch.allclose(indicators[0, 3], torch.tensor(0.5))
    assert torch.allclose(indicators[0, 4], torch.tensor(2.0))
    assert torch.allclose(indicators[0, 5], expected_ttc_risk)
    assert reliable.item()


def test_masked_quantile_matches_per_agent_torch_quantile():
    torch.manual_seed(17)
    value = torch.randn((32, 10))
    valid_mask = torch.rand((32, 10)) > 0.35
    valid_mask[0] = False
    expected = torch.zeros(32)
    for agent_idx in range(value.size(0)):
        agent_value = value[agent_idx, valid_mask[agent_idx]]
        if agent_value.numel() != 0:
            expected[agent_idx] = torch.quantile(agent_value, 0.9)

    actual = Backbone._masked_quantile(value, valid_mask, 0.9)

    assert torch.equal(actual, expected)


def test_style_reliability_requires_car_and_five_visible_prefix_frames():
    backbone = make_backbone(num_historical_steps=10)
    style_feature = torch.zeros((5, 10, 13))
    visible = torch.tensor([
        [True] * 10,
        [False, False, False] + [True] * 7,
        [True] * 9 + [False],
        [True] * 5 + [False] + [True] * 4,
        [True] * 10,
    ])
    valid = visible.clone()

    _, reliable = backbone._build_vehicle_style_indicators(
        style_feature=style_feature,
        visible_mask=visible,
        agent_type=torch.tensor([0, 0, 0, 0, 1]),
        velocity_valid_mask=valid,
        acceleration_valid_mask=valid,
        jerk_valid_mask=valid,
        heading_valid_mask=valid,
    )

    assert torch.equal(
        reliable,
        torch.tensor([True, True, True, True, False]),
    )


def test_style_prefix_supervision_starts_at_five_visible_frames():
    backbone = make_backbone(num_historical_steps=7)
    style_feature = torch.zeros((3, 7, 13))
    visible = torch.tensor([
        [False, False, False, False, False, False, False],
        [True, True, True, True, False, False, False],
        [False, True, True, True, True, True, False],
    ])
    valid = visible.clone()

    indicators, reliable = backbone._build_vehicle_style_prefix_indicators(
        style_feature=style_feature,
        visible_mask=visible,
        agent_type=torch.tensor([0, 0, 0]),
        velocity_valid_mask=valid,
        acceleration_valid_mask=valid,
        jerk_valid_mask=valid,
        heading_valid_mask=valid,
    )

    assert indicators.shape == (3, 7, 6)
    assert not reliable[0].any()
    assert not reliable[1].any()
    assert torch.equal(
        reliable[2],
        torch.tensor([False, False, False, False, False, True, True]),
    )


def test_style_pseudo_label_uses_latest_class_order_and_inclusive_quantiles():
    backbone = make_backbone()
    backbone.style_indicator_median.zero_()
    backbone.style_indicator_iqr.fill_(0.999)
    backbone.style_score_q33.fill_(0.0)
    backbone.style_score_q66.fill_(1.0)
    backbone.style_statistics_ready.fill_(True)
    indicators = torch.zeros((4, 6))
    indicators[:, 0] = torch.tensor([0.0, 2.0, 4.0, 8.0])
    reliable = torch.tensor([True, True, True, False])

    standardized, score, label = backbone._build_vehicle_style_pseudo_label(
        indicators,
        reliable,
    )

    assert torch.allclose(standardized, indicators)
    assert score[0] == 0.0
    assert torch.equal(label, torch.tensor([2, 1, 0, -1]))
    assert backbone.STYLE_CLASS_NAMES == ('aggressive', 'normal', 'conservative')


def test_style_classifier_returns_three_normalized_probabilities():
    backbone = make_backbone()
    style_embedding = torch.randn((4, 8))

    logits = backbone.style_classifier(style_embedding)
    probability = backbone.style_probability_norm(logits)

    assert logits.shape == (4, 3)
    assert probability.shape == (4, 3)
    assert torch.allclose(probability.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_style_statistics_file_is_loaded_with_matching_configuration(tmp_path):
    backbone = make_backbone(num_historical_steps=10)
    statistics_path = tmp_path / 'style_statistics.pt'
    torch.save({
        'indicator_names': list(backbone.STYLE_SCORE_INDICATOR_NAMES),
        'score_weights': backbone.style_score_weights.clone(),
        'median': torch.arange(6.0),
        'iqr': torch.arange(1.0, 7.0),
        'q33': torch.tensor(-0.5),
        'q66': torch.tensor(0.5),
        'epsilon': backbone.style_statistics_epsilon,
        'configuration': {
            'num_historical_steps': backbone.num_historical_steps,
            'num_future_steps': backbone.num_future_steps,
            'observation_step_time': backbone.observation_step_time,
            'a2a_radius': backbone.a2a_radius,
            'interaction_distance_margin': backbone.interaction_distance_margin,
            'lane_arrival_time_threshold': backbone.lane_arrival_time_threshold,
            'ttc_speed_epsilon': backbone.ttc_speed_epsilon,
            'interaction_ttc_threshold': backbone.interaction_ttc_threshold,
            'interaction_collision_distance': backbone.interaction_collision_distance,
            'traffic_current_radius': backbone.traffic_current_radius,
            'traffic_lane_lateral_radius': backbone.traffic_lane_lateral_radius,
            'traffic_topology_base_distance': backbone.traffic_topology_base_distance,
            'style_ttc_max': backbone.style_ttc_max,
            'style_min_visible_steps': backbone.style_min_visible_steps,
            'style_min_last_segment_steps': backbone.style_min_last_segment_steps,
        },
    }, statistics_path)

    backbone.load_style_statistics(str(statistics_path))

    assert backbone.style_statistics_ready.item()
    assert torch.equal(backbone.style_indicator_median, torch.arange(6.0))
    assert torch.equal(backbone.style_indicator_iqr, torch.arange(1.0, 7.0))
    assert backbone.style_score_q33 == -0.5
    assert backbone.style_score_q66 == 0.5
