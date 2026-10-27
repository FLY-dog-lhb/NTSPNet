import pytest
import torch

from cross_domain.reporting import METRIC_KEYS, SceneAccumulator, marginal_actor_metrics
from cross_domain.runtime import joint_metrics


def test_different_actor_modes_and_mr_threshold():
    target = torch.zeros(2, 3, 2)
    pred = torch.zeros(2, 2, 3, 2)
    pred[0, 1, :, 0] = 6.
    pred[1, 0, :, 0] = 6.
    marginal = marginal_actor_metrics(pred, target)
    assert all(value.tolist() == [0., 0.] for value in marginal.values())
    joint = joint_metrics(pred, target)
    assert joint == dict(minJointADE=3., minJointFDE=3., minJointMR=1.)
    pred[:, :, :, 0] = 2.
    assert marginal_actor_metrics(pred, target)['marginal_actorMR'].tolist() == [0., 0.]
    pred[0, :, :, 0] += .01
    assert marginal_actor_metrics(pred, target)['marginal_actorMR'].tolist() == [1., 0.]


def test_single_actor_equivalence_and_independent_ade_fde_oracles():
    target = torch.zeros(1, 30, 2)
    pred = torch.zeros(1, 2, 30, 2)
    pred[0, 0, -1, 0] = 4.
    pred[0, 1, :, 0] = 3.
    pred[0, 1, -1, 0] = 1.
    marginal = marginal_actor_metrics(pred, target)
    assert marginal['marginal_minADE'].item() == pytest.approx(4 / 30)
    assert marginal['marginal_minFDE'].item() == 1.
    joint = joint_metrics(pred, target)
    for a, b in [('marginal_minADE', 'minJointADE'), ('marginal_minFDE', 'minJointFDE'),
                 ('marginal_actorMR', 'minJointMR')]:
        assert marginal[a].item() == joint[b]


def test_marginal_bound_under_per_actor_mode_permutation():
    torch.manual_seed(4)
    pred = torch.randn(4, 6, 30, 2)
    target = torch.randn(4, 30, 2)
    permuted = pred.clone()
    for actor in range(4):
        permuted[actor] = pred[actor, torch.randperm(6)]
    before, after = (marginal_actor_metrics(p, target) for p in (pred, permuted))
    for key in before:
        assert torch.equal(before[key], after[key])
    for p in (pred, permuted):
        joint = joint_metrics(p, target)
        for suffix in ('ADE', 'FDE'):
            assert before['marginal_min' + suffix].mean() <= joint['minJoint' + suffix] + 1e-6


def test_scene_and_actor_weighting_focal_denominator_and_empty_subset():
    acc = SceneAccumulator()
    acc.update(dict(status='no_eligible_scored_vehicle', targets=0))
    empty = acc.summary()
    assert empty['evaluated_scenes'] == 0
    assert all(v is None for v in empty['metrics'].values())
    for n, value, focal in ((1, 1., True), (3, 3., False)):
        row = {key: value for key in METRIC_KEYS if focal or not key.startswith('focal_')}
        row.update(status='evaluated', targets=n)
        acc.update(row)
    result = acc.summary()
    assert result['evaluated_scenes'] == 2 and result['evaluated_targets'] == 4
    assert result['focal_vehicle_scenes'] == 1
    assert result['metrics']['focal_minADE'] == 1.
    assert result['metrics']['marginal_minADE'] == 2.
    assert result['marginal_actor_weighted']['marginal_minADE'] == 2.5


def test_reject_empty_nonfinite_or_mismatched_predictions():
    with pytest.raises(ValueError):
        marginal_actor_metrics(torch.zeros(0, 6, 30, 2), torch.zeros(0, 30, 2))
    with pytest.raises(ValueError):
        marginal_actor_metrics(torch.zeros(1, 6, 30, 2), torch.zeros(1, 29, 2))
    with pytest.raises(ValueError):
        marginal_actor_metrics(torch.zeros(1, 6, 30, 2), torch.full((1, 30, 2), float('nan')))
