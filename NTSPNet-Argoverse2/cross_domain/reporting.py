"""Matched-target marginal metrics and scene-count strata for zero-shot reports."""
import math

MARGINAL_KEYS = ('marginal_minADE', 'marginal_minFDE', 'marginal_actorMR')
METRIC_KEYS = ('minJointADE', 'minJointFDE', 'minJointMR',
               'focal_minADE', 'focal_minFDE', 'focal_MR', 'cv_ADE', 'cv_FDE', 'cv_MR',
               *MARGINAL_KEYS, 'joint_minus_marginal_ADE', 'joint_minus_marginal_FDE')


def marginal_actor_metrics(prediction, target):
    """Per-actor independent ADE/FDE oracles, each minimizing over K modes.

    No scene mode alignment is enforced here. MR uses each actor's minimum FDE
    with a strict >2m threshold. These are custom diagnostics, not an assertion
    of compliance with an official AV2 evaluation protocol.
    """
    import torch
    if (prediction.ndim != 4 or target.ndim != 3 or
            prediction.shape[0] != target.shape[0] or
            prediction.shape[2:] != target.shape[1:] or target.shape[-1] != 2 or
            min(prediction.shape[:3]) == 0):
        raise ValueError('Expected nonempty [N,K,F,2] predictions and [N,F,2] targets')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite predictions or targets')
    errors = torch.linalg.vector_norm(prediction - target[:, None], dim=-1)
    ade = errors.mean(-1).min(-1).values
    fde = errors[:, :, -1].min(-1).values
    return dict(marginal_minADE=ade, marginal_minFDE=fde,
                marginal_actorMR=(fde > 2.).to(errors.dtype))


class SceneAccumulator:
    """Streaming scene means, with separate target-weighted marginal statistics."""
    def __init__(self):
        self.scenes = self.targets = self.focal_scenes = 0
        self.sums = dict.fromkeys(METRIC_KEYS, 0.)
        self.counts = dict.fromkeys(METRIC_KEYS, 0)
        self.actor_sums = dict.fromkeys(MARGINAL_KEYS, 0.)

    def update(self, row):
        if row['status'] != 'evaluated':
            return
        n = int(row['targets'])
        if n <= 0:
            raise ValueError('Evaluated scene must contain targets')
        self.scenes += 1; self.targets += n
        self.focal_scenes += int(row.get('focal_minADE') is not None)
        for key in METRIC_KEYS:
            if key not in row:
                if not key.startswith('focal_'):
                    raise ValueError('Missing scene metric: ' + key)
                continue
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError('Nonfinite scene metric: ' + key)
            self.sums[key] += value; self.counts[key] += 1
        for key in MARGINAL_KEYS:
            self.actor_sums[key] += float(row[key]) * n

    def summary(self):
        return dict(evaluated_scenes=self.scenes, evaluated_targets=self.targets,
                    focal_vehicle_scenes=self.focal_scenes,
                    metrics={key: self.sums[key] / self.counts[key] if self.counts[key] else None
                             for key in METRIC_KEYS},
                    marginal_actor_weighted={key: value / self.targets if self.targets else None
                                             for key, value in self.actor_sums.items()})
