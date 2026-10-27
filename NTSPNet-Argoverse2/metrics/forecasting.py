"""Explicit AV2 single-agent / multi-world metric reductions."""
import torch
from torchmetrics import Metric


def scene_probabilities(probability):
    """Geometric-mean consensus over actors, normalized over shared worlds."""
    return torch.softmax(probability.clamp_min(1e-8).log().mean(dim=0), dim=-1)


def trajectory_metrics(prediction, target, probability, joint=False):
    """Input [M,K,60,2], [M,60,2], [M,K]; complete future required.

    Select the best FDE mode, then report its ADE and Brier-FDE. For joint
    evaluation the same world is selected for every actor; MR is the fraction
    of actors with terminal error >2m in that selected world.
    """
    error = torch.linalg.vector_norm(prediction - target[:, None], dim=-1)
    fde = error[..., -1]; ade = error.mean(dim=-1)
    if joint:
        best = fde.mean(dim=0).argmin()
        p = scene_probabilities(probability)
        return torch.stack((ade[:, best].mean(), fde[:, best].mean(),
                            (fde[:, best] > 2.).float().mean(),
                            fde[:, best].mean() + (1 - p[best]).square()))[None]
    best = fde.argmin(dim=-1)
    rows = torch.arange(len(best), device=best.device)
    selected_fde = fde[rows, best]
    return torch.stack((ade[rows, best], selected_fde, (selected_fde > 2.).float(),
                        selected_fde + (1 - probability[rows, best]).square()), dim=-1)


class ForecastMetrics(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.zeros(4, dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state('count', default=torch.zeros((), dtype=torch.long), dist_reduce_fx='sum')

    def update(self, values):
        self.total += values.double().sum(dim=0)
        self.count += len(values)

    def compute(self):
        return self.total / self.count.clamp_min(1)
