from typing import List

import torch
from torchmetrics import Metric


class minJointMR(Metric):
    """Joint Miss Rate for interactive prediction.

    A scenario is "missed" if, for the best mode (chosen by minimum
    worst-case agent FDE), the worst-case agent FDE exceeds `threshold`.
    """

    def __init__(self,
                 threshold: float = 2.0,
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self,
               scored_predict_traj: List[torch.Tensor],   # [(n1,K,F,2),...,(nb,K,F,2)]
               scored_target_traj: List[torch.Tensor],    # [(n1,F,2),...,(nb,F,2)]
               scored_target_mask: List[torch.Tensor],    # [(n1,F),...,(nb,F)]
               ) -> None:
        batch_size = len(scored_predict_traj)
        for i in range(batch_size):
            valid = scored_target_mask[i][:, -1]          # [ni] bool, last-step valid
            errors = torch.norm(
                scored_predict_traj[i] - scored_target_traj[i].unsqueeze(1),
                p=2, dim=-1,
            )                                             # [ni, K, F]
            fde = errors[:, :, -1]                        # [ni, K]
            fde_valid = fde[valid]                        # [ni_valid, K]

            if fde_valid.numel() == 0:
                continue

            worst_per_mode = fde_valid.max(dim=0).values  # [K]
            min_worst = worst_per_mode.min()              # scalar
            self.sum = self.sum + (min_worst > self.threshold).float()
        self.count = self.count + batch_size

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
