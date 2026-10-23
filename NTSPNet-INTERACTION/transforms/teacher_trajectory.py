from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class TeacherTrajectoryTransform(BaseTransform):
    """Attach cached final-anchor teacher trajectories to predicted agents."""

    schema_version = 1

    def __init__(self, cache_dir: str, split: str = 'train') -> None:
        super(TeacherTrajectoryTransform, self).__init__()
        self.cache_dir = Path(cache_dir).expanduser()
        self.split = split

    @staticmethod
    def _case_id(data: HeteroData) -> int:
        case_id = data['case_id']
        if torch.is_tensor(case_id):
            if case_id.numel() != 1:
                raise ValueError('teacher cache expects one case before batching')
            return int(case_id.item())
        return int(case_id)

    def __call__(self, data: HeteroData) -> HeteroData:
        sample_id = f"{data['scenario_name']}_{self._case_id(data)}.pt"
        sidecar_path = self.cache_dir / self.split / sample_id
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                'teacher trajectory sidecar is required but was not found: '
                f'{sidecar_path}. Run compute_teacher_trajectory_cache.py.'
            )

        sidecar = torch.load(sidecar_path, map_location='cpu')
        if int(sidecar.get('schema_version', -1)) != self.schema_version:
            raise ValueError(
                f'unsupported teacher trajectory schema: {sidecar_path}'
            )
        if sidecar.get('sample_id') != sample_id:
            raise ValueError(
                f'teacher trajectory sample ID mismatch: {sidecar_path}'
            )

        agent_store = data['agent']
        processed_agent_id = agent_store['id'].to(torch.long).cpu()
        sidecar_agent_id = sidecar['agent_id'].to(torch.long).cpu()
        if not torch.equal(sidecar_agent_id, processed_agent_id):
            raise ValueError(
                'teacher trajectory agent order does not match processed data: '
                f'{sidecar_path}'
            )

        predicted_mask = agent_store['category'].eq(1).cpu()
        sidecar_predicted_mask = sidecar['predicted_agent_mask'].bool().cpu()
        if not torch.equal(sidecar_predicted_mask, predicted_mask):
            raise ValueError(
                'teacher trajectory predicted-agent mask does not match '
                f'processed data: {sidecar_path}'
            )

        teacher_trajectory = sidecar['trajectory'].float().cpu()
        expected_agent_count = int(predicted_mask.sum().item())
        if teacher_trajectory.ndim != 4:
            raise ValueError(
                f'teacher trajectory must have shape [N,K,F,2]: {sidecar_path}'
            )
        if (
            teacher_trajectory.size(0) != expected_agent_count
            or teacher_trajectory.size(-1) != 2
        ):
            raise ValueError(
                'teacher trajectory has an incompatible shape in '
                f'{sidecar_path}'
            )
        if not torch.isfinite(teacher_trajectory).all():
            raise ValueError(
                f'teacher trajectory contains non-finite values: {sidecar_path}'
            )

        full_trajectory = agent_store['position'].new_zeros(
            (agent_store['position'].size(0),)
            + tuple(teacher_trajectory.shape[1:])
        )
        full_trajectory[predicted_mask] = teacher_trajectory.to(
            dtype=full_trajectory.dtype,
        )
        agent_store['teacher_trajectory'] = full_trajectory
        agent_store['teacher_trajectory_valid_mask'] = predicted_mask.to(
            device=full_trajectory.device,
        )
        return data
