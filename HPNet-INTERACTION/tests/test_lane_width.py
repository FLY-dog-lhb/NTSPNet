import pytest
import torch

from datasets.INTERACTION_dataset import INTERACTIONDataset


def test_average_lane_width_uses_nearest_left_and_right_boundary_distances():
    centerline = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    left_boundary = torch.tensor([[0.0, 2.0], [2.0, 2.0]])
    right_boundary = torch.tensor([[0.0, -1.0], [2.0, -1.0]])

    width = INTERACTIONDataset.compute_average_lane_width(
        centerline,
        left_boundary,
        right_boundary,
    )

    assert torch.allclose(width, torch.tensor(3.0))


def test_average_lane_width_rejects_missing_geometry():
    with pytest.raises(ValueError, match='must be non-empty'):
        INTERACTIONDataset.compute_average_lane_width(
            torch.empty((0, 2)),
            torch.empty((0, 2)),
            torch.empty((0, 2)),
        )
