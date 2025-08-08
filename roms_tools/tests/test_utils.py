import numpy as np
import pytest

from roms_tools.utils import generate_focused_coordinate_range


@pytest.mark.parametrize(
    "min_val, max_val, center, sc, N",
    [
        (-20.0, 5.5, -3.1, 1.0, 100),
        (100.0, 200.0, 150.0, 30.0, 100),
        (0.0, 2000.0, 150.0, 0.0, 100),
        (0.0, 2000.0, 150.0, 30.0, 100),
    ],
)
def test_coordinate_range_monotonicity(min_val, max_val, center, sc, N):
    centers, faces = generate_focused_coordinate_range(
        min_val=min_val, max_val=max_val, center=center, sc=sc, N=N
    )
    assert np.all(np.diff(faces) > 0), "faces is not strictly increasing"
    assert np.all(np.diff(centers) > 0), "centers is not strictly increasing"
