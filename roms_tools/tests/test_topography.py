from roms_tools import Grid
from roms_tools.setup.topography import _compute_rfactor
import numpy as np
import numpy.testing as npt
from scipy.ndimage import label


def test_enclosed_regions():
    """Test that there are only two connected regions, one dry and one wet."""

    grid = Grid(
        nx=100,
        ny=100,
        size_x=1800,
        size_y=2400,
        center_lon=30,
        center_lat=61,
        rot=20,
    )

    reg, nreg = label(grid.ds.mask_rho)
    npt.assert_equal(nreg, 2)


def test_rmax_criterion():
    grid = Grid(
        nx=100,
        ny=100,
        size_x=1800,
        size_y=2400,
        center_lon=30,
        center_lat=61,
        rot=20,
    )
    r_eta, r_xi = _compute_rfactor(grid.ds.h)
    rmax0 = np.max([r_eta.max(), r_xi.max()])
    npt.assert_array_less(rmax0, 0.2)


def test_hmin_criterion():
    grid = Grid(
        nx=100,
        ny=100,
        size_x=1800,
        size_y=2400,
        center_lon=30,
        center_lat=61,
        rot=20,
        hmin=5,
    )

    assert np.less_equal(grid.hmin, grid.ds.h.min())


def test_data_consistency():
    """
    Test that the topography generation remains consistent.
    """

    grid = Grid(
        nx=3, ny=3, size_x=1500, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )

    expected_h = np.array(
        [
            [4505.16995868, 4505.16995868, 4407.37986032, 4306.51226663, 4306.51226663],
            [4505.16995868, 4505.16995868, 4407.37986032, 4306.51226663, 4306.51226663],
            [4400.69482254, 4400.69482254, 3940.84931344, 3060.19573878, 3060.19573878],
            [4234.97356606, 4234.97356606, 2880.90226836, 2067.46801754, 2067.46801754],
            [4234.97356606, 4234.97356606, 2880.90226836, 2067.46801754, 2067.46801754],
        ]
    )

    assert np.allclose(grid.ds["h"].values, expected_h)
