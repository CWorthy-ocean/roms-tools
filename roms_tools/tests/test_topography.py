from roms_tools import Grid
from roms_tools.setup.topography import _compute_rfactor
import numpy as np
import numpy.testing as npt
from scipy.ndimage import label


class TestTopography:
    def test_enclosed_regions(self):
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

    def test_rmax_criterion(self):
        grid = Grid(
            nx=100,
            ny=100,
            size_x=1800,
            size_y=2400,
            center_lon=30,
            center_lat=61,
            rot=20,
            smooth_factor=4,
            rmax=0.2,
        )
        r_eta, r_xi = _compute_rfactor(grid.ds.h)
        rmax0 = np.max([r_eta.max(), r_xi.max()])
        npt.assert_array_less(rmax0, grid.rmax)

    def test_hmin_criterion(self):
        grid = Grid(
            nx=100,
            ny=100,
            size_x=1800,
            size_y=2400,
            center_lon=30,
            center_lat=61,
            rot=20,
            smooth_factor=2,
            rmax=0.2,
            hmin=5,
        )

        assert np.less_equal(grid.hmin, grid.ds.h.min())
