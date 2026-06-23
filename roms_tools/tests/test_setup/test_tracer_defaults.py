import pytest
import xarray as xr

from roms_tools.datasets.download import download_river_tracer_defaults
from roms_tools.datasets.river_datasets import (
    RECOMMENDED_VALUE_INDEX,
    VALUE_OPTION_DIM,
    RiverTracerDefaultsDataset,
)
from roms_tools.setup.utils import (
    MARBL_TRACER_NAMES,
    _load_tracer_defaults,
    get_tracer_defaults,
)


@pytest.fixture(autouse=True)
def clear_tracer_defaults_cache():
    _load_tracer_defaults.cache_clear()
    yield
    _load_tracer_defaults.cache_clear()


@pytest.fixture
def tracer_defaults_path():
    """Path to the river tracer defaults NetCDF, fetched via pooch."""
    return download_river_tracer_defaults()


class TestTracerDefaults:
    def test_reads_recommended_values_from_netcdf(self):
        defaults = get_tracer_defaults()

        assert defaults["DIC"] == pytest.approx(1640.071)
        assert defaults["ALK"] == pytest.approx(1173.693)
        assert defaults["DOC"] == pytest.approx(460.476)
        assert defaults["DON"] == pytest.approx(31.61635)
        assert defaults["DOP"] == pytest.approx(1.0242011)
        assert defaults["NH4"] == 0.0
        assert defaults["spC"] == 0.0
        # 32 MARBL tracers plus temperature and salinity
        assert len(defaults) == len(MARBL_TRACER_NAMES)
        assert len(defaults) == 34

    def test_dataclass_exposes_dataset(self, tracer_defaults_path):
        data = RiverTracerDefaultsDataset(filename=tracer_defaults_path)

        assert "DIC" in data.ds
        assert data.ds["DIC"].attrs["units"] == "mmol/m3"
        assert data.defaults["DIC"] == pytest.approx(1640.071)

    def test_uses_recommended_index_not_alternate(self, tmp_path):
        data_vars = {
            "PO4": (VALUE_OPTION_DIM, [4.2453, 99.9]),
            "NO3": (VALUE_OPTION_DIM, [57.3565, 88.8]),
        }
        for tracer in MARBL_TRACER_NAMES:
            if tracer not in data_vars:
                data_vars[tracer] = (VALUE_OPTION_DIM, [1.0, 999.0])

        ds = xr.Dataset(data_vars, coords={VALUE_OPTION_DIM: [0, 1]})
        nc_path = tmp_path / "river_tracer_defaults.nc"
        ds.to_netcdf(nc_path)

        data = RiverTracerDefaultsDataset(filename=nc_path)

        assert data.defaults["PO4"] == pytest.approx(4.2453)
        assert data.defaults["NO3"] == pytest.approx(57.3565)
        assert data.defaults["temp"] == pytest.approx(1.0)
        assert data.defaults["PO4"] != pytest.approx(99.9)

    def test_recommended_index_is_zero(self):
        assert RECOMMENDED_VALUE_INDEX == 0

    def test_requires_calendar_discharge_time_is_false(self, tracer_defaults_path):
        data = RiverTracerDefaultsDataset(filename=tracer_defaults_path)
        assert data.requires_calendar_discharge_time is False

    def test_temporal_interpolation_is_none(self, tracer_defaults_path):
        data = RiverTracerDefaultsDataset(filename=tracer_defaults_path)
        assert data.temporal_interpolation == "none"
