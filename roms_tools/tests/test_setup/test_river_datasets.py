from datetime import datetime

import numpy as np
import xarray as xr

from roms_tools.setup.river_datasets import RiverDataset


class TestRiverDataset:
    def test_deduplicate_river_names(self, tmp_path):
        sample_dim_and_var_names = {
            "dim_names": {"station": "station", "time": "time"},
            "var_names": {
                "latitude": "lat",
                "longitude": "lon",
                "flux": "flux",
                "ratio": "ratio",
                "name": "name",
            },
        }

        data = {
            "lat": (["station"], [10.0, 20.0, 30.0]),
            "lon": (["station"], [100.0, 110.0, 120.0]),
            "flux": (["time", "station"], np.random.rand(1, 3)),
            "ratio": (["time", "station"], np.random.rand(1, 3)),
            "name": (["station"], ["Amazon", "Nile", "Amazon"]),  # duplicate
        }
        coords = {"station": [0, 1, 2], "time": [0]}
        ds = xr.Dataset(data, coords=coords)

        # Write to temporary NetCDF file
        file_path = tmp_path / "rivers.nc"
        ds.to_netcdf(file_path)

        river_dataset = RiverDataset(
            filename=file_path,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 1, 2),
            dim_names=sample_dim_and_var_names["dim_names"],
            var_names=sample_dim_and_var_names["var_names"],
        )

        names = river_dataset.ds["name"].values
        assert "Amazon_1" in names
        assert "Amazon_2" in names
        assert "Nile" in names
        assert len(set(names)) == len(names)  # all names must be unique
