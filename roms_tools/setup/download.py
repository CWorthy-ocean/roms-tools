import pooch
import xarray as xr
from dataclasses import dataclass, field
import glob
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Optional, List
import dask
import warnings

# Create a Pooch object to manage the global topography data
pup_data = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("roms-tools"),
    base_url="https://github.com/CWorthy-ocean/roms-tools-data/raw/main/",
    # The registry specifies the files that can be fetched
    registry={
        "etopo5.nc": "sha256:23600e422d59bbf7c3666090166a0d468c8ee16092f4f14e32c4e928fbcd627b",
    },
)

# Create a Pooch object to manage the test data
pup_test_data = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("roms-tools"),
    base_url="https://github.com/CWorthy-ocean/roms-tools-test-data/raw/main/",
    # The registry specifies the files that can be fetched
    registry={
        "GLORYS_test_data.nc": "648f88ec29c433bcf65f257c1fb9497bd3d5d3880640186336b10ed54f7129d2",
        "ERA5_regional_test_data.nc": "bd12ce3b562fbea2a80a3b79ba74c724294043c28dc98ae092ad816d74eac794",
        "ERA5_global_test_data.nc": "8ed177ab64c02caf509b9fb121cf6713f286cc603b1f302f15f3f4eb0c21dc4f",
        "TPXO_global_test_data.nc": "457bfe87a7b247ec6e04e3c7d3e741ccf223020c41593f8ae33a14f2b5255e60",
        "TPXO_regional_test_data.nc": "11739245e2286d9c9d342dce5221e6435d2072b50028bef2e86a30287b3b4032",
    },
)


def fetch_topo(topography_source: str) -> xr.Dataset:
    """
    Load the global topography data as an xarray Dataset.

    Parameters
    ----------
    topography_source : str
        The source of the topography data to be loaded. Available options:
        - "ETOPO5"

    Returns
    -------
    xr.Dataset
        The global topography data as an xarray Dataset.
    """
    # Mapping from user-specified topography options to corresponding filenames in the registry
    topo_dict = {"ETOPO5": "etopo5.nc"}

    # Fetch the file using Pooch, downloading if necessary
    fname = pup_data.fetch(topo_dict[topography_source])

    # Load the dataset using xarray and return it
    ds = xr.open_dataset(fname)
    return ds


def download_test_data(filename: str) -> str:
    """
    Download the test data file.

    Parameters
    ----------
    filename : str
        The name of the test data file to be downloaded. Available options:
        - "GLORYS_test_data.nc"
        - "ERA5_regional_test_data.nc"
        - "ERA5_global_test_data.nc"

    Returns
    -------
    str
        The path to the downloaded test data file.
    """
    # Fetch the file using Pooch, downloading if necessary
    fname = pup_test_data.fetch(filename)

    return fname
