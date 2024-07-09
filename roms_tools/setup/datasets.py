import pooch
import xarray as xr


FRANK = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("roms-tools"),
    base_url="https://github.com/CWorthy-ocean/roms-tools-data/raw/main/",
    # If this is a development version, get the data from the "main" branch
    # The registry specifies the files that can be fetched
    registry={
        "etopo5.nc": "sha256:23600e422d59bbf7c3666090166a0d468c8ee16092f4f14e32c4e928fbcd627b",
    },
)


def fetch_topo(topography_source) -> xr.Dataset:
    """
    Load the global topography data as an xarray Dataset.
    """
    # Mapping from user-specified topography options to corresponding filenames in the registry
    topo_dict = {"etopo5": "etopo5.nc"}

    # The file will be downloaded automatically the first time this is run
    # returns the file path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    fname = FRANK.fetch(topo_dict[topography_source])
    # The "fetch" method returns the full path to the downloaded data file.
    # All we need to do now is load it with our standard Python tools.
    ds = xr.open_dataset(fname)
    return ds


def fetch_ssr_correction(correction_source) -> xr.Dataset:
    """
    Load the SSR correction data as an xarray Dataset.
    """
    # Mapping from user-specified topography options to corresponding filenames in the registry
    topo_dict = {"corev2": "SSR_correction.nc"}

    # The file will be downloaded automatically the first time this is run
    # returns the file path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    fname = FRANK.fetch(topo_dict[correction_source])
    # The "fetch" method returns the full path to the downloaded data file.
    # All we need to do now is load it with our standard Python tools.
    ds = xr.open_dataset(fname)
    return ds
