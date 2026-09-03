import logging
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import pooch

#: Number of times a download is attempted before the error is raised.
MAX_DOWNLOAD_ATTEMPTS = 3

#: Seconds to wait before the first retry; doubled for each further attempt.
RETRY_BACKOFF_SECONDS = 2.0


def _fetch(manager: pooch.Pooch, filename: str) -> str:
    """Fetch a registered file, retrying transient network failures.

    Pooch has no retry logic of its own, so a single dropped connection or a
    read timeout (30 s by default) against the data repository aborts the whole
    call. Retries use exponential backoff. Files already in the local cache are
    never re-downloaded, so a retry only refetches what actually failed.

    The caught type is ``OSError``, which covers every network failure raised
    here (``requests.exceptions.RequestException`` subclasses it) without
    importing ``requests``. Local I/O errors are therefore retried too, which
    only delays an unavoidable failure. A checksum mismatch raises
    ``ValueError`` and is deliberately not retried, so a bad registry entry
    fails immediately.

    Parameters
    ----------
    manager : pooch.Pooch
        The Pooch instance the file is registered with.
    filename : str
        The name of the file to fetch.

    Returns
    -------
    str
        The path to the file in the local cache.
    """
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS):
        try:
            return manager.fetch(filename)
        except OSError as error:
            delay = RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
            logging.warning(
                "Download of %s failed (attempt %d of %d): %s. Retrying in %.1f s.",
                filename,
                attempt,
                MAX_DOWNLOAD_ATTEMPTS,
                error,
                delay,
            )
            time.sleep(delay)

    # Final attempt: let the error propagate if this one fails too.
    return manager.fetch(filename)


# Create a Pooch object to manage the global topography data
topo_data = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("roms-tools"),
    base_url="https://github.com/CWorthy-ocean/roms-tools-data/raw/main/",
    # The registry specifies the files that can be fetched
    registry={
        "etopo5.nc": "sha256:23600e422d59bbf7c3666090166a0d468c8ee16092f4f14e32c4e928fbcd627b",
    },
)

# Create a Pooch object to manage the global SWR correction data
correction_data = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("roms-tools"),
    base_url="https://github.com/CWorthy-ocean/roms-tools-data/raw/main/",
    # The registry specifies the files that can be fetched
    registry={
        "ERA5_correction.nc": "sha256:7729179c90c5c4d1202659194bf82409dce21ec821911db1ad2dc027ce4a87ec",
    },
)

# Create a Pooch object to manage the global river data
river_data = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("roms-tools"),
    base_url="https://github.com/CWorthy-ocean/roms-tools-data/raw/main/",
    # The registry specifies the files that can be fetched
    registry={
        "dai_trenberth_may2019.nc": "sha256:793849e6aa60d1f6bdb480c345515fb2453d903c0a30599241b3d752f53715ab",
        "river_tracer_defaults.nc": "sha256:58b7f2e00c0a4f489fc0f345988b79a378ed56f5039d1409a00dc947275e7a61",
    },
)

# Create a Pooch object to manage the global SAL TPXO data
sal_data = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("roms-tools"),
    base_url="https://github.com/CWorthy-ocean/roms-tools-data/raw/main/",
    # The registry specifies the files that can be fetched
    registry={
        "sal_tpxo10.v2a.nc": "sha256:4309ce204a5e4884aae4dd5209c2ac5a130121176a2f49c1fad2021bba8737a1",
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
        "GLORYS_coarse_test_data.nc": "ed14ca6aa72810e2472e6ee21c59e5e38f59cd6eb39c14ff6a01ccba05d11d48",
        "GLORYS_NA_2012.nc": "b862add892f5d6e0d670c8f7fa698f4af5290ac87077ca812a6795e120d0ca8c",
        "GLORYS_NA_20120101.nc": "647a6a3227efff8520aedc757ecb591376464b41494ed3bb5d119700e98bba29",
        "GLORYS_NA_20121231.nc": "03c1155087195deff76ad3f136d6a7f35bc01ccae3402f3d95557a2886d39e71",
        "ERA5_regional_test_data.nc": "bd12ce3b562fbea2a80a3b79ba74c724294043c28dc98ae092ad816d74eac794",
        "ERA5_global_test_data.nc": "8ed177ab64c02caf509b9fb121cf6713f286cc603b1f302f15f3f4eb0c21dc4f",
        "global_grid_tpxo10.v2.nc": "26eb97cd135cd6f2b4e894c5f11bf7f860ff19cec8dbaa9190e37d30ee6e744e",
        "global_h_tpxo10.v2.nc": "ef60fae6d52fa514dcc59a737435d74aa798dc114b57f01b123aa39dbaffc592",
        "global_u_tpxo10.v2.nc": "022e57e6287e51f52eb1e5296614b1086e0e22ecd0bd57c9fd8d0e155babf5c3",
        "regional_grid_tpxo10v2a.nc": "c5022bfe93ead7cd46e836578645bd87cb5be63c736e660937c7f5703c968cbc",
        "regional_h_tpxo10.v2.nc": "202fd0c197490ac460af12cd9fa1156aa40023c0023c705f145c596de5b5ad3d",
        "regional_grid_tpxo10v2.nc": "0789b6a24ecb2ced522481dfcfb7282e32f999984747b9b9f46f044a8898d0ac",
        "regional_grid_tpxo9v5a.nc": "497a2ae9e6adc7e4b06408dadb57734e2ad24afaa3f0e2e4fd90ebc6eafc2557",
        "regional_h_tpxo10v2a.nc": "2df2f181f748a960e4072f975226f6f98f6a6c4d5b77da23057946585152d59c",
        "regional_h_tpxo10v2.nc": "202fd0c197490ac460af12cd9fa1156aa40023c0023c705f145c596de5b5ad3d",
        "regional_h_tpxo9v5a.nc": "c7e4d9ab73bc11dcb415f88c48131531488e1aed5113df5797e80d3d374607fc",
        "regional_u_tpxo10v2a.nc": "2d1680ecd53242e858281a762221d91827999967f8e1f3cb7de3d23b47efe8c8",
        "regional_u_tpxo10v2.nc": "3b0849473cbb7f9076ca907e4fc39eceda3c7d64659c121fa0692024d59dcdb3",
        "regional_u_tpxo9v5a.nc": "b0cc5f6934d2e212549c7120d458d61a4963ba73d17055e67cc9e4312901b041",
        "CESM_BGC_coarse_global_clim.nc": "20806e4e99285d6de168d3236e2d9245f4e9106474b1464beaa266a73e6ef79f",
        "CESM_BGC_2012.nc": "e374d5df3c1be742d564fd26fd861c2d40af73be50a432c51d258171d5638eb6",
        "CESM_regional_test_data_one_time_slice.nc": "43b578ecc067c85f95d6b97ed7b9dc8da7846f07c95331c6ba7f4a3161036a17",
        "CESM_regional_test_data_climatology.nc": "986a200029d9478fd43e6e4a8bc43e8a8f4407554893c59b5fcc2e86fd203272",
        "CESM_regional_coarse_test_data_climatology.nc": "5cde5f968fba7900b6ff5bcf135126b5e25185fc3bd842bf66052cc2a6197d81",
        "CESM_BGC_SURFACE_2012.nc": "3c4d156adca97909d0fac36bf50b99583ab37d8020d7a3e8511e92abf2331b38",
        "CESM_surface_global_test_data_climatology.nc": "a072757110c6f7b716a98f867688ef4195a5966741d2f368201ac24617254e35",
        "CESM_surface_global_test_data.nc": "874106ffbc8b1b220db09df1551bbb89d22439d795b4d1e5a24ee775e9a7bf6e",
        # Pre-v2.1 layout ('lat'/'lon'/'dep' dimensions, day-of-year 'month'); kept
        # registered so the legacy path stays covered by the tests.
        "coarsened_UNIFIED_bgc_dataset.nc": "sha256:269d5bcd8e6e64d3400362ae0a65afe049810ce06c536ccf31cca7c00f321bc1",
        "coarsened_UNIFIED_bgc_dataset_v2_1.nc": "sha256:732af496416cbfe181a87d056012363d5456dd268023de53646cd9ebf54cf712",
        "WOA_2018_quarterDeg_coarsened.nc": "sha256:673ce3c3a98bb386ccd899dbc23eeedf7d9a665b68ea52c96fd69829a4b929a7",
        "mbl_co2_bgc_dataset.nc": "sha256:797a9ef48f3c83a6920e44c0b441feb00fb35553db09dea9ed4ff36dcd68d968",
        "coarsened_OceanSODA_dataset.nc": "sha256:b4a284303c9c1a8904a6ea3fa338ff05ac042c0b024fb4e1ad4bbb1d9fedff38",
        "grid_created_with_matlab.nc": "fd537ef8159fabb18e38495ec8d44e2fa1b7fb615fcb1417dd4c0e1bb5f4e41d",
        "etopo5_coarsened_and_shifted.nc": "9a5cb4b38c779d22ddb0ad069b298b9722db34ca85a89273eccca691e89e6f96",
        "srtm15_coarsened.nc": "48bc8f4beecfdca9c192b13f4cbeef1455f49d8261a82563aaec5757e100dff9",
        "eastpac25km_rst.19980106000000.nc": "8f56d72bd8daf72eb736cc6705f93f478f4ad0ae4a95e98c4c9393a38e032f4c",
        "eastpac25km_rst.19980126000000.nc": "20ad9007c980d211d1e108c50589183120c42a2d96811264cf570875107269e4",
        "epac25km_grd.nc": "ec26c69cda4c4e96abde5b7756c955a7e1074931ab5a0641f598b099778fb617",
        "GSHHS_l_L1.dbf": "181236ffbf553a83d2afedc5fe5e1f2fea64190f56ea366fc7a8ff5aa6163663",
        "GSHHS_l_L1.prj": "98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
        "GSHHS_l_L1.shp": "bc76f101f9b8671f90e734b4026da91c20066fc627cc8b5889ba22d90cbf97e9",
        "GSHHS_l_L1.shx": "72879354892d80d6c39c612f645661ec0edc75f3f9f8f74b19d9387ae0327377",
        "EMODnet_C2_coarse100.nc": "4202a6a5877de726bf13f41de7a1edfea2db83278ce371fe7732eb4b6770ed6d",
        "rivr2o_riverinputs_2000.nc": "sha256:1ff94f4b732bb5fd91276120979a6af38d9a6f257570bb7fcc676c651f36dac0",
        "rivr2o_riverinputs_2001.nc": "sha256:3ca4ca6d12103ef8bdd67592c9eeaaf0e7a2a765f1d18ad4c978dee52ebbec04",
        "rivr2o_riverinputs_2002.nc": "sha256:0f760d85962ad025c88d39158e2a80ee33bb2d039877ef7dcb650ef32f56f660",
    },
)

RIVR2O_TEST_DATA_FILES = (
    "rivr2o_riverinputs_2000.nc",
    "rivr2o_riverinputs_2001.nc",
    "rivr2o_riverinputs_2002.nc",
)


def download_topo(filename: str) -> str:
    """Download simple topography file.

    Parameters
    ----------
    filename : str
        The name of the test data file to be downloaded. Available options:
        - "etopo5.nc"

    Returns
    -------
    str
        The path to the downloaded test data file.
    """
    # Fetch the file using Pooch, downloading if necessary
    fname = _fetch(topo_data, filename)

    return fname


def download_river_data(filename: str) -> str:
    """Download river data file.

    Parameters
    ----------
    filename : str
        The name of the river data file to be downloaded. Available options:
        - "dai_trenberth_may2019.nc"
        - "river_tracer_defaults.nc"

    Returns
    -------
    str
        The path to the downloaded file.
    """
    # Fetch the file using Pooch, downloading if necessary
    fname = _fetch(river_data, filename)

    return fname


def download_river_tracer_defaults(filename: str = "river_tracer_defaults.nc") -> str:
    """Download the river tracer default values NetCDF file.

    Parameters
    ----------
    filename : str
        The name of the river tracer defaults file to be downloaded. Available
        options:
        - "river_tracer_defaults.nc"

    Returns
    -------
    str
        The path to the downloaded file.
    """
    fname = _fetch(river_data, filename)

    return fname


def download_correction_data(filename: str) -> str:
    """Download the correction data file.

    Parameters
    ----------
    filename : str
        The name of the test data file to be downloaded. Available options:
        - "ERA5_correction.nc"

    Returns
    -------
    str
        The path to the downloaded test data file.
    """
    # Fetch the file using Pooch, downloading if necessary
    fname = _fetch(correction_data, filename)

    return fname


def download_sal_data(filename: str) -> str:
    """Download the SAL data file.

    Parameters
    ----------
    filename : str
        The name of the test data file to be downloaded. Available options:
        - "sal_tpxo10.v2a.nc"

    Returns
    -------
    str
        The path to the downloaded test data file.
    """
    # Fetch the file using Pooch, downloading if necessary
    fname = _fetch(sal_data, filename)

    return fname


def download_rivr2o_test_data() -> list[str]:
    """Download the regional RIVR2O test files from roms-tools-test-data.

    Returns
    -------
    list[str]
        Paths to the yearly NetCDF files (2000-2002).
    """
    return [download_test_data(filename) for filename in RIVR2O_TEST_DATA_FILES]


def download_test_data(filename: str) -> str:
    """Download the test data file.

    Parameters
    ----------
    filename : str
        The name of the test data file to be downloaded. Available options:
        - "GLORYS_test_data.nc"
        - "ERA5_regional_test_data.nc"
        - "ERA5_global_test_data.nc"
        - "TPXO_global_test_data.nc"
        - "TPXO_regional_test_data.nc"
        - "CESM_regional_test_data_one_time_slice.nc"
        - "CESM_regional_test_data_climatology.nc"

    Returns
    -------
    str
        The path to the downloaded test data file.
    """
    # Fetch the file using Pooch, downloading if necessary
    fname = _fetch(pup_test_data, filename)

    return fname


# ---------------------------------------------------------------------------
# World Ocean Atlas 2023
# ---------------------------------------------------------------------------

#: Root of the NCEI WOA23 netCDF tree.
WOA23_BASE_URL = "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA"

#: Internal variable key -> (NCEI directory, decade token, one-letter file code).
#:
#: The decade token is the averaging period *over years*, not the period within a
#: year: ``decav`` pools all years 1955-2022 for T/S, and ``all`` is the equivalent
#: all-years token for the nutrients and oxygen (1965-2022). The month is carried
#: entirely by the two-digit suffix on the filename (``01``-``12``; ``00`` is the
#: annual field). Nutrients and oxygen are published on the 1 degree grid only.
WOA23_BGC_VARIABLES: dict[str, tuple[str, str, str]] = {
    "NO3": ("nitrate", "all", "n"),
    "PO4": ("phosphate", "all", "p"),
    "SiO3": ("silicate", "all", "i"),
    "O2": ("oxygen", "all", "o"),
    "temp_bgc": ("temperature", "decav", "t"),
    "salt_bgc": ("salinity", "decav", "s"),
}

#: Grid-resolution suffix for the 1 degree product.
WOA23_GRID = "01"


def woa23_filename(code: str, period: int, decade: str, grid: str = WOA23_GRID) -> str:
    """Build a WOA23 netCDF filename.

    Parameters
    ----------
    code : str
        One-letter variable code, e.g. ``"n"`` for nitrate.
    period : int
        ``0`` for the annual field, ``1``-``12`` for a monthly field.
    decade : str
        Over-years averaging token, e.g. ``"decav"`` or ``"all"``.
    grid : str, optional
        Grid-resolution suffix; ``"01"`` (1 degree) by default.

    Returns
    -------
    str
        The bare filename, e.g. ``"woa23_all_n01_01.nc"``.
    """
    return f"woa23_{decade}_{code}{period:02d}_{grid}.nc"


def woa23_url(directory: str, decade: str, filename: str) -> str:
    """Build the full NCEI download URL for a WOA23 file."""
    return f"{WOA23_BASE_URL}/{directory}/netcdf/{decade}/1.00/{filename}"


def _download_one(url: str, path: Path) -> None:
    """Download ``url`` to ``path`` atomically, retrying transient failures.

    The file is streamed to a temporary file in the destination directory and
    then moved into place, so an interrupted download never leaves a truncated
    file that a later run would mistake for a complete one.
    """
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, dir=str(path.parent), suffix=".part"
            ) as tmpfile:
                tmp_path = Path(tmpfile.name)
                with urlopen(url) as response:
                    while chunk := response.read(1 << 20):
                        tmpfile.write(chunk)
            tmp_path.replace(path)
            return
        except OSError as error:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise
            delay = RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
            logging.warning(
                "Download of %s failed (attempt %d of %d): %s. Retrying in %.1f s.",
                url,
                attempt,
                MAX_DOWNLOAD_ATTEMPTS,
                error,
                delay,
            )
            time.sleep(delay)


def download_woa23_bgc(
    target_dir: str | Path | None = None,
    variables: list[str] | None = None,
    include_annual: bool = True,
    ts_decade: str = "decav",
    clobber: bool = False,
) -> Path:
    """Download the WOA23 1 degree BGC climatology into a local directory.

    Fetches the twelve monthly files for each requested variable and, when
    ``include_annual`` is set, the full-depth annual file as well. The annual
    files are what :class:`~roms_tools.datasets.lat_lon_datasets.WOABGCDataset`
    uses to backfill below the monthly depth limits (800 m for the nutrients,
    1500 m for oxygen and T/S).

    Files already present are left alone unless ``clobber`` is set, so an
    interrupted download can be resumed by simply calling this again.

    Parameters
    ----------
    target_dir : str or Path, optional
        Destination directory. Defaults to ``pooch.os_cache("roms-tools") / "WOA23"``.
    variables : list of str, optional
        Internal variable keys to fetch; defaults to all of
        :data:`WOA23_BGC_VARIABLES`.
    include_annual : bool, optional
        Also fetch the ``*00`` annual files. Required for the ``"annual_blend"``
        deep-fill mode. Default ``True``.
    ts_decade : str, optional
        Over-years averaging token for temperature and salinity. Default
        ``"decav"`` (all years). WOA23 also offers the 30-year normals
        ``"decav71A0"``, ``"decav81B0"`` and ``"decav91C0"``. The nutrients are
        published under ``"all"`` only and ignore this.
    clobber : bool, optional
        Re-download files that already exist. Default ``False``.

    Returns
    -------
    Path
        The directory the files were written to.
    """
    directory = (
        Path(target_dir)
        if target_dir is not None
        else Path(pooch.os_cache("roms-tools")) / "WOA23"
    )
    directory.mkdir(parents=True, exist_ok=True)

    keys = variables if variables is not None else list(WOA23_BGC_VARIABLES)
    unknown = sorted(set(keys) - set(WOA23_BGC_VARIABLES))
    if unknown:
        raise ValueError(
            f"Unknown WOA23 variable key(s) {unknown}. "
            f"Valid keys: {sorted(WOA23_BGC_VARIABLES)}."
        )

    periods = list(range(1, 13)) + ([0] if include_annual else [])

    for key in keys:
        subdir, decade, code = WOA23_BGC_VARIABLES[key]
        if key in ("temp_bgc", "salt_bgc"):
            decade = ts_decade
        for period in periods:
            filename = woa23_filename(code, period, decade)
            path = directory / filename
            if path.exists() and not clobber:
                continue
            url = woa23_url(subdir, decade, filename)
            logging.info("Downloading %s -> %s", url, path)
            _download_one(url, path)

    return directory
