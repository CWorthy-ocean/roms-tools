import pooch

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
        "SSR_correction.nc": "sha256:a170c1698e6cc2765b3f0bb51a18c6a979bc796ac3a4c014585aeede1f1f8ea0",
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
    },
)

# Create a Pooch object to manage the global SAL TPXO data
sal_data = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("roms-tools"),
    base_url="https://github.com/CWorthy-ocean/roms-tools-data/raw/main/",
    # The registry specifies the files that can be fetched
    registry={
        "sal_tpxo9.v2a.nc": "sha256:5343d745b4374170a069bfc2c67dcdd9f1dc4eb0df7c0e6de5c004432c903f40",
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
        "coarsened_UNIFIED_bgc_dataset.nc": "34bc5f48d5aeac45187b3da04ecd3d86cc892147fd71cdc1e66efe7055d081ba",
        "grid_created_with_matlab.nc": "fd537ef8159fabb18e38495ec8d44e2fa1b7fb615fcb1417dd4c0e1bb5f4e41d",
        "etopo5_coarsened_and_shifted.nc": "9a5cb4b38c779d22ddb0ad069b298b9722db34ca85a89273eccca691e89e6f96",
        "srtm15_coarsened.nc": "48bc8f4beecfdca9c192b13f4cbeef1455f49d8261a82563aaec5757e100dff9",
        "eastpac25km_rst.19980106000000.nc": "8f56d72bd8daf72eb736cc6705f93f478f4ad0ae4a95e98c4c9393a38e032f4c",
        "eastpac25km_rst.19980126000000.nc": "20ad9007c980d211d1e108c50589183120c42a2d96811264cf570875107269e4",
        "epac25km_grd.nc": "ec26c69cda4c4e96abde5b7756c955a7e1074931ab5a0641f598b099778fb617",
    },
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
    fname = topo_data.fetch(filename)

    return fname


def download_river_data(filename: str) -> str:
    """Download river data file.

    Parameters
    ----------
    filename : str
        The name of the test data file to be downloaded. Available options:
        - "dai_trenberth_may2019.nc"
    Returns
    -------
    str
        The path to the downloaded test data file.
    """

    # Fetch the file using Pooch, downloading if necessary
    fname = river_data.fetch(filename)

    return fname


def download_correction_data(filename: str) -> str:
    """Download the correction data file.

    Parameters
    ----------
    filename : str
        The name of the test data file to be downloaded. Available options:
        - "SSR_correction.nc"

    Returns
    -------
    str
        The path to the downloaded test data file.
    """
    # Fetch the file using Pooch, downloading if necessary
    fname = correction_data.fetch(filename)

    return fname


def download_sal_data(filename: str) -> str:
    """Download the SAL data file.

    Parameters
    ----------
    filename : str
        The name of the test data file to be downloaded. Available options:
        - "sal_tpxo9.v2a.nc"

    Returns
    -------
    str
        The path to the downloaded test data file.
    """
    # Fetch the file using Pooch, downloading if necessary
    fname = sal_data.fetch(filename)

    return fname


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
    fname = pup_test_data.fetch(filename)

    return fname
