Datasets
========

ROMS-Tools relies on several external datasets. Some are accessed automatically; others must be downloaded manually before running ROMS-Tools.

.. list-table::
   :header-rows: 1
   :widths: 22 35 20 23

   * - Dataset
     - Required for
     - Download needed
     - Access
   * - ETOPO5
     - Grid (Topography)
     - No (auto-downloaded)
     - `NOAA <https://www.ncei.noaa.gov/products/etopo-global-relief-model>`_
   * - SRTM15
     - Grid (Topography, high-resolution)
     - Yes
     - `UCSD <https://topex.ucsd.edu/WWW_html/srtm15_plus.html>`_
   * - EMODnet
     - Grid (Topography, high-resolution)
     - Yes
     - `EMODnet <https://doi.org/10.12770/ff3aff8a-cff1-44a3-a2c8-1910bf109f85>`_
   * - ETOPO2022
     - Grid (Topography, high-resolution)
     - Yes
     - `NOAA <https://www.ncei.noaa.gov/products/etopo-global-relief-model>`_
   * - Natural Earth
     - Grid (Land-Sea Mask)
     - No (auto-downloaded)
     - `Natural Earth <https://www.naturalearthdata.com/>`_
   * - GSHHG
     - Grid (Land-Sea Mask, high-resolution)
     - Yes
     - `SOEST <https://www.soest.hawaii.edu/pwessel/gshhg/>`_
   * - TPXO
     - Tidal Forcing
     - Yes
     - `OSU <https://www.tpxo.net/global>`_
   * - GLORYS
     - Initial & Boundary Conditions
     - Yes
     - `Copernicus Marine <https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description>`_
   * - Unified BGC Dataset
     - BGC Initial, Boundary & Surface Forcing; Salinity Restoring
     - Yes
     - `Google Drive <https://drive.google.com/uc?id=1NKbAe1ARtU68Np3bcwdd7nadeEUgdcef>`_
   * - ERA5
     - Surface Forcing
     - Optional (streaming supported)
     - `Climate Data Store <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>`_
   * - Dai & Trenberth
     - River Forcing (discharge)
     - No (auto-downloaded)
     - `NCAR RDA <https://rda.ucar.edu/datasets/d551000/dataaccess/>`_
   * - GloFAS
     - River Forcing (discharge)
     - Yes
     - `Copernicus CDS <https://cds.climate.copernicus.eu/datasets/cems-glofas-historical>`_
   * - RIVR2O
     - River Forcing (BGC tracers)
     - Yes
     - `Zenodo <https://zenodo.org/records/14032712>`_
   * - River tracer defaults
     - River Forcing (BGC fill values)
     - No (auto-downloaded)
     - roms-tools-data repository
   * - WOA23 BGC
     - BGC Initial & Boundary Conditions (nutrients and oxygen)
     - No (auto-downloaded)
     - `NCEI <https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/>`_
   * - WOA
     - Surface Restoring Forcing, sea surface salinity (`sss`)
     - Yes
     - `WOA, NOAA <https://www.ncei.noaa.gov/products/world-ocean-atlas>`_
   * - MBL_co2
     - Time-varying CO2, Surface Forcing
     - No (auto-downloaded)
     - `MBL, GML, NOAA <https://gml.noaa.gov/ccgg/mbl/data.php>`_
   * - SODA
     - Surface Restoring Forcing, sea surface salinity (`sDIC`, `sALK`)
     - No (auto-downloaded)
     - `OceanSODA-ETHZ, NOAA <https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0220059>`_


Manual Downloads
----------------

SRTM15
~~~~~~

High-resolution 1/240° global topography dataset. As an alternative, ROMS-Tools can use ETOPO5 at coarser 1/12° resolution, which is downloaded automatically.

:Version: V2.6
:Required for: Grid (Topography)
:Available at: `UCSD SRTM15+ Product <https://topex.ucsd.edu/WWW_html/srtm15_plus.html>`_

.. dropdown:: Required fields

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``lat``
        - Latitude (degrees north)
      * - ``lon``
        - Longitude (degrees east)
      * - ``z``
        - Topography (m)


EMODnet
~~~~~~~

European high-resolution bathymetry dataset. An alternative to SRTM15, particularly well-suited for domains in European waters.

:Required for: Grid (Topography)
:Available at: `EMODnet Digital Bathymetry <https://doi.org/10.12770/ff3aff8a-cff1-44a3-a2c8-1910bf109f85>`_

.. dropdown:: Required fields

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``lat``
        - Latitude (degrees north)
      * - ``lon``
        - Longitude (degrees east)
      * - ``elevation``
        - Bathymetry/topography (m)


ETOPO2022
~~~~~~~~~

High-resolution global relief model at up to 15 arc-second resolution, the successor to ETOPO5. An alternative to SRTM15 and EMODnet, providing combined topography and bathymetry worldwide. Unlike ETOPO5, ROMS-Tools does not download ETOPO2022 automatically.

:Required for: Grid (Topography)
:Available at: `NOAA ETOPO 2022 Global Relief Model <https://www.ncei.noaa.gov/products/etopo-global-relief-model>`_

.. dropdown:: Required fields

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``lat``
        - Latitude (degrees north)
      * - ``lon``
        - Longitude (degrees east)
      * - ``z``
        - Topography (m)


GSHHG
~~~~~

Global coastline shapefiles, provided at five resolutions:

* f (full): original highest-detail dataset
* h (high): ~80% reduction in detail and file size
* i (intermediate): another ~80% reduction
* l (low): another ~80% reduction
* c (crude): another ~80% reduction

The full-resolution (f) dataset is recommended for accurate representation of fjords, narrow straits, and other complex coastal geometries.

Alternatively, ROMS-Tools supports Natural Earth coastlines for land-sea mask generation, which are downloaded automatically.

:Version: V2.3.7
:Required for: Grid (Land-Sea Mask)
:Available at: `GSHHG Product <https://www.soest.hawaii.edu/pwessel/gshhg/>`_

For download instructions see :doc:`datasets`.

.. dropdown:: Required fields

   Coastline polygons/lines (Level-1 shapefiles: ``GSHHS_*_L1.shp`` and companion files).


TPXO
~~~~

Global barotropic tidal model providing tidal potential, elevation, and velocities including self-attraction and loading (SAL) corrections.

:Supported versions: TPXO9v5a, TPXO10v2, TPXO10v2a (all 1/6°)
:Required for: Tidal Forcing
:Available at: `OSU TPXO Tide Models <https://www.tpxo.net/global>`_

.. dropdown:: Required fields

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``lat_z``, ``lon_z``
        - Latitude/longitude of z nodes
      * - ``lat_u``, ``lon_u``
        - Latitude/longitude of u nodes
      * - ``lat_v``, ``lon_v``
        - Latitude/longitude of v nodes
      * - ``mz``, ``mu``, ``mv``
        - Water/land mask for z, u, v nodes
      * - ``hRe``, ``hIm``
        - Tidal elevation, real and imaginary parts (m)
      * - ``URe``, ``UIm``
        - Tidal transport WE, real and imaginary parts (m²/s)
      * - ``VRe``, ``VIm``
        - Tidal transport SN, real and imaginary parts (m²/s)


GLORYS
~~~~~~

1/12° global ocean physics reanalysis providing physical ocean initial and boundary conditions.

:Required for: Initial Conditions, Boundary Forcing
:Available at: `Copernicus Marine Data Store <https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description>`_

For download instructions see :doc:`datasets`.

.. dropdown:: Required fields

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``time``
        - Time
      * - ``latitude``
        - Latitude (degrees north)
      * - ``longitude``
        - Longitude (degrees east)
      * - ``depth``
        - Depth (m)
      * - ``zos``
        - Sea surface height (m)
      * - ``thetao``
        - Temperature (°C)
      * - ``so``
        - Salinity (psu)
      * - ``uo``
        - Eastward velocity (m/s)
      * - ``vo``
        - Northward velocity (m/s)


Unified BGC Dataset
~~~~~~~~~~~~~~~~~~~

A monthly biogeochemical climatology at 1° horizontal resolution, integrating multiple observational and model-based sources: World Ocean Atlas 2023 nutrients, oxygen, temperature and salinity; GLODAPv2 carbon chemistry; in-situ iron and nitrous oxide reconstructions; and CESM model output for the remaining nutrients and dissolved organic matter. It also carries the surface deposition fluxes (dust, iron, NOx, NHy) used for BGC surface forcing.

The file is filled across land and below the seafloor, so it has no missing values at any depth level, and ``ROMS-Tools`` applies no lateral fill to it before regridding.

Use version 2.1 or later (``BGCdataset_v2_1.nc``). Earlier versions are still read, with a warning: they name their dimensions ``lon``/``lat``/``dep``, and the oldest ones lack the ``temp_WOA``/``salt_WOA`` fields required for density-space BGC interpolation and salinity restoring.

:Required for: BGC Initial Conditions, BGC Boundary Forcing, BGC Surface Forcing, Surface Forcing (Restoring Forces; Salinity)
:Available at: `Google Drive <https://drive.google.com/uc?id=1NKbAe1ARtU68Np3bcwdd7nadeEUgdcef>`_

For download instructions see :doc:`datasets`.


WOA23 BGC Dataset
~~~~~~~~~~~~~~~~~

The World Ocean Atlas 2023 nutrients and oxygen, as a gridded BGC source for initial
conditions and boundary forcing. Unlike the unified dataset (which repackages WOA among
other sources into a single file), this reads the WOA23 files directly from NCEI.

**It supplies only** ``NO3``, ``PO4``, ``SiO3`` and ``O2``. WOA measures no carbon
chemistry or iron, so a MARBL run needs a companion source for ``DIC``, ``ALK`` and
``Fe`` -- combine them through the ``bgc_sources`` list, e.g. WOA for the nutrients
alongside GLODAP or ESPER for the carbon system.

Temperature and salinity are read too, but never written to ROMS output. They serve two
purposes: converting the tracers from µmol kg⁻¹ to mmol m⁻³ via TEOS-10 sigma-0 density,
and providing the source density coordinate for ``density`` / ``density_mld`` BGC
interpolation. The monthly T/S fields are paired with the monthly tracer fields.

.. note::
   In the NCEI path layout, the ``decav`` / ``all`` token is the averaging period *over
   years* (``decav`` pools 1955-2022 for T/S; ``all`` is the equivalent all-years token
   for nutrients and oxygen, 1965-2022). The **month** is the two-digit filename suffix:
   ``01``-``12`` monthly, ``00`` annual. So ``woa23_decav_t01_01.nc`` is the January
   temperature climatology, not an annual mean.

**Resolution.** WOA23 publishes nutrients and oxygen on the 1° grid only -- the 0.25°
product covers temperature and salinity alone -- so this source is 1° throughout,
including its T/S.

**Depth.** Monthly WOA fields are shallow: 800 m (43 levels) for the nutrients and
1500 m (57 levels) for oxygen and T/S. Only the annual field is full-depth (5500 m,
102 levels). Because the shallow axes are exact leading slices of the annual axis, each
variable is extended onto the full grid without vertical interpolation, controlled by
``deep_fill`` in the source dictionary:

``"annual_blend"`` (default)
    Splices the full-depth annual climatology underneath the monthly data, with a linear
    taper across the seam. The band is centred on *each variable's own* deepest monthly
    level, with a half-width of ``deep_blend_halfwidth`` (default 100 m) -- so 700-900 m
    for the nutrients and 1400-1600 m for oxygen and T/S. Above the band the field is
    purely monthly; below it, purely annual.

``"ffill"``
    Persists the deepest monthly value to the seafloor. Simpler, and it avoids
    downloading the annual files, but it badly misrepresents the deep ocean -- most
    obviously for oxygen, whose minimum near 800-1000 m does not persist downward.

Example::

    InitialConditions(
        grid=grid,
        ini_time=ini_time,
        source={"name": "GLORYS", "path": glorys_path},
        bgc_sources=[
            {"source": {"name": "WOA"}},                       # auto-downloads
            {"source": {"name": "GLODAP", "path": glodap_dir},  # DIC / ALK
             "use_vars": ["DIC", "ALK"]},
        ],
        bgc_model=BGCMarbl,
        bgc_interpolation_method="density_mld",
    )

Omitting ``path`` downloads the files to the ``roms-tools`` cache on first use. Passing
a ``path`` points at a directory that already holds them (this is what ``C-Star Forge``
stages). ``climatology`` defaults to ``True``, since WOA exists only as a twelve-month
climatology.

:Required for: BGC Initial Conditions, BGC Boundary Forcing
:Available at: `NCEI <https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/>`_


WOA Salinity Data
~~~~~~~~~~~~~~~~~

A collection of salinity (and other variables) means based on profile data from the World Ocean Database (WOD). The `s_an` variable provided is the 'Objectively analyzed mean fields for sea_water_salinity'.

:Required for: Surface Forcing (Restoring Forces; Salinity)
:Available at: `NOAA website <https://www.ncei.noaa.gov/products/world-ocean-atlas>`_

For download instructions see :doc:`datasets`.

.. dropdown:: Required fields

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``time``
        - Time
      * - ``lat``
        - Latitude (degrees north)
      * - ``lon``
        - Longitude (degrees east)
      * - ``depth``
        - Depth (m)
      * - ``s_an``
        - Objectively analyzed mean sea_water_salinity (PSU)


Automatically Accessed
----------------------

ETOPO5
~~~~~~

Global 1/12° topography dataset. Downloaded automatically by ROMS-Tools as a coarser alternative to SRTM15.

:Required for: Grid (Topography)
:Available at: `NOAA ETOPO Global Relief Model <https://www.ncei.noaa.gov/products/etopo-global-relief-model>`_

.. dropdown:: Required fields

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``lat``
        - Latitude (degrees north)
      * - ``lon``
        - Longitude (degrees east)
      * - ``topo``
        - Topography (m)


Natural Earth
~~~~~~~~~~~~~

1:10m coastline dataset, used to generate the land-sea mask. The label 1:10m refers to a map scale of 1:10,000,000, which corresponds to an effective spatial resolution of approximately 1–5 km. Accessed automatically via the ``regionmask`` package — no file download required.

:Required for: Grid (Land-Sea Mask)
:Available at: `Natural Earth <https://www.naturalearthdata.com/>`_


ERA5
~~~~

Global 1/4° atmospheric reanalysis from ECMWF providing meteorological surface forcing. ROMS-Tools can stream ERA5 data directly from the cloud, so downloading is optional.

:Required for: Surface Forcing
:Available at: `Copernicus Climate Data Store <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>`_

.. dropdown:: Required fields

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``time``
        - Time
      * - ``latitude``
        - Latitude (degrees north)
      * - ``longitude``
        - Longitude (degrees east)
      * - ``u10``
        - 10 m U wind component (m/s)
      * - ``v10``
        - 10 m V wind component (m/s)
      * - ``ssr``
        - Surface net short-wave radiation (W/m²)
      * - ``strd``
        - Surface long-wave radiation downwards (W/m²)
      * - ``t2m``
        - 2 m temperature (K)
      * - ``d2m``
        - 2 m dewpoint temperature (K)
      * - ``tp``
        - Total precipitation (m)
      * - ``sst``
        - Sea surface temperature (K) — used for land masking


Dai & Trenberth
~~~~~~~~~~~~~~~

Monthly coastal river discharge for ~1,000 of the world's largest rivers. Downloaded automatically by ROMS-Tools when ``source={"name": "DAI"}`` (the default).

:Version: May 2019 update
:Required for: River Forcing (discharge; default source)
:Available at: `NCAR RDA <https://rda.ucar.edu/datasets/d551000/dataaccess/>`_

.. dropdown:: Required fields

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``station``
        - Station index
      * - ``time``
        - Time (encoded as YYYYMM numeric values)
      * - ``lat_mou``
        - River mouth latitude
      * - ``lon_mou``
        - River mouth longitude
      * - ``FLOW``
        - Monthly mean volume at station (m³/s)
      * - ``ratio_m2s``
        - Ratio of volume between river mouth and station
      * - ``riv_name``
        - River name
      * - ``vol_stn`` (optional)
        - Annual volume at station; used to sort rivers by size


GloFAS
~~~~~~

Global daily river discharge from the Global Flood Awareness System (GloFAS) v4.0. Must be downloaded and preprocessed by the user before use. ROMS-Tools expects a NetCDF file in which river mouths have been placed on coastal cells using the GloFAS Large-scale Drainage Direction (LDD) algorithm. For a ready-to-run preprocessing workflow, see the :doc:`process_GloFAS` notebook.

:Version: v4.0
:Required for: River Forcing (discharge; ``source={"name": "GLOFAS", "path": ...}``)
:Available at: `Copernicus CDS <https://cds.climate.copernicus.eu/datasets/cems-glofas-historical>`_

.. dropdown:: Required fields

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``station``
        - Station index
      * - ``time``
        - Time (CF-compliant ``datetime64``)
      * - ``lat_mou``
        - River mouth latitude
      * - ``lon_mou``
        - River mouth longitude
      * - ``FLOW``
        - River discharge (m³/s)
      * - ``ratio_m2s``
        - Ratio of volume between river mouth and station
      * - ``riv_name``
        - River name
      * - ``vol_stn`` (optional)
        - Station volume metric; used to sort rivers by size


RIVR2O
~~~~~~

Global annual river biogeochemical export fields on a regular lat/lon grid (~0.5°). One NetCDF file per year; used as the dynamic BGC source when ``include_bgc=True`` and ``bgc_source={"name": "RIVR2O", "path": ...}``. MARBL tracers not provided by RIVR2O are filled from ``river_tracer_defaults.nc``.

:Coverage: 1903–2024
:Required for: River Forcing (BGC tracers; optional)
:Available at: `Zenodo <https://zenodo.org/records/14032712>`_

.. dropdown:: Required fields (per yearly file)

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``lat``, ``lon``
        - Regular lat/lon grid coordinates
      * - ``DIC``, ``DOC_l``, ``DOC_sl``, ``POC``
        - Carbon export (10⁶ g C yr⁻¹)
      * - ``DIN``, ``DIP``
        - Nitrogen and phosphorus export (renamed to ``NO3`` and ``PO4`` internally)


River tracer defaults
~~~~~~~~~~~~~~~~~~~~~

Recommended default MARBL river tracer concentrations. Downloaded automatically by ROMS-Tools. Used as the default BGC source (``bgc_source={"name": "CONSTANTS"}``) and as the fill source for tracers missing from RIVR2O.

:Required for: River Forcing (BGC; optional)
:Source: roms-tools-data repository (``river_tracer_defaults.nc``)


MBL_co2
~~~~~~~

Marine boundary layer values for CO2 (µmol mol⁻¹). Data are from a collection of NOAA's atmospheric sampling sites, and available about weekly.
Data are available for 1979 to 2025. Downloaded automatically by ROMS-Tools.

:Required for: Surface Forcing (time-varying CO₂)
:Available at: `NOAA's GML, MBL <https://gml.noaa.gov/ccgg/mbl/data.php>`_


OceanSODA
~~~~~~~~~~~~~~~

A global gridded marine carbonate system dataset calculated from machine learning estimates of Total Alkalinity and the fugacity of carbon dioxide. Data taken from NOAA's OceanSODA-ETHZ version 2025. Monthly data for years 1982-2024 at 1 degree resolution for the surface water. Downloaded automatically by ROMS-Tools.

:Version: 2025
:Required for: Surface Forcing (Restoring Forces; DIC & ALK)
:Available at: `NOAA's OceanSODA <https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0220059>`_


Download Instructions
---------------------

.. toctree::
   :maxdepth: 1

   datasets
   datasets_read
