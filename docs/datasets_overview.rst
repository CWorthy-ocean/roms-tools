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
     - BGC Initial, Boundary & Surface Forcing
     - Yes
     - `Google Drive <https://drive.google.com/uc?id=1wUNwVeJsd6yM7o-5kCx-vM3wGwlnGSiq>`_
   * - ERA5
     - Surface Forcing
     - Optional (streaming supported)
     - `Climate Data Store <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>`_
   * - Dai & Trenberth
     - River Forcing
     - No (auto-downloaded)
     - `NCAR RDA <https://rda.ucar.edu/datasets/d551000/dataaccess/>`_
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

A unified biogeochemical climatology integrating multiple observational and model-based sources, including World Ocean Atlas nutrients, GLODAPv2 carbon chemistry, and CESM model output.

:Required for: BGC Initial Conditions, BGC Boundary Forcing, BGC Surface Forcing
:Available at: `Google Drive <https://drive.google.com/uc?id=1wUNwVeJsd6yM7o-5kCx-vM3wGwlnGSiq>`_

For download instructions see :doc:`datasets`.


WOA Salinity Data
~~~~~~

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

Monthly coastal river discharge climatology. Downloaded automatically by ROMS-Tools.

:Version: 2019
:Required for: River Forcing
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
        - Time
      * - ``lat_mou``
        - River mouth latitude
      * - ``lon_mou``
        - River mouth longitude
      * - ``FLOW``
        - Monthly mean volume at station
      * - ``ratio_m2s``
        - Ratio of volume between river mouth and station
      * - ``riv_name``
        - River name
      * - ``vol_stn`` (optional)
        - Annual volume at station


MBL_co2
~~~~~~~~~~~~~~~

Marine boundary layer values for CO2 (µmol mol⁻¹). Data are from a collection of NOAA's atmospheric sampling sites, and available about weekly.
Data are available for 1979 to 2025. Downloaded automatically by ROMS-Tools.

:Version: 2019
:Required for: River Forcing
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
