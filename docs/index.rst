.. roms-tools documentation master file, created by
   sphinx-quickstart on Fri Jun  7 15:20:27 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://img.shields.io/conda/vn/conda-forge/roms-tools.svg
   :target: https://anaconda.org/conda-forge/roms-tools

.. image:: https://img.shields.io/pypi/v/roms-tools.svg
   :target: https://pypi.org/project/roms-tools/

.. image:: https://github.com/CWorthy-ocean/roms-tools/actions/workflows/tests.yaml/badge.svg
   :target: https://github.com/CWorthy-ocean/roms-tools/actions/workflows/tests.yaml?query=branch%3Amain

.. image:: https://codecov.io/gh/CWorthy-ocean/roms-tools/graph/badge.svg?token=5S1oNu39xE
   :target: https://codecov.io/gh/CWorthy-ocean/roms-tools

.. image:: https://readthedocs.org/projects/roms-tools/badge/?version=latest
   :target: https://roms-tools.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/pypi/pyversions/roms-tools
   :target: https://img.shields.io/pypi/pyversions/roms-tools

.. image:: https://static.pepy.tech/personalized-badge/roms-tools?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads
   :target: https://pepy.tech/projects/roms-tools


ROMS-Tools: Reproducible Preprocessing and Analysis for Regional Ocean Modeling with ROMS
==========================================================================================

**ROMS-Tools** is a Python package for **preparing and analyzing ROMS simulations**, with optional `MARBL biogeochemistry (BGC) <https://marbl-ecosys.github.io/versions/latest_release/index.html>`_ support.

``ROMS-Tools`` enables users to generate regional grids, prepare model inputs, and analyze outputs through a modern, user-friendly interface that standardizes common workflows and reduces data-preparation overhead. The package is designed for reproducible research, with YAML-based configuration, optional ``dask`` parallelization, interactive Jupyter support, and CI-tested reliability with comprehensive documentation.

Current capabilities are fully compatible with UCLA-ROMS :cite:`ucla-roms,ucla-roms-cworthy`, with potential support for other ROMS implementations, such as Rutgers ROMS :cite:`rutgers-roms`, in the future.


Overview of ROMS-Tools Functionality
--------------------------------------

``ROMS-Tools`` provides a comprehensive workflow for generating, processing, and analyzing ROMS-MARBL model inputs and outputs, as detailed below.

Input Data and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Built on ``xarray`` and optionally powered by ``dask``, ``ROMS-Tools`` automates the generation of all major ROMS–MARBL inputs, including:

1. **Model Grid**: Customizable, curvilinear, and orthogonal grid designed to maintain a nearly uniform horizontal resolution across the domain. The grid is rotatable to align with coastlines and features a terrain-following vertical coordinate.

2. **Bathymetry**: Derived from **SRTM15** :cite:`tozer_global_2019`.

3. **Land Mask**: Inferred from coastlines provided by **Natural Earth** or the Global Self-consistent, Hierarchical, High-resolution Geography (**GSHHG**) Database :cite:`wessel_global_1996`.

4. **Physical Ocean Conditions**: Initial and open boundary conditions for sea surface height, temperature, salinity, and velocities derived from the 1/12° Global Ocean Physics Reanalysis (**GLORYS**) :cite:`jean-michel_copernicus_2021`.

5. **BGC Ocean Conditions**: Initial and open boundary conditions for dissolved inorganic carbon, alkalinity, and other biogeochemical tracers from Community Earth System Model (**CESM**) output :cite:`yeager_2022` or hybrid observational-model sources :cite:`garcia2019woa,lauvset_new_2016,huang_data-driven_2022,yang_global_2020,yeager_2022`.

6. **Meteorological forcing**: Wind, radiation, precipitation, and air temperature/humidity processed from the global 1/4° ECMWF Reanalysis v5 (**ERA5**) :cite:`hersbach_era5_2020` with optional corrections for radiation bias and coastal wind.

7. **BGC surface forcing**: Partial pressure of carbon dioxide, as well as iron, dust, and nitrogen deposition from **CESM** output :cite:`yeager_2022` or hybrid observational-model sources :cite:`landschutzer_decadal_2016,kok_improved_2021,hamilton_earth_2022,yeager_2022`.

8. **Tidal Forcing**: Tidal potential, elevation, and velocities derived from **TPXO** :cite:`egbert_efficient_2002` including self-attraction and loading (SAL) corrections.

9. **River Forcing**: Freshwater runoff derived from **Dai & Trenberth** :cite:`dai_estimates_2002` or user-provided custom files.

10. **CDR Forcing**: User-defined interventions that inject BGC tracers at point sources or as larger-scale Gaussian perturbations to simulate CDR interventions. The CDR forcing is prescribed as volume and tracer fluxes (e.g., alkalinity for ocean alkalinity enhancement, iron for iron fertilization, or other BGC constituents). Users can control the magnitude, spatial footprint, and temporal evolution, allowing flexible representation of CDR interventions.

11. **Nesting**: Support for creating nested grids and parent-child configurations.

Some source datasets are accessed automatically by ``ROMS-Tools``, including Natural Earth, Dai & Trenberth runoff, and ERA5 meteorology, while users must manually download SRTM15, GSHHG, GLORYS, the BGC datasets, and TPXO tidal files. Although these are the datasets currently supported, the modular design of ``ROMS-Tools`` makes it straightforward to add new source datasets in the future.

To generate the model inputs, ``ROMS-Tools`` automates several intermediate processing steps, including:

* **Bathymetry processing**: The bathymetry is smoothed in two stages, first across the entire model domain and then locally in areas with steep slopes, to ensure local steepness ratios do not exceed a prescribed threshold in order to reduce pressure-gradient errors. A minimum depth is enforced to prevent water levels from becoming negative during large tidal excursions.

* **Mask definition**: The land-sea mask is generated by comparing the ROMS grid’s horizontal coordinates with a coastline dataset using the ``regionmask`` package :cite:`hauser_regionmaskregionmask_2024`. Enclosed basins are subsequently filled with land.

* **Land value handling**: Land values are filled via an algebraic multigrid method using ``pyamg`` :cite:`pyamg2023` prior to horizontal regridding. This extends ocean values into land areas to reconcile discrepancies between source data and ROMS land masks, ensuring that no NaNs or land-originating values contaminate ocean grid cells.

* **Regridding**: Ocean and atmospheric fields are horizontally and vertically regridded from standard latitude-longitude-depth grids to the model’s curvilinear grid with a terrain-following vertical coordinate using ``xarray`` :cite:`hoyer2017xarray` and ``xgcm`` :cite:`xgcm`. Velocities are rotated to align with the curvilinear ROMS grid.

* **Longitude conventions**: ``ROMS-Tools`` handles differences in longitude conventions, converting between [-180°, 180°] and [0°, 360°] as needed.

* **River locations**: Rivers that fall within the model domain are automatically identified and relocated to the nearest coastal grid cell. Rivers that need to be shifted manually or span multiple cells can be configured by the user.

* **Data streaming**: ERA5 atmospheric data can be accessed directly from the cloud, removing the need for users to pre-download large datasets locally. Similar streaming capabilities may be implemented for other datasets in the future.

Users can quickly design and visualize regional grids and inspect all input fields with built-in plotting utilities.

Postprocessing and Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ROMS-Tools`` supports postprocessing and analysis of ROMS-MARBL output, including regridding from the native curvilinear, terrain-following grid to a standard latitude-longitude-depth grid using ``xesmf`` :cite:`xesmf`, with built-in plotting for both grid types. The analysis layer also includes specialized utilities for evaluating carbon dioxide removal (CDR) interventions, such as generating carbon uptake and efficiency curves.


Getting Started
---------------------------

.. toctree::
   :maxdepth: 1

   installation
   methods
   datasets
   End-to-end workflow (laptop) <end_to_end>


Preparing a ROMS Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following examples cover task-specific workflows for the Perlmutter and Anvil supercomputers. Pre-download data and adjust paths as needed. For a full end-to-end workflow designed to run on a laptop, refer to the “End-to-end workflow (laptop)” section above.

.. toctree::
   :maxdepth: 1

   Creating a grid <grid>
   Creating tidal forcing <tides>
   Creating surface forcing <surface_forcing>
   Creating initial conditions <initial_conditions>
   Creating boundary forcing <boundary_forcing>
   Creating river forcing <river_forcing>
   Creating carbon dioxide removal forcing <cdr_forcing>
   Preparing nested simulations <nesting>
   Partitioning the input files <partition>

Analyzing a ROMS Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following examples cover task-specific workflows for the Perlmutter and Anvil supercomputers. Pre-download data and adjust paths as needed. For a full end-to-end workflow designed to run on a laptop, refer to the “End-to-end workflow (laptop)” section above.

.. toctree::
   :maxdepth: 1

   Joining ROMS output files <join>
   Reading ROMS Output <reading_roms_output>
   Visualizing ROMS Output <plotting_roms_output>
   Regridding ROMS Output <regridding_roms_output>
   Analyzing Carbon Dioxide Removal (CDR) Metrics <cdr_analysis>


Advanced Topics
---------------------------
.. toctree::
   :maxdepth: 1

   Using ROMS-Tools with Dask <using_dask>


For Developers
---------------------------
.. toctree::
   :maxdepth: 1

   contributing
   releases

References
---------------------------
.. toctree::
   :maxdepth: 1

   references
   api
