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

ROMS-Tools: Preparing and Analyzing ROMS Simulations
=========================================================================

**ROMS-Tools** is a Python package that automates the creation, management, and analysis of all input and output files needed for regional ocean simulations with `UCLA-ROMS <https://github.com/CESR-lab/ucla-roms>`_, with optional `MARBL biogeochemistry (BGC) <https://marbl-ecosys.github.io/versions/latest_release/index.html>`_.

Built on ``xarray`` and optionally powered by ``dask``, ``ROMS-Tools`` automates the generation of all major ROMS–MARBL inputs, including:

1. **Model Grid**: Customizable, curvilinear, and orthogonal grid designed to maintain a nearly uniform horizontal resolution across the domain. The grid is rotatable to align with coastlines and features a terrain-following vertical coordinate.
2. **Bathymetry**: Derived from **SRTM15**.
3. **Land Mask**: Inferred from coastlines provided by **Natural Earth** or **GSHHG**.
4. **Physical Ocean Conditions**:  Initial and open boundary conditions for sea surface height, temperature, salinity, and velocities derived from **GLORYS**.
5. **BGC Ocean Conditions**: Initial and open boundary conditions for dissolved inorganic carbon, alkalinity, and other BGC tracers from **CESM** output or hybrid observational-model sources.
6. **Meteorological forcing**: Wind, radiation, precipitation, and air temperature/humidity processed from **ERA5** with optional corrections for radiation bias and coastal wind.
7. **BGC surface forcing**: Partial pressure of carbon dioxide, as well as iron, dust, and nitrogen deposition from **CESM** output or hybrid observational-model sources.
8. **Tidal Forcing:** Tidal potential, elevation, and velocities derived from **TPXO** including self-attraction and loading (SAL) corrections.
9. **River Forcing:** Freshwater runoff derived from **Dai & Trenberth** or user-provided custom files.
10. **CDR Forcing**: User-defined interventions that inject BGC tracers at point sources or as larger-scale Gaussian perturbations, designed to simulate CDR interventions.
11. **Nesting**: Support for creating nested grids and parent-child configurations.

Beyond input generation, ``ROMS-Tools`` provides a suite of analysis and postprocessing utilities, including regridding fields to standard latitude-longitude-depth grids and performing specialized CDR-focused analysis.

ROMS-Tools supports modern, reproducible workflows with YAML-based configuration, cloud-accessible datasets, optional ``dask`` parallelization, interactive Jupyter usage, and CI-tested reliability with comprehensive documentation.

.. note::
  Below is a series of examples. The `end-to-end workflow <https://roms-tools.readthedocs.io/en/latest/end_to_end.html>`_ demonstrates using ``ROMS-Tools`` to prepare inputs and analyze outputs for a ROMS–MARBL simulation, and is designed to run on a laptop. Subsequent examples cover more detailed, task-specific workflows and are configured for the Perlmutter supercomputing system, so you may need to pre-download data and adjust paths when running them elsewhere.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   methods
   datasets
   end_to_end

.. toctree::
   :maxdepth: 1
   :caption: Preparing a ROMS Simulation

   Creating a grid <grid>
   Creating tidal forcing <tides>
   Creating surface forcing <surface_forcing>
   Creating initial conditions <initial_conditions>
   Creating boundary forcing <boundary_forcing>
   Creating river forcing <river_forcing>
   Creating carbon dioxide removal forcing <cdr_forcing>
   Preparing nested simulations <nesting>
   Partitioning the input files <partition>

.. toctree::
   :maxdepth: 1
   :caption: Analyzing a ROMS Simulation

   Joining ROMS output files <join>
   Reading ROMS Output <reading_roms_output>
   Visualizing ROMS Output <plotting_roms_output>
   Regridding ROMS Output <regridding_roms_output>
   Analyzing Carbon Dioxide Removal (CDR) Metrics <cdr_analysis>

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics

   Using ROMS-Tools with Dask <using_dask>


.. toctree::
   :maxdepth: 1
   :caption: For Developers

   contributing
   releases

.. toctree::
   :maxdepth: 1
   :caption: References

   references
   api
