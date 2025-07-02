.. roms-tools documentation master file, created by
   sphinx-quickstart on Fri Jun  7 15:20:27 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the ROMS-Tools Documentation!
========================================

**ROMS-Tools** is a Python package designed for creating the input files necessary to run a `UCLA-ROMS <https://github.com/CESR-lab/ucla-roms>`_ simulation, with or without `MARBL biogeochemistry <https://marbl-ecosys.github.io/versions/latest_release/index.html>`_.


The package is designed with the following goals in mind:

- **Automation** of complex preprocessing steps
- **Intuitive usability** for new and experienced users
- **Reproducibility** through configuration-based workflows
- **Code efficiency** with support for parallel and lazy evaluation
- **Commitment to best software practices**, including testing and documentation

ROMS-Tools streamlines the creation of the following inputs:

- **Grid**:

  - Coordinates and metrics
  - Bathymetry (derived from SRTM15)
  - Land-sea mask (based on Natural Earth datasets)

- **Tidal forcing**:

  - Derived from TPXO tidal constituents

- **Surface forcing**:

  - Physical/Meteorological forcing: wind, radiation, etc. (from ERA5)
  - Biogeochemical forcing: atmospheric pCO₂, etc. (from CESM or hybrid observational/model sources)

- **Initial conditions**:

  - Physical conditions: temperature, velocities, etc. (from GLORYS)
  - Biogeochemical conditions: alkalinity, etc. (from CESM or hybrid observational/model sources)

- **Boundary forcing**:

  - Physical forcing: temperature, velocities, etc. (from GLORYS)
  - Biogeochemical forcing: alkalinity, etc. (from CESM or hybrid observational/model sources)

- **River forcing**:

  - Physical forcing: river volume flux, river temperature, river salinity (from Dai and Trenberth, 2019, or a custom river dataset)

- **Carbon Dioxide Removal (CDR) forcing**

- **Nesting**

In addition to input generation, ROMS-Tools includes utilities for postprocessing and analysis, particularly for CDR monitoring, reporting, and verification (MRV).

This Python package is inspired by the `UCLA MATLAB tools <https://github.com/nmolem/ucla-tools/tree/main>`_.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   methods
   datasets

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

   Reading ROMS Output <reading_roms_output>
   Visualizing ROMS Output <plotting_roms_output>
   Regridding ROMS Output <regridding_roms_output>

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
