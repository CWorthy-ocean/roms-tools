.. roms-tools documentation master file, created by
   sphinx-quickstart on Fri Jun  7 15:20:27 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the ROMS-Tools Documentation!
========================================

**ROMS-Tools** is a Python package designed for creating the input files necessary
to run a `ROMS <https://github.com/CESR-lab/ucla-roms>`_ simulation. This tool simplifies the process of generating:

- **Grid**
- **Tidal forcing**
- **Surface forcing**:

  - **Physical forcing**: wind, radiation, etc.
  - **Biogeochemical forcing**: atmospheric pCO₂, etc.

- **Initial conditions**:

  - **Physical conditions**: temperature, velocities, etc.
  - **Biogeochemical conditions**: alkalinity, etc.

- **Boundary forcing**:

  - **Physical forcing**: temperature, velocities, etc.
  - **Biogeochemical forcing**: alkalinity, etc.

- **River forcing**:

  - **Physical forcing**: river volume flux, river temperature, river salinity

Currently, **ROMS-Tools** does **not** support:

- Nesting

.. important::

   **DISCLAIMER**: This project is **ALPHA** and will be **EXPERIMENTAL** at least throughout 2024. We do not recommend using this software before this ALPHA development and test phase is completed.

This Python package is inspired by the `UCLA MATLAB tools <https://github.com/nmolem/ucla-tools/tree/main>`_.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   methods
   datasets

.. toctree::
   :maxdepth: 1
   :caption: Examples

   Creating a grid <grid>
   Creating tidal forcing <tides>
   Creating surface forcing <surface_forcing>
   Creating initial conditions <initial_conditions>
   Creating boundary forcing <boundary_forcing>
   Creating river forcing <river_forcing>

   Partitioning the input files <partition>

.. toctree::
   :maxdepth: 1
   :caption: Contributor Guide

   contributing

.. toctree::
   :maxdepth: 1
   :caption: References

   references
   api
