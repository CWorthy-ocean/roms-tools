.. roms-tools documentation master file, created by
   sphinx-quickstart on Fri Jun  7 15:20:27 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the ROMS-Tools documentation!
========================================

**ROMS-Tools** is a python package for creating the input files that are necessary
to run a `ROMS <https://github.com/CESR-lab/ucla-roms>`_ simulation.
This includes creating a grid, tidal, boundary, and atmospheric forcings, initial conditions, and more!
This python package is strongly inspired by the `UCLA MATLAB tools <https://github.com/nmolem/ucla-tools/tree/main>`_.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   methods

.. toctree::
   :maxdepth: 1
   :caption: Examples

   Creating a grid <grid>
   Creating tidal forcing <tides>
   Creating atmospheric forcing <atmospheric_forcing>
   Creating a vertical coordinate <vertical_coordinate>
   Creating initial conditions <initial_conditions>
   Creating boundary forcing <boundary_forcing>

.. toctree::
   :maxdepth: 1
   :caption: Community

   support

.. toctree::
   :maxdepth: 1
   :caption: References

   references
   api
