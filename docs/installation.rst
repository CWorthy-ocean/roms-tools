Installation
############

Installation from pip
=====================

roms-tools can be installed using pip::

    pip install roms-tools


Installation from GitHub
========================

To obtain the latest development version, clone
`the repository <https://github.com/CWorthy-ocean/roms-tools.git>`_
and install it as follows::

    git clone https://github.com/CWorthy-ocean/roms-tools.git
    cd roms-tools
    pip install -e . --no-deps

Running the tests
=================

Check the installation has worked by running the tests::

    pytest

Dependencies
============
Dependencies required are xarray, scipy, netcdf4, and pooch, plus matplotlib and cartopy for visualising grids.

roms-tools should run on any platform that can install the above dependencies.

You can set up a conda environment with all required dependencies as follows::

    cd roms-tools
    conda env create -f ci/environment.yml
    conda activate romstools-test



