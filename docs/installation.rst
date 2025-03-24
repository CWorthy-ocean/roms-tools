Installation
############

Installation from conda forge
=============================

ROMS-Tools can be installed via conda forge::

    conda install -c conda-forge roms-tools

This command installs ``ROMS-Tools`` along with its ``dask`` and ``xesmf`` dependencies.

Installation from pip
=====================

``ROMS-Tools`` can be installed using pip::

    pip install roms-tools

If you want to use ``ROMS-Tools`` together with ``dask`` (which we recommend for parallel and out-of-core computation), you can install ``ROMS-Tools`` along with the additional dependency via::

    pip install roms-tools[dask]

Note: The ``ROMS-Tools`` versions installed from pip do not include ``xesmf``, so some features will be unavailable.


Installation from GitHub
========================

To obtain the latest development version, first clone
`the source repository <https://github.com/CWorthy-ocean/roms-tools.git>`_::

    git clone https://github.com/CWorthy-ocean/roms-tools.git
    cd roms-tools

Then, install and activate the following conda environment::

    conda env create -f ci/environment-with-xesmf.yml
    conda activate romstools-test

Finally, install ``ROMS-Tools`` in the same environment::

    pip install -e .

If you want to use ``ROMS-Tools`` together with ``dask`` (which we recommend), you can
install ``ROMS-Tools`` along with the additional dependency via::

    pip install ".[dask]"
