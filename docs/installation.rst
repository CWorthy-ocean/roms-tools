Installation
############

Installation from Conda-Forge
=============================

To install ``ROMS-Tools`` with all dependencies, including ``xesmf``, ``dask`` and all packages required for streaming source data directly from the cloud, use::

    conda install -c conda-forge roms-tools

Installation from Conda-Forge is the recommended installation method to ensure all features of ``ROMS-Tools`` are available.


Installation from PyPI (pip)
============================

``ROMS-Tools`` can be installed using pip::

    pip install roms-tools

If you want to use ``ROMS-Tools`` together with ``dask`` (which we recommend for parallel and out-of-core computation), you can install ``ROMS-Tools`` along with the additional dependency via::

    pip install roms-tools[dask]

If you want to use ``ROMS-Tools`` with ``dask`` and all packages required for streaming source data directly from the cloud, install it with the additional dependencies via::

    pip install roms-tools[stream]


Note: The PyPI versions of ``ROMS-Tools`` do not include ``xesmf``, so some features will be unavailable.


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

If you want to use ``ROMS-Tools`` together with ``dask`` and all packages required for
streaming source data directly from the cloud, you can
install ``ROMS-Tools`` along with the additional dependencies via::

    pip install ".[stream]"
