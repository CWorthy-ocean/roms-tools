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

``dask`` (recommended for parallel and out-of-core computation) is included by default.

If you want to use ``ROMS-Tools`` with all packages required for streaming source data directly from the cloud, install it with the additional dependencies via::

    pip install roms-tools[stream]

If you want to use the :meth:`~roms_tools.ROMSOutput.create_movie` functionality to generate animations, install ``ROMS-Tools`` with the ``movie`` extra, which provides ``ffmpeg`` via the ``imageio-ffmpeg`` package::

    pip install roms-tools[movie]

If you already have ``ffmpeg`` installed (e.g. via conda or your system package manager), the ``[movie]`` extra is not needed.

To work through the example notebooks in a Jupyter session, the ``notebooks`` extra provides ``jupyter``, ``ipykernel``, and ``gdown`` (for downloading example data)::

    pip install roms-tools[notebooks]

(The conda-forge ``roms-tools`` package already includes these.)

Multiple extras can be combined. For example, to use both streaming and movie creation::

    pip install roms-tools[stream,movie]

Note: The PyPI versions of ``ROMS-Tools`` do not include ``xesmf``, so some features will be unavailable.


Installation from GitHub
========================

To obtain the latest development version, first clone
`the source repository <https://github.com/CWorthy-ocean/roms-tools.git>`_::

    git clone https://github.com/CWorthy-ocean/roms-tools.git
    cd roms-tools

Then, install and activate the following conda environment::

    conda env create -f ci/environment.yml
    conda activate roms-tools-env
    conda install -c conda-forge xesmf

Finally, install ``ROMS-Tools`` in the same environment, along with the ``dev`` extra
(test/lint tooling used by the test suite and pre-commit)::

    pip install -e ".[dev]"

``dask`` (recommended for parallel and out-of-core computation) is included in the core
dependencies by default.

If you want to use ``ROMS-Tools`` with all packages required for
streaming source data directly from the cloud, you can
install ``ROMS-Tools`` along with the additional dependencies via::

    pip install -e ".[dev,stream]"

If you want to use the :meth:`~roms_tools.ROMSOutput.create_movie` functionality and you
do not have ``ffmpeg`` installed by other means, install with the ``movie`` extra::

    pip install -e ".[dev,movie]"

Multiple extras can be combined, for example::

    pip install -e ".[dev,stream,movie]"

If you want to build the documentation locally, install the ``docs`` extra::

    pip install -e ".[docs]"
