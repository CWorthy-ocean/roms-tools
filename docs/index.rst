.. roms-tools documentation master file, created by
   sphinx-quickstart on Fri Jun  7 15:20:27 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


ROMS-Tools
===========

**ROMS-Tools** is a Python package for **preparing and analyzing** reproducible simulations with `ROMS <https://cworthy-ucla-roms.readthedocs.io/en/latest/index.html>`_, optionally coupled to `MARBL biogeochemistry (BGC) <https://marbl-ecosys.github.io/versions/latest_release/index.html>`_.


Why ROMS-Tools?
---------------

Configuring a regional ocean model like ROMS-MARBL is technically demanding. Model setup requires initialization and time-dependent forcing from oceanic and atmospheric datasets drawn from multiple external sources in diverse formats. These global source datasets can span petabytes and must be subsetted, processed, and mapped onto the target model grid — a workflow that is time-consuming, error-prone, and difficult to reproduce.

``ROMS-Tools`` addresses this bottleneck by automating the full ROMS-MARBL preprocessing pipeline. The package builds on foundational preprocessing work established by the UCLA MATLAB scripts :cite:`ucla-matlab` and ``pyroms`` :cite:`pyroms`, which encode decades of expertise in ROMS model setup. While these tools have long served the community well, ``ROMS-Tools`` represents a modern Python framework designed to meet contemporary demands. Analogous frameworks have emerged for other regional models, for example ``regional-mom6`` :cite:`barnes_regional-mom6_2024` for MOM6, reflecting a broader community shift toward standardized, reproducible workflows.

``ROMS-Tools`` streamlines model configuration through high-level APIs and YAML-based configuration that makes simulation setups reproducible. Optional ``dask`` integration enables parallelization on HPC clusters, and built-in visualization supports quick inspection of grids and model fields at every stage. Current capabilities are fully compatible with UCLA-ROMS :cite:`ucla-roms,ucla-roms-cworthy`, with potential support for other ROMS implementations, such as Rutgers ROMS :cite:`rutgers-roms`, in the future.

.. grid:: 1 2 2 2
   :gutter: 3
   :padding: 0

   .. grid-item-card:: Features
      :link: features
      :link-type: doc

      :octicon:`checklist;2em;sd-text-primary`

      Overview of all model inputs and processing capabilities supported by ROMS-Tools.

   .. grid-item-card:: Methods
      :link: methods
      :link-type: doc

      :octicon:`beaker;2em;sd-text-primary`

      Algorithmic details behind grid generation, regridding, masking, and bathymetry processing.


Getting Started
---------------

.. grid:: 1 2 2 3
   :gutter: 3
   :padding: 0

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      :octicon:`desktop-download;2em;sd-text-primary`

      Install ROMS-Tools and set up your environment.

   .. grid-item-card:: Datasets
      :link: datasets_overview
      :link-type: doc

      :octicon:`stack;2em;sd-text-primary`

      Which datasets ROMS-Tools requires, where to download them, and what fields are needed.

   .. grid-item-card:: Examples
      :link: examples
      :link-type: doc

      :octicon:`book;2em;sd-text-primary`

      Step-by-step workflow examples for preparing and analyzing ROMS simulations.


.. toctree::
   :hidden:
   :maxdepth: 1

   features
   methods
   installation
   datasets_overview
   examples
   contributing
   releases
   references
   api

For Developers
--------------

- :doc:`contributing`
- :doc:`releases`


References
----------

- :doc:`references`
- :doc:`api`


