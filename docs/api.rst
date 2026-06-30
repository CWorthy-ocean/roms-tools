API Reference
#############


Grid
------------------------

.. autosummary::
   :toctree: generated/

   roms_tools.Grid

Tidal Forcing
------------------

.. autosummary::
   :toctree: generated/

   roms_tools.TidalForcing

Surface Forcing
----------------

.. autosummary::
   :toctree: generated/

   roms_tools.SurfaceForcing

Initial Conditions
--------------------

.. autosummary::
   :toctree: generated/

   roms_tools.InitialConditions

Boundary Forcing
--------------------

.. autosummary::
   :toctree: generated/

   roms_tools.BoundaryForcing

River Forcing
--------------------

.. autosummary::
   :toctree: generated/

   roms_tools.RiverForcing
   roms_tools.datasets.river_datasets.DaiRiverDataset
   roms_tools.datasets.river_datasets.GloFASRiverDataset

CDR Forcing
--------------------

.. autosummary::
   :toctree: generated/

   roms_tools.VolumeRelease
   roms_tools.TracerPerturbation
   roms_tools.CDRForcing

Analyzing ROMS output
----------------------

.. autosummary::
   :toctree: generated/

   roms_tools.ROMSOutput
   roms_tools.ROMSOutput.plot
   roms_tools.ROMSOutput.create_movie
   roms_tools.ROMSOutput.regrid
   roms_tools.ROMSOutput.cdr_metrics
   roms_tools.Ensemble

=======

Utilities
---------

.. autosummary::
   :toctree: generated/

   roms_tools.setup.nesting.align_grids
   roms_tools.setup.nesting.make_nesting_info
   roms_tools.tiling.partition.partition_netcdf
   roms_tools.tiling.join.join_netcdf
   roms_tools.plot.plot
   roms_tools.datasets.lat_lon_datasets.get_glorys_bounds
   roms_tools.setup.utils.compute_potential_density
   roms_tools.setup.utils.compute_mld
