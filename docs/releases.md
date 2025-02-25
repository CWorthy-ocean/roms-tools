# Release notes

## v2.5.0 (unreleased)

### New Features

### Breaking Changes

* Deprecate `type` parameter in `ROMSOutput` ([#253](https://github.com/CWorthy-ocean/roms-tools/pull/253))
* Write and read the parameter `bypass_validation` to/from YAML ([#249](https://github.com/CWorthy-ocean/roms-tools/pull/249))
* Refactor `Nesting` class and renamed it to `ChildGrid` class to ensure definite serialization ([#250](https://github.com/CWorthy-ocean/roms-tools/pull/250))

### Internal Changes

* Enforce double precision on source data to ensure reproducible results ([#244](https://github.com/CWorthy-ocean/roms-tools/pull/244))
* Results produced with vs. without Dask in test suite now pass with `xr.testing.assert_equal` confirming reproducibility ([#244](https://github.com/CWorthy-ocean/roms-tools/pull/244))
* Document the option for `start_time = None` and `end_time = None` in the docstrings for `BoundaryForcing` and `SurfaceForcing`, specifying that when both are `None`, no time filtering is applied to the data. Also, ensure a warning is raised in this case to inform the user. ([#249](https://github.com/CWorthy-ocean/roms-tools/pull/249))

### Documentation

* Documentation on how to use ROMS-Tools with Dask ([#245](https://github.com/CWorthy-ocean/roms-tools/pull/245))

### Bugfixes

## v2.4.0

### New Features

* Introduce new parameter `coarse_grid_mode` for `SurfaceForcing`. The default `coarse_grid_mode = "auto"` automatically decides whether it makes sense to interpolate onto the coarse grid, which saves computations ([#228](https://github.com/CWorthy-ocean/roms-tools/pull/228))
* New default for `correct_radiation` in `SurfaceForcing` is `True` ([#228](https://github.com/CWorthy-ocean/roms-tools/pull/228))
* New default for `bathymetry` in `Grid.plot()` is `True` ([#234](https://github.com/CWorthy-ocean/roms-tools/pull/234))
* New default for `group` in `SurfaceForcing.save()` and `BoundaryForcing.save()` is `True` ([#236](https://github.com/CWorthy-ocean/roms-tools/pull/236))
* Option to adjust depth for sea surface height when creating `InitialConditions` and `BoundaryForcing` ([#240](https://github.com/CWorthy-ocean/roms-tools/pull/240))
* New parameter `horizontal_chunk_size` for `InitialConditions`, which ensures the feasibility of processing initial conditions for large domains, both in terms of memory footprint and compute times ([#241](https://github.com/CWorthy-ocean/roms-tools/pull/241))

### Breaking Changes

* Remove support for partitioning files upon saving ([#221](https://github.com/CWorthy-ocean/roms-tools/pull/221))
* Remove parameter `use_coarse_grid` for `SurfaceForcing` class ([#228](https://github.com/CWorthy-ocean/roms-tools/pull/228))
* Remove parameter `filepath_grid` from `RiverForcing.save()` method ([#232](https://github.com/CWorthy-ocean/roms-tools/pull/232))

### Internal Changes

* Parallelize computation of radiation correction, leading to a hugely improved memory footprint for surface forcing generation ([#227](https://github.com/CWorthy-ocean/roms-tools/pull/227))
* For computation of radiation correction, swap order of temporal and spatial interpolation to further improve memory footprint ([#227](https://github.com/CWorthy-ocean/roms-tools/pull/227))
* Write river locations into river forcing file rather than into grid file ([#232](https://github.com/CWorthy-ocean/roms-tools/pull/232))
* When appropriate, only compare data hashes rather than file hashes in automated test suite ([#235](https://github.com/CWorthy-ocean/roms-tools/pull/235))
* Slightly shift one of the test grids away from the Greenwich meridian ([#235](https://github.com/CWorthy-ocean/roms-tools/pull/235))
* The partitioning functions are moved to their own subdirectory `tiling` ([#236](https://github.com/CWorthy-ocean/roms-tools/pull/236))
* Internal refactoring of depth coordinate computation ([#240](https://github.com/CWorthy-ocean/roms-tools/pull/240))

### Documentation

* New features and defaults are documented for `SurfaceForcing` (([#228](https://github.com/CWorthy-ocean/roms-tools/pull/228))
* Improvements to the notebook that documents the partioning functionality ([#236](https://github.com/CWorthy-ocean/roms-tools/pull/236))
* Document the option to adjust depth for sea surface height ([#240](https://github.com/CWorthy-ocean/roms-tools/pull/240))
* More realistic, higher-resolution domains in the example notebooks, which use the SRTM15 topography to discourage the user from employing the default ETOPO5 topography ([#241](https://github.com/CWorthy-ocean/roms-tools/pull/241))

### Bugfixes

* Fix bug in validation step for surface forcing ([#227](https://github.com/CWorthy-ocean/roms-tools/pull/227))

## v2.3.0

### New Features

* `ROMSOutput` class for analyzing ROMS model output ([#217](https://github.com/CWorthy-ocean/roms-tools/pull/217))

### Bugfixes

* Correctly handle list of files in `source` and `bgc_source` in YAML files ([#218](https://github.com/CWorthy-ocean/roms-tools/pull/218))

## v2.2.1

### Bugfixes

* Correctly write Pathlib `bgc_source` to YAML file ([#215](https://github.com/CWorthy-ocean/roms-tools/pull/215))

## v2.2.0

### Bugfixes

* Fix bug in validation of tidal forcing which led to memory blow-ups ([#211](https://github.com/CWorthy-ocean/roms-tools/pull/211))
* Make sure correct masks are used for validation ([#214](https://github.com/CWorthy-ocean/roms-tools/pull/214))
* Correctly write Pathlib objects to YAML file ([#212](https://github.com/CWorthy-ocean/roms-tools/pull/212))

## v2.1.0

### New Features

* Nesting capability ([#204](https://github.com/CWorthy-ocean/roms-tools/pull/204))
* Option to bypass validation ([#206](https://github.com/CWorthy-ocean/roms-tools/pull/206))
* Latitude and longitude labels are added to spatial plots ([#208](https://github.com/CWorthy-ocean/roms-tools/pull/208))

### Internal Changes

* Dirty tagging in the versioning has been removed ([#202](https://github.com/CWorthy-ocean/roms-tools/pull/202)).

### Documentation

* Release notes have been added to the documentation ([#202](https://github.com/CWorthy-ocean/roms-tools/pull/202)).

### Bugfixes

* Bug with radiation correction in surface forcing has been fixed ([#209](https://github.com/CWorthy-ocean/roms-tools/pull/209)).

## v2.0.0

### New Features

* This release supports more accurate topography regridded from SRTM15, but also still supports the previous ETOPO5 topography ([#172](https://github.com/CWorthy-ocean/roms-tools/pull/172)).
* The land mask is improved since it is now inferred from a separate coastline dataset rather than from the regridded topography ([#172](https://github.com/CWorthy-ocean/roms-tools/pull/172)).

### Breaking Changes

* The topography format is no longer backward compatible. The syntax for specifying the topography source has changed ([#172](https://github.com/CWorthy-ocean/roms-tools/pull/172)). The new format is
```
grid = Grid(topography_source={"name": "ETOPO5"})
```
instead of the  previous format
```
grid = Grid(topography_source="ETOPO5")
```

### Internal Changes

* Topography regridding function has been generalized ([#172](https://github.com/CWorthy-ocean/roms-tools/pull/172)).
* The classes {py:obj}`roms_tools.setup.datasets.ETOPO5Dataset` and {py:obj}`roms_tools.setup.datasets.SRTM15Dataset` have been introduced ([#172](https://github.com/CWorthy-ocean/roms-tools/pull/172)).
* The {py:obj}`roms_tools.setup.datasets.Dataset` class has been made more performant by
    - ensuring that global datasets are only concatenated along the longitude dimension when necessary and
    - restricting to the necessary latitute range first before concatenation ([#172](https://github.com/CWorthy-ocean/roms-tools/pull/172)).
* Layer and interface depths are not computed as part of grid generation any longer because this can blow memory for large domains. Instead, layer and interface depths are only computed when needed as part of initial conditions and boundary forcing generation ([#172](https://github.com/CWorthy-ocean/roms-tools/pull/172)).

### Documentation

* SRTM15 topography examples have been added to the documentation ([#172](https://github.com/CWorthy-ocean/roms-tools/pull/172)).

### Bugfixes
