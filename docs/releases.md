# Release notes

## v2.4.0 (unreleased)

### New Features

### Breaking Changes

* Remove support for partitioning files upon saving ([#221](https://github.com/CWorthy-ocean/roms-tools/pull/221))

### Internal Changes

### Documentation

### Bugfixes

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
