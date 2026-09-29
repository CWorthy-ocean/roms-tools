# Catalog trial: findings

`roms_tools.yaml` describes four source datasets as intake v2 catalog entries, in the
shape ocean-skill uses, and `roms_tools.datasets.catalog.from_catalog` builds ordinary
`LatLonDataset` / `ROMSDataset` objects from them. No per-source subclass is involved.
The path is opt-in (`catalog.use_catalog()` or `ROMS_TOOLS_USE_CATALOG=1`) and needs
`pip install roms-tools[catalog]`. Without intake nothing changes.

Tested with intake 2.0.9, cf_xarray 0.11.0, and ocean-skill at commit 446d0d2.

## What was shown

| Check | Result |
|---|---|
| SRTM15, ETOPO5, GLORYS files, ROMS restart (physics and BGC, one file and two) built from the catalog | processed `.ds` identical to the class-built object, same name maps, with and without dask |
| `Grid` topography and `InitialConditions` (GLORYS, ROMS physics and BGC) with the catalog on and off | identical `.ds` |
| An entry written by ocean-skill (`glorys_my_daily_geo`) | resolves and is eligible with nothing appended to it |
| All ocean-skill shipped catalogs indexed through `$OCEAN_SKILL_CATALOGS` | over 100 aliases, no errors |
| A site catalog shadowing a packaged entry, and a ROMS entry carrying its own grid file and reference date | works |
| Full existing suite | 1264 passed with intake installed, 1228 passed without |

Not run: the Copernicus streaming test (`use_copernicus`, needs credentials), and
multi-file lat/lon input (only ROMS was tried with two files).

## How much became data

| Source | Class today | Catalog entry |
|---|---|---|
| SRTM15 | 16 lines | 16 lines, no code |
| ETOPO5 | 38 lines, with a `clean_up` override | 18 lines plus a 9-line named transform |
| GLORYS (files) | 67 lines, with a `post_process` override | 20 lines plus a 21-line named transform |
| GLORYS (Copernicus) | 105 lines (`GLORYSDefaultDataset`) | ocean-skill's 62-line entry with 3 lines added |
| ROMS output | no subclass | 23 lines |

Entries are not much shorter than the classes they replace. What changes is that the facts
are data that can be shadowed by site catalogs and shared with ocean-skill, and that adding
a source that needs no processing takes no Python.

## What the engine needed

Two small changes to the existing classes, both no-ops for existing code:

* `LatLonDataset` and `ROMSDataset` split `__post_init__` into load plus `_process(ds)`, with
  a `from_dataset` classmethod, because the catalog reader owns loading. Routing a catalog
  reader through `load_data`'s `ds_loader_fn` would have skipped `chunks`, `use_dask`,
  `initial_slice_bounds` and the zarr checks.
* `LatLonDataset` gained `clean_up_fn` / `post_process_fn`. A transform takes and returns
  `(ds, names)`, so a transform that must change the name maps does so by return value.

## Findings

1. **Roles resolve through the vocabulary, but only with anchored patterns.** Every role
   resolved on all four real files, including ETOPO5, SRTM15 and the restart, which carry
   no `standard_name`. cf-xarray matches from the start of a name, so an unanchored `z`
   also matched `zeta` and `zos`; a test now rejects unanchored patterns. Ambiguity raises.
2. **The vocabulary covers physics and topography only.** ROMS output still takes its
   variable names from `_set_required_vars`; 22 of its 33 BGC tracers have no CF name and
   are not in `vocabulary.json`. Extending the vocabulary to them is the natural next step.
3. **Which forcing an entry can serve is not fully derivable from metadata.** `has()` checks
   the dataset kind, vertical axis and an optional `contexts` restriction. It cannot tell
   that GLORYS lacks a topography variable; that is reported as a clear error when the data
   is opened. Entries carrying `variables` (as ocean-skill's probed ones do) could close this
   without opening the data.
4. **Runtime paths work as a catalog-scoped `path` parameter.** Applying it on the entry
   instead does not (it leaks into the reader's keyword arguments). intake offers no way to
   ask which entries use a parameter, so `_template_entries` inspects the data descriptors.
5. **intake gotchas found while building this.** A one-element file list is opened with
   `open_dataset`, so multi-file keyword arguments must be added only for longer lists.
   intake's `HDF5` datatype defaults to the h5netcdf engine, which cannot open ETOPO5
   (netCDF3 classic), so every entry pins `engine: netcdf4`. The `data:` and `entries:` keys
   need empty `metadata` and `user_parameters` mappings or loading fails with a `KeyError`.
6. **What intake actually provided.** YAML loading, instantiating a reader from an import
   string, and substituting the `path` parameter. Discovery and search path, eligibility,
   role resolution, and reader-name mapping are ours. That is a modest amount to stand on
   a dependency that ocean-skill itself calls provisional; a plain YAML loader with the same
   file shape would cost an estimated 40 to 60 lines.
7. **The catalog path is dask-backed only when asked.** Reader `chunks` are `None` unless
   `use_dask`, and `initial_slice_bounds` is applied after opening rather than per file
   while opening. Results matched; memory behaviour on many large files is untested.
8. **No change was needed in ocean-skill** for any of this. Follow-ups that would help:
   a builder helper for path-templated entries, a probe for `land_masked`, a neutral
   catalog search-path variable, and a documented convention for the `transforms` key.
9. **Docs bug found in passing.** `docs/reading_roms_output.ipynb` refers to
   `Grid.from_file`, which does not exist; the public route is `Grid(filename=...)`.

## What was not tried and is likely harder

ERA5 (derived humidity, unit conversions, name maps changed in `post_process`), CESM
(depth dimension renamed and spliced), WOA (self-downloading, extra options), Unified BGC
(two on-disk generations), TPXO (three files), and river datasets (station tables, no
lat/lon grid). The transform signature was chosen with the first two in mind but has not
been exercised on them.

## Trial-only scaffolding

The `use_catalog()` switch, the two guarded branches in `setup/topography.py` and
`setup/initial_conditions.py`, and the mapping of `ocean_skill.readers:CopernicusMarineReader`
to the local copy. Everything else is written to stay.
