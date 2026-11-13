# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project overview

ROMS-Tools (`roms_tools`; PyPI and conda-forge name `roms-tools`) generates
input files for UCLA-ROMS simulations, with or without MARBL biogeochemistry.
It builds the grid and topography, initial conditions, boundary, surface, tidal,
and river forcing, CDR release forcing, and nesting by regridding source
datasets (GLORYS, ERA5, CESM and Unified BGC, TPXO, and the Dai-Trenberth,
GloFAS, and RIVR2O river datasets) onto a ROMS grid. It also provides post-run
analysis (`ROMSOutput`, CDR metrics, ensembles) and NetCDF partition and join
tiling. The sibling project C-Star consumes its YAML output.

## Development setup

A conda environment is required, because xesmf and esmpy cannot be installed
with pip:

```bash
conda env create -f ci/environment.yml    # env name: roms-tools-env
conda activate roms-tools-env
pip install -e ".[dev]"                   # add ",stream" for cloud streaming
```

- Install other packages with `mamba install -c conda-forge <pkg>` (or
  `conda install` if you do not have mamba). Use pip only for the editable
  install above or for a package that is not on conda-forge.
- Ask the user before changing an environment you did not create.
- The docs build is described in `docs/contributing.rst`.

## Testing

The full suite is slow: it does heavy regridding, and CI splits it into several
jobs. On a cold cache it downloads test data with pooch on first use, so it
needs network access before collection finishes.

For small changes, run only the test files or `-k` selections that cover the
code you touched. Do not run bare `pytest` as your edit loop. Run the full suite
only as a final gate or for wide-reaching changes.

```bash
# Targeted (preferred while developing). Tests live inside the package.
pytest roms_tools/tests/test_setup/test_river_forcing.py -q
pytest roms_tools/tests -k "pattern" -q

# Full local suite (final gate). Tests that need dask, streaming, Copernicus,
# or gcsfs are skipped unless you pass the matching flag.
pytest roms_tools/tests -q

# Dask variant. CI runs both ways, and behavior differs.
pytest --use_dask roms_tools/tests -q
```

- The root `conftest.py` registers the markers and flags and holds the
  session-scoped fixtures. Flags: `--use_dask`, `--stream`, `--use_copernicus`
  (needs `copernicusmarine login`), `--use_gcsfs`, and
  `--overwrite=<fixture>|all`.
- `--overwrite` regenerates the stored zarr reference fixtures under
  `roms_tools/tests/test_setup/test_data/`. A diff there is a scientific-output
  change. Use it only on purpose, and tell the user which fixtures changed.

## Lint, format, and type checks

```bash
pre-commit run --all-files
```

The hooks are trailing-whitespace, end-of-file-fixer, check-docstring-first,
check-json, check-yaml, debug-statements, mixed-line-ending, `absolufy-imports`,
`ruff-check --fix`, `ruff-format`, and mypy.

- Inside `roms_tools/`, use absolute imports, never relative ones.
- Ruff selects `I`, `E`, `W`, `F`, `D`, `RUF`, and `UP`, with line length 88.
  The `docs/` folder is excluded.
- New public code needs type hints, docstrings, and tests.

## Layout

- `roms_tools/setup/`: `grid.py` is the foundation for everything else. The
  forcing generators are `initial_conditions`, `boundary_forcing`,
  `surface_forcing`, `tides`, `river_forcing`, `cdr_forcing`, `cdr_release`, and
  `nesting`. `utils.py` holds the shared `to_dict` and `from_yaml`
  serialization.
- `roms_tools/datasets/`: source-dataset classes. `lat_lon_datasets.py` is the
  largest file (GLORYS, ERA5, TPXO, CESM and Unified BGC).
- `roms_tools/processing_methods.py`: the single source of truth for the
  prefill, regrid, and extrapolation option vocabulary. It imports nothing from
  `roms_tools`, to avoid import cycles. Edit it when you add a processing
  option.
- Also `roms_tools/analysis/`, `roms_tools/tiling/`, `roms_tools/regrid.py`,
  `roms_tools/fill.py`, tests in `roms_tools/tests/`, docs in `docs/`, and CI
  scripts in `ci/`.

## Design and API conventions

- Keep one source of truth for each name, option list, and constant. Import it
  instead of repeating a literal. Fail fast on invalid input, with a clear error
  and no silent default.
- Setup classes mix `dataclass` and Pydantic (the CDR classes are Pydantic).
  Follow the style of the module you are editing.
- Constructor options are plain strings, so the YAML round trip stays simple.
- Domains can be large. A single forcing or output file can be tens to hundreds
  of GB, and one 3-D field about 1.4 GB. In code under `setup/`, `datasets/`,
  `regrid.py`, `fill.py`, and `tiling/`, avoid realizing whole lazy arrays,
  holding them until save, or rebuilding an interpolator on every call. Grids
  and CDR forcing stay small.

## Gotchas

- Multi-document YAML output is intentional. `write_to_yaml` emits a
  `roms_tools_version` header document, then the payload. Read it with
  `yaml.safe_load_all`.
- Some dependencies are deliberately pinned in `pyproject.toml` (`numpy`,
  `xgcm`, `matplotlib`). Do not loosen the pins casually. There is no upper cap
  on `xarray`; do not add one.
- xesmf is optional (conda only). Code and tests branch on whether it is
  installed, and the scipy and xESMF regrid engines give numerically different
  results.
- Import order in `roms_tools/__init__.py` matters: `Grid` comes first.
- An uninstalled checkout reports version `9999`, and that string lands in YAML
  headers.
- River forcing puts a river's whole discharge into one coastline cell unless
  you pass `RiverForcing(indices=...)`, which includes only the rivers you
  name. See the `indices` entry in the `RiverForcing` docstring before running
  high-resolution domains.

## Related repositories

- **ucla-roms** is the Fortran ocean model. It reads the NetCDF files that
  roms-tools writes. River cells must be coastal land cells (land with a wet
  neighbor), the rule ucla-roms checks at startup and roms-tools checks when
  you pass `indices`.
- **C-Star** orchestrates runs. It consumes the YAML that roms-tools writes
  and calls roms-tools to build domains. A change to the YAML layout, a
  `to_dict` key, or a public signature can break C-Star, so check it before
  you merge.

## Data safety

Never delete, move, or overwrite model data, scratch or project storage, a
shared source-data cache, or any file another user owns without the user's
explicit OK, even to start fresh. When you need a destructive command, stop and
give the user the exact command.

## Release notes

A pull request with a user-visible change adds its release note as a file,
`changes/+<name>.<type>.md`, committed with the code (`<name>` is any short
label). The types are the `[[tool.towncrier.type]]` entries in
`pyproject.toml`. `breaking` (a setup that worked now fails or needs an edit)
and `output` (the same setup still runs but writes different files) are
separate types. Write each note for users in one line, optionally followed by
`* ` sub-bullets, one file per note. At release time the pull request number is
filled in and towncrier builds `docs/releases.md` from the files, so do not
edit `docs/releases.md` on a branch. The pull request description is for the
reviewer. `docs/contributing.rst` has the details.

## Posting on GitHub

This project follows the [C]Worthy AI policy:
https://github.com/CWorthy-ocean/.github/blob/main/AI_POLICY.md

- You may draft pull request descriptions, bug reports, and summaries of test
  results for the user.
- Do not post replies, review comments, answers, or discussion comments on the
  user's behalf, and do not join a discussion on your own. If asked to, tell
  the user that the project asks for replies in their own words, and give them
  notes to write from. If they share your output, they quote it and say where
  it came from.
- Fill in `.github/PULL_REQUEST_TEMPLATE.md`, including the `AI disclosure:`
  line: `none`, or `<tool>; <what it did>`.
- Do not add `Co-authored-by`, `Assisted-by`, `Generated-by`, or similar AI
  trailers to commits.
- The user answers for what you open under their account. Tell them plainly
  what you did not run or could not verify.
