# ROMS-Tools

[![Conda version](https://img.shields.io/conda/vn/conda-forge/roms-tools.svg)](https://anaconda.org/conda-forge/roms-tools)
[![PyPI version](https://img.shields.io/pypi/v/roms-tools.svg)](https://pypi.org/project/roms-tools/)
[![Run Tests](https://github.com/CWorthy-ocean/roms-tools/actions/workflows/tests.yaml/badge.svg)](https://github.com/CWorthy-ocean/roms-tools/actions/workflows/tests.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/CWorthy-ocean/roms-tools/graph/badge.svg?token=5S1oNu39xE)](https://codecov.io/gh/CWorthy-ocean/roms-tools)
[![Documentation Status](https://readthedocs.org/projects/roms-tools/badge/?version=latest)](https://roms-tools.readthedocs.io/en/latest/?badge=latest)
![Supported Python Versions](https://img.shields.io/pypi/pyversions/roms-tools)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/roms-tools?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/roms-tools)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.10234/status.svg)](https://doi.org/10.21105/joss.10234)

## Overview

A suite of Python tools for setting up and analyzing a [UCLA-ROMS](https://github.com/CWorthy-ocean/ucla-roms) simulation with or without [MARBL biogeochemistry](https://marbl-ecosys.github.io/versions/latest_release/index.html).

## Installation

### ⚡️ **Installation from conda-forge (recommended)**

To install `ROMS-Tools` with all dependencies, including `xesmf`, `dask` and all packages required for streaming source data directly from the cloud, use:

```bash
conda install -c conda-forge roms-tools
```

> [!Note]
>  Installation from Conda-Forge is the recommended installation method to ensure all features of `ROMS-Tools` are available.

### 📦 **Installation from PyPI (pip) (recommend for Windows only)**

You can also install `ROMS-Tools` from `pip`:

```bash
pip install roms-tools
```

`dask` (recommended for parallel and out-of-core computation) is included by default.

If you want to use `ROMS-Tools` with all packages required for streaming source data directly from the cloud, install it with the additional dependencies:

```bash
pip install roms-tools[stream]
```


> [!Note]
>  The PyPI versions of `ROMS-Tools` do not include `xesmf`, so some features will be unavailable.


### Installation from GitHub

To obtain the latest development version, first clone the source repository:

```bash
git clone https://github.com/CWorthy-ocean/roms-tools.git
cd roms-tools
```

Next, install and activate the following conda environment:

```bash
conda env create -f ci/environment.yml
conda activate roms-tools-env
```

Finally, install `ROMS-Tools` in the same environment, along with the `dev` extra (test/lint tooling):

```bash
pip install -e ".[dev]"
```

`dask` (recommended for parallel and out-of-core computation) is included in the core dependencies by default.

If you want to use `ROMS-Tools` with all packages required for streaming source data directly from the cloud, you can
install `ROMS-Tools` along with the additional dependencies via:

```bash
pip install -e ".[dev,stream]"
```

## Getting Started

To learn how to use `ROMS-Tools`, check out the [documentation](https://roms-tools.readthedocs.io/en/latest/).

## How to cite ROMS-Tools

If you use ROMS-Tools in your work, we would be happy if you cite our paper in the Journal of Open Source Software:

> Loose, N., Nicholas, T., Maticka, S., Eilerman, S., McBride, C., Stephenson, D., Heede, U., Saenz, B., Thyng, K. M., Bachman, S., Damien, P., Karspeck, A., Long, M. C., Molemaker, M. J., & Wyatt, A. (2026). ROMS-Tools: Reproducible Preprocessing and Analysis for Regional Ocean Modeling with ROMS. *Journal of Open Source Software*, *11*(124), 10234. https://doi.org/10.21105/joss.10234

Or use the following BibTeX entry:

```bibtex
@article{loose2026romstools,
  author  = {Loose, Nora and Nicholas, Tom and Maticka, Sam and Eilerman, Scott
             and McBride, Christopher and Stephenson, Dafydd and Heede, Ulla
             and Saenz, Benjamin and Thyng, Kristen M. and Bachman, Scott
             and Damien, Pierre and Karspeck, Alicia and Long, Matthew C.
             and Molemaker, M. Jeroen and Wyatt, Abigale},
  title   = {{ROMS-Tools}: Reproducible Preprocessing and Analysis for
             Regional Ocean Modeling with {ROMS}},
  journal = {Journal of Open Source Software},
  year    = {2026},
  volume  = {11},
  number  = {124},
  pages   = {10234},
  doi     = {10.21105/joss.10234},
  url     = {https://joss.theoj.org/papers/10.21105/joss.10234},
}
```

## Feedback and contributions

**If you find a bug, have a feature suggestion, or any other kind of feedback, please start a Discussion.**

We also accept contributions in the form of Pull Requests.

## See also

- [C-Star](https://github.com/CWorthy-ocean/C-Star)
