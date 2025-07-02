# ROMS-Tools
[![Conda version](https://img.shields.io/conda/vn/conda-forge/roms-tools.svg)](https://anaconda.org/conda-forge/roms-tools)
[![PyPI version](https://img.shields.io/pypi/v/roms-tools.svg)](https://pypi.org/project/roms-tools/)
[![codecov](https://codecov.io/gh/CWorthy-ocean/roms-tools/graph/badge.svg?token=5S1oNu39xE)](https://codecov.io/gh/CWorthy-ocean/roms-tools)
[![Documentation Status](https://readthedocs.org/projects/roms-tools/badge/?version=latest)](https://roms-tools.readthedocs.io/en/latest/?badge=latest)
![Run Tests](https://github.com/CWorthy-ocean/roms-tools/actions/workflows/tests.yaml/badge.svg)
![Supported Python Versions](https://img.shields.io/pypi/pyversions/roms-tools)

## Overview

A suite of Python tools for setting up and analyzing a [UCLA-ROMS](https://github.com/CESR-lab/ucla-roms) simulation.

## Installation

### ⚡️ **Installation from Conda-Forge**

To install `ROMS-Tools` with all dependencies, including `xesmf`, `dask` and all packages required for streaming source data directly from the cloud, use:

```bash
conda install -c conda-forge roms-tools
```

> [!Note]
>  Installation from Conda-Forge is the recommended installation method to ensure all features of `ROMS-Tools` are available.

### 📦 **Installation from PyPI (pip)**

You can also install `ROMS-Tools` from `pip`:

```bash
pip install roms-tools
```

If you want to use `ROMS-Tools` with `dask` (recommended for parallel and out-of-core computation), install it with the additional dependency:

```bash
pip install roms-tools[dask]
```

If you want to use `ROMS-Tools` with `dask` and all packages required for streaming source data directly from the cloud, install it with the additional dependencies:

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
conda env create -f ci/environment-with-xesmf.yml
conda activate romstools-test
```

Finally, install `ROMS-Tools` in the same environment:

```bash
pip install -e .
```

If you want to use `ROMS-Tools` with `dask` (recommended for parallel and out-of-core computation), you can
install `ROMS-Tools` along with the additional dependency via:

```bash
pip install -e ".[dask]"
```

If you want to use `ROMS-Tools` with `dask` and all packages required for streaming source data directly from the cloud, you can
install `ROMS-Tools` along with the additional dependencies via:

```bash
pip install -e ".[stream]"
```

## Getting Started

To learn how to use `ROMS-Tools`, check out the [documentation](https://roms-tools.readthedocs.io/en/latest/).

## Feedback and contributions

**If you find a bug, have a feature suggestion, or any other kind of feedback, please start a Discussion.**

We also accept contributions in the form of Pull Requests.

## See also

- [ROMS source code](https://github.com/CESR-lab/ucla-roms)
- [C-Star](https://github.com/CWorthy-ocean/C-Star)
