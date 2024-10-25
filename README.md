# ROMS-Tools
[![PyPI version](https://img.shields.io/pypi/v/roms-tools.svg)](https://pypi.org/project/roms-tools/)
[![codecov](https://codecov.io/gh/CWorthy-ocean/roms-tools/graph/badge.svg?token=5S1oNu39xE)](https://codecov.io/gh/CWorthy-ocean/roms-tools)
[![Documentation Status](https://readthedocs.org/projects/roms-tools/badge/?version=latest)](https://roms-tools.readthedocs.io/en/latest/?badge=latest)
![Run Tests](https://github.com/CWorthy-ocean/roms-tools/actions/workflows/tests.yaml/badge.svg)
![Supported Python Versions](https://img.shields.io/pypi/pyversions/roms-tools)


## Overview

A suite of python tools for setting up a [ROMS](https://github.com/CESR-lab/ucla-roms) simulation.

<mark>**DISCLAIMER**: This project is **ALPHA** and will be **EXPERIMENTAL** at least throughout Fall 2024. We do not recommend to use this software before this ALPHA development and test phase is completed.</mark>

## Installation

### Installation from conda forge

```bash
conda install -c conda-forge roms-tools
```

This command installs `ROMS-Tools` along with its `dask` dependency.

### Installation from pip

```bash
pip install roms-tools
```

If you want to use `ROMS-Tools` together with dask (which we recommend), you can
install `ROMS-Tools` along with the additional dependency via:

```bash
pip install roms-tools[dask]
```

### Installation from GitHub

To obtain the latest development version, first clone the source repository:

```bash
git clone https://github.com/CWorthy-ocean/roms-tools.git
cd roms-tools
```

Next, install and activate the following conda environment:

```bash
conda env create -f ci/environment.yml
conda activate romstools-test
```

Finally, install `ROMS-Tools` in the same environment:

```bash
pip install -e .
```

If you want to use `ROMS-Tools` together with dask (which we recommend), you can
install `ROMS-Tools` along with the additional dependency via:

```bash
pip install -e .[dask]
```


### Run the tests

Before running the tests, you can activate the conda environment created in the previous section:

```bash
conda activate romstools-test
```

Check the installation of `ROMS-Tools` has worked by running the test suite
```bash
cd roms-tools
pytest
```

## Getting Started

To learn how to use `ROMS-Tools`, check out the [documentation](https://roms-tools.readthedocs.io/en/latest/).

## Feedback and contributions

**If you find a bug, have a feature suggestion, or any other kind of feedback, please start a Discussion.**

We also accept contributions in the form of Pull Requests.


## See also

- [ROMS source code](https://github.com/CESR-lab/ucla-roms)
- [C-Star](https://github.com/CWorthy-ocean/C-Star)
