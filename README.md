# ROMS-Tools

[![Documentation Status](https://readthedocs.org/projects/roms-tools/badge/?version=latest)](https://roms-tools.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/roms-tools.svg)](https://badge.fury.io/py/roms-tools)

## Overview

A suite of python tools for setting up a [ROMS](https://github.com/CESR-lab/ucla-roms) simulation.

## Installation instructions

### Installation from pip

```bash
pip install roms-tools
```

### Installation from GitHub

To obtain the latest development version, clone the source repository and install it:

```bash
git clone https://github.com/CWorthy-ocean/roms-tools.git
cd roms-tools
pip install -e .
```


### Run the tests

Before running the tests you can install and activate the following conda environment:

```bash
cd roms-tools
conda env create -f ci/environment.yml
conda activate romstools
```

Check the installation of `ROMS-Tools` has worked by running the test suite
```bash
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
