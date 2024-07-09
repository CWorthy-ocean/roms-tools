Installation
############

Installation from pip
=====================

ROMS-Tools can be installed using pip::

    pip install roms-tools


Installation from GitHub
========================

To obtain the latest development version, clone
`the repository <https://github.com/CWorthy-ocean/roms-tools.git>`_
and install it as follows::

    git clone https://github.com/CWorthy-ocean/roms-tools.git
    cd roms-tools
    pip install -e .


Conda environment
=================

You can install and activate the following conda environment::

    cd roms-tools
    conda env create -f ci/environment.yml
    conda activate romstools-test

This conda environment is useful for any of the following steps:

1. Running the example notebooks
2. Contributing code and running the testing suite
3. Building the documentation locally

Running the tests
=================

You can check the functionality of the ROMS-Tools code by running the test suite::

    cd roms-tools
    pytest


Contributing code
=================

If you have written new code, you can run the tests as described in the previous step. You will likely have to iterate here several times until all tests pass.
The next step is to make sure that the code is formatted properly. You can run all the linters with::

    pre-commit run --all-files

Some things will automatically be reformatted, others need manual fixes. Follow the instructions in the terminal until all checks pass.
Once you got everything to pass, you can stage and commit your changes and push them to the remote github repository.


Building the documentation locally
==================================

Activate the environment::

    conda activate romstools-test

Then navigate to the docs folder and build the docs via::

    cd docs
    make fresh
    make html

You can now open ``docs/_build/html/index.html`` in a web browser.
