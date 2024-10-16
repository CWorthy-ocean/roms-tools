Contributor Guide
##################

Running the tests
=================
Install and activate the following conda environment::

    cd roms-tools
    conda env create -f ci/environment.yml
    conda activate romstools-test
    pip install -e ".[dask]"

You can check the functionality of the ROMS-Tools code by running the test suite::

    conda activate romstools-test
    pytest


Contributing code
=================

If you have written new code, you can run the tests as described in the previous step. You will likely have to iterate here several times until all tests pass.
The next step is to make sure that the code is formatted properly. Activate the environment::

    conda activate romstools-test

You can now run all the linters with::

    pre-commit run --all-files

Some things will automatically be reformatted, others need manual fixes. Follow the instructions in the terminal until all checks pass.
Once you got everything to pass, you can stage and commit your changes and push them to the remote github repository.


Building the documentation locally
==================================

Install and activate the following conda environment::

    cd roms-tools
    conda env create -f docs/environment.yml
    conda activate romstools-docs
    pip install -e ".[dask]"

Then navigate to the docs folder and build the docs via::

    cd docs
    make fresh
    make html

You can now open ``docs/_build/html/index.html`` in a web browser.

