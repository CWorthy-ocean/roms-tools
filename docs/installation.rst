Installation
############

Installation from pip
=====================

ROMS-Tools can be installed using pip::

    pip install roms-tools


Installation from GitHub
========================

To obtain the latest development version, first clone
`the source repository <https://github.com/CWorthy-ocean/roms-tools.git>`_::

    git clone https://github.com/CWorthy-ocean/roms-tools.git
    cd roms-tools

Next, install and activate the following conda environment::

    conda env create -f ci/environment.yml
    conda activate romstools

Finally, install ROMS-Tools in the same environment:

    pip install -e .


Running the tests
=================

You can check the functionality of the ROMS-Tools code by running the test suite::

    conda activate romstools
    cd roms-tools
    pytest


Contributing code
=================

If you have written new code, you can run the tests as described in the previous step. You will likely have to iterate here several times until all tests pass.
The next step is to make sure that the code is formatted properly. Activate the environment::

    conda activate romstools

You can now run all the linters with::

    pre-commit run --all-files

Some things will automatically be reformatted, others need manual fixes. Follow the instructions in the terminal until all checks pass.
Once you got everything to pass, you can stage and commit your changes and push them to the remote github repository.


Building the documentation locally
==================================

Activate the environment::

    conda activate romstools

Then navigate to the docs folder and build the docs via::

    cd docs
    make fresh
    make html

You can now open ``docs/_build/html/index.html`` in a web browser.
