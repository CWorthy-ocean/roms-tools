Contributor Guide
##################

Running the tests
=================
Install and activate the following conda environment::

    cd roms-tools
    conda env create -f ci/environment.yml
    conda activate roms-tools-env
    pip install -e ".[dev]"

You can check the functionality of the ROMS-Tools code by running the test suite::

    conda activate roms-tools-env
    pytest


Contributing code
=================

If you have written new code, you can run the tests as described in the previous step. You will likely have to iterate here several times until all tests pass.
The next step is to make sure that the code is formatted properly. Activate the environment::

    conda activate roms-tools-env

You can now run all the linters with::

    pre-commit run --all-files

Some things will automatically be reformatted, others need manual fixes. Follow the instructions in the terminal until all checks pass.
Once you got everything to pass, you can stage and commit your changes and push them to the remote github repository.


Writing a release note
======================

A pull request that changes something users could notice adds a release note: a file in the ``changes/`` directory named ``<PR number>.<type>.md``. Open the pull request first to get its number, then commit the file to your branch. At release time the notes are collected into ``docs/releases.md`` and the files are deleted, so don't edit ``docs/releases.md`` yourself.

The type decides the section the note appears under:

.. list-table::
   :header-rows: 1

   * - Type
     - Section
     - For
   * - ``breaking``
     - Breaking Changes
     - changes to interfaces, inputs or outputs that can break existing code or change results
   * - ``deprecation``
     - Deprecations
     - something that still works but will be removed
   * - ``feature``
     - New Features
     - new capabilities
   * - ``bugfix``
     - Bug Fixes
     - code that did not do what it should, and now does
   * - ``improvement``
     - Improvements
     - better behavior, messages or performance of existing code
   * - ``misc``
     - Miscellaneous
     - documentation, CI, packaging and test changes worth mentioning

Write the note for someone reading the release notes, not for the reviewer: what changed for them and anything they need to do, in one line. Details for the reviewer go in the pull request description. If the reader needs more, add ``* `` sub-bullets on the following lines. For example, ``changes/709.bugfix.md``:

.. code-block:: markdown

    Initial conditions from a ROMS restart no longer repeat the parent's deepest water at child levels below the parent's bottom; those tracers now come from the nearest deeper parent column at the same depth.
    * This changes the initial state of any nested child that is deeper than its parent somewhere, for physics and BGC tracers alike; velocities are unchanged.

becomes, in ``docs/releases.md``:

.. code-block:: markdown

    ### Bug Fixes

    * Initial conditions from a ROMS restart no longer repeat the parent's deepest water at child levels below the parent's bottom; those tracers now come from the nearest deeper parent column at the same depth. ([#709](https://github.com/CWorthy-ocean/roms-tools/pull/709))
      * This changes the initial state of any nested child that is deeper than its parent somewhere, for physics and BGC tracers alike; velocities are unchanged.

A pull request with notes of several types adds one file per type, such as ``715.breaking.md`` and ``715.feature.md``. Two separate notes of the same type are numbered: ``715.breaking.1.md`` and ``715.breaking.2.md``. A deprecation note reads ``Deprecated: <what>, removed in <release>; use <replacement>.``

A pull request with nothing for users to read about, such as a test-only change or a typo fix, needs no note; ask a maintainer to add the ``no-changelog`` label. The ``towncrier check`` job on the pull request fails until it has a note or the label.

To preview the notes as they will appear, run this in the ``roms-tools-env`` environment (``towncrier`` comes with ``pip install -e ".[dev]"``)::

    towncrier build --draft --version Unreleased


Making a release
================

For maintainers:

1. Run the *Finalize Release Notes* workflow from the repository's Actions tab with the new version, for example ``5.4.0``. It turns the files in ``changes/`` into a ``## 5.4.0`` section of ``docs/releases.md``, deletes them, and opens a *Finalize release notes for 5.4.0* pull request from the ``next_release_notes`` branch.
2. Edit the notes on that pull request if they need it: merge overlapping notes, reorder, fix wording.
3. Merge it. This creates the tagged GitHub release with those notes, which publishes the package to PyPI.


Building the documentation locally
==================================

Install and activate the following conda environment::

    cd roms-tools
    conda env create -f docs/environment.yml
    conda activate roms-tools-docs-env
    pip install -e ".[docs]"

Navigate to the docs folder and build the docs with the activated environment::

    cd docs
    conda activate roms-tools-docs-env
    make fresh
    make html

You can now open ``docs/_build/html/index.html`` in a web browser.
