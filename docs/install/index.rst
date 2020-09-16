.. _install:

Installation
============

The only requirement is a Python3.x installation with a compatible package
manager (e.g. pip).

A virtual environment manager such as the one provided by an Anaconda
(or Miniconda) installation supporting such python version is recommended.

These instructions we will assume that this is the case.

There are two different ways to install `pyirf`,

* if you just want to use it as it is (:ref:`basic`),
* or if you also want to develop it (:ref:`developer`).

.. warning::
  We are in the early phases of development: even if a pre-release already
  exists, it is likely that you will benefit more from the development version.

After installing `pyirf`, you can start using it (:ref:`usage`).

.. Note::

  For a faster use, edit your preferred login script (e.g. ``.bashrc`` on Linux or
  ``.profile`` on macos) with a function that initializes the environment.
  The following is a minimal example using Bash.

  .. code-block:: bash

    alias pyirf_init="pyirf_init"

    function pyirf_init() {

        conda activate pyirf # Then activate the pyirf environment
        export PYIRF=$WHEREISPYIRF/pyirf/scripts # A shortcut to the scripts folder

    }

.. toctree::
    :hidden:
    :maxdepth: 2

    basic
    developer
