.. _install:

Installation
============

The only requirement is an Anaconda (or Miniconda) installation which supports
Python 3.

There are two different ways to install `pyirf`,

* if you just want to use it as it is (:ref:`install-basic`),
* or if you also want to develop it (:ref:`install-developer`).

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
    :maxdepth: 1

    basic
    developer
