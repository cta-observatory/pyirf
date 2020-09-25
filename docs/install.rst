.. _install:

Installation
============


``pyirf`` requires Python ≥3.6 and the packages as defined in the ``setup.py``.
Core dependencies are

* ``numpy``
* ``astropy``
* ``scipy``

Dependencies for development, like unit tests and building the documentation
are defined in ``extras``.

Installing a released version
-----------------------------


To install a release version, just install the ``pyirf`` package using

.. code-block:: bash

    pip install pyirf

or add it to the dependencies of your project.


Installing for development
--------------------------

If you want to work on pyirf itself, clone the repository or your fork of
the repository and install the local copy of pyirf in development mode.

Make sure you add the ``tests`` and ``docs`` extra to also install the dependencies
for unit tests and building the documentation.
You can also simply install the ``all`` extra:

.. code-block:: bash

    pip install -e '.[all]'  # or [docs,tests]
