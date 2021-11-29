.. _install:

Installation
============

``pyirf`` requires Python ≥3.7 and pip, plus the packages defined in
the ``setup.py``.

Core dependencies are

* ``numpy``
* ``astropy``
* ``scipy``

We provide an environment file for Anaconda or Miniconda users.

Installing a released version
-----------------------------

To install a released version, just install the ``pyirf`` package using

.. code-block:: bash

    pip install pyirf

or add it to the dependencies of your project.

Installing for development
--------------------------

If you want to work on pyirf itself, clone the repository and install the local
copy of pyirf in development mode.

The dependencies required to perform unit-testing and to build the documentation
are defined in ``extras`` under ``tests`` and ``docs`` respectively

These requirements can be enabled by installing the ``all`` extra:

Best practice is to create a new conda environment for development as below 
(which will get the name ``pyirf``):

.. code-block:: bash

    conda env create -f environment.yml
    conda activate pyirf
    pip install -e '.[all]' # or [docs,tests] to install them separately

where the ```-e``` switch for ``pip`` is for editable, 
also called "Developer Mode". 
It installs symlinks from the current directory 
(the ``.`` in the command) into the Python in use.

So this creates a new conda env with all needed dependencies 
and symlinks to your development repository for pyirf.
Then, your modifications can be committed with git, 

*Note:* if you want to use your ongoing modifications in 
another of your conda environments, 
just execute the 

.. code-block:: bash

    pip install -e .  

from your cloned directory, with that environment active.