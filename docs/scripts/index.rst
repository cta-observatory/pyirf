.. _scripts:

=======
Scripts
=======

Introduction
============

This module contains the scripts to produce DL3 data as explained in :ref:`usage`.

At the moment there are 3 such scripts:

- `make_performance.py`, the old script from protopipe,
- `make_DL3.py`, the new version which is supposed to be final one at least for DL3 data based on simulations,
- `lst_performance.py`, a script specific for LSTchain.

.. todo::
  
  Remove `make_performance.py`.

Details
=======

make_DL3
--------

The usage is the following,

.. code-block::
  
  >$ python $PYIRF/make_DL3.py -h
  usage: make_DL3.py [-h] --config_file CONFIG_FILE --obs_time OBS_TIME
                   --pipeline PIPELINE [--debug]

  Produce DL3 data from DL2.

  optional arguments:
    -h, --help            show this help message and exit
    --config_file CONFIG_FILE
                          A configuration file
    --obs_time OBS_TIME   An observation time written as (value.unit), e.g.
                          '50.h'
    --pipeline PIPELINE   Name of the pipeline that has produced the DL2 files.
    --debug               Print debugging information.

Currently the only accepted pipeline is *EventDisplay*.
The configuration file to be used should be `config.yaml` (:ref:`resources`).

lst_performance
---------------
  
The usage is the following,

.. code-block::
  
  >$ python $PYIRF/lst_performance.py -h
  usage: lst_performance.py [-h] [--obs_time OBS_TIME] --dl2_gamma
                          DL2_GAMMA_FILENAME --dl2_proton DL2_PROTON_FILENAME
                          --dl2_electron DL2_ELECTRON_FILENAME
                          [--outdir OUTDIR] [--conf CONFIG_FILE]

  Make performance files

  optional arguments:
    -h, --help            show this help message and exit
    --obs_time OBS_TIME   Observation time in hours
    --dl2_gamma DL2_GAMMA_FILENAME, -g DL2_GAMMA_FILENAME
                          path to the gamma dl2 file
    --dl2_proton DL2_PROTON_FILENAME, -p DL2_PROTON_FILENAME
                          path to the proton dl2 file
    --dl2_electron DL2_ELECTRON_FILENAME, -e DL2_ELECTRON_FILENAME
                          path to the electron dl2 file
    --outdir OUTDIR, -o OUTDIR
                          Output directory
    --conf CONFIG_FILE, -c CONFIG_FILE
                          Optional. Path to a config file. If none is given, the
                          standard performance config is used
                          
.. todo::
  
  Add further information.

Reference/API
-------------

.. automodapi:: pyirf.scripts
    :no-inheritance-diagram:
    :include-all-objects:
