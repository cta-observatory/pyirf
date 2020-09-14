.. _EventDisplay:

=============================================
How to build DL3 data from EventDisplay files
=============================================

.. toctree::
   :hidden:

   ../contribute/comparison_with_EventDisplay.ipynb

Retrieve EventDisplay data
--------------------------

DL2
+++

- hosted `here <https://desycloud.desy.de/index.php/s/DzF7EYc9icbdQGo>`__
- La Palma and Paranal datasets, both 20 deg zenith 180 Azimuth
- all datasets are provided in 3 quality levels depending on direction and classification cuts

  + *cut 0*, neither cut applied
  + *cut 1*, passing classification cut and not direction cut
  + *cut 2*, passing both

For details, see `documentation <https://forge.in2p3.fr/projects/cta_analysis-and-simulations/wiki/Eventdisplay_Prod3b_DL2_Lists>`__.

IRFs
++++

- in ROOT format
- download `here <https://forge.in2p3.fr/attachments/download/63418/20190927-LongObsIRFs.tar.gz>`__
- after unpacking the folder contains

  + 3 summary PDF performance plots for different azimuth directions
  + IRFs stored under ``data/WPPhys201890925LongObs``

The ROOT files named
``DESY.d20180113.V3.ID0_180degNIM2LST4MST4SST4SCMST4.prod3b-paranal20degs05b-NN.S.3HB9-FD``
are related to the DL2 data above and replicated for different observing times in seconds.

Launch *pyirf*
--------------

To create the DL3 data you will need to

- copy the configuration file in your working directory,
- modify it according to your setup,
- launch the ``pyirf.scripts.make_DL3`` script.

To produce e.g. DL3 data for 50 hours,

``python $PYIRF/pyirf/scripts/make_DL3.py --config_file config.yml --pipeline EventDisplay --obs_time 50.h``

Results
-------

Your directory tree structure containing the output data should look like this,

::

  .
  ├── config.yml
  └── irf_EventDisplay_Time50h
      ├── diagnostic
      ├── electron_processed.h5
      ├── gamma_processed.h5
      ├── irf.fits.gz
      ├── proton_processed.h5
      └── table_best_cutoff.fits

where the `diagnostic` folder contains some plots with further information.

You can open the single files or check directly the results using the `Comparison with EventDisplay <../contribute/comparison_with_EventDisplay.ipynb>`__ notebook.
