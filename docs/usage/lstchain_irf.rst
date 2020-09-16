.. _lstchain_irf:

=====================================
How to build IRFs from lstchain files
=====================================


Install the alpha release of pyirf (v0.1.0-alpha):
(creating a new env is optional)

.. code-block:: bash

    PYIRF_VER=v0.1.0-alpha
    wget https://raw.githubusercontent.com/cta-observatory/pyirf/$PYIRF_VER/environment.yml
    conda env create -n pyirf -f environment.yml
    conda activate pyirf
    pip install  https://github.com/cta-observatory/pyirf/archive/$PYIRF_VER.zip


Once you have generated DL2 files using lstchain v0.5.x for gammas, protons and electrons, you may use the script:

.. code-block:: bash

    python pyirf/scripts/lst_performance.py -g dl2_gamma.h5 -p dl2_proton.h5 -e dl2_electron.h5 -o .


This will create a subdirectory with some control plots and the file irf.fits.gz containing the IRFs.


Authors are aware that there are numerous caveat at the moment.
If you are interested in generating IRFs, you contribution to improve pyirf is most welcome.
