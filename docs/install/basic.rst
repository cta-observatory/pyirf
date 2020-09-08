.. _basic:

Installation for basic users
============================

.. warning::
  Given that *pyirf* is undergoing fast development, it is likely that you
  will benefit more from a more recent version of the code for now.

  The development version could disrupt functionalities that were working for
  you, but the latest released version could lack some of those you need.

  To install the latest development version go to :ref:`developer`.

If you are a user with no interest in developing *pyirf*, you can start by
downloading the `latest released version <https://github.com/cta-observatory/pyirf/releases>`__

Steps for installation:

  1. uncompress the file which is always called *pyirf-X.Y.Z* depending on version,
  2. enter the folder ``cd pyirf-X.Y.Z``
  3. create a dedicated environment with ``conda env create -f environment.yml``
  4. activate it with ``conda activate pyirf``
  5. install *pyirf* itself with ``pip install .``.

Next steps:

 * get accustomed to the basics (:ref:`perf`),
 * start using *pyirf* (:ref:`usage`),
 * for bugs and new features, please contribute to the project (:ref:`contribute`).
