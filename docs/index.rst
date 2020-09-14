.. pyirf documentation master file, created by
   sphinx-quickstart on Sat Apr 25 16:39:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyirf's documentation!
=================================

`pyirf` is a prototype for the generation of Instrument Response Functions (IRFs)
for the `Cherenkov Telescope Array <https://www.cta-observatory.org/>`__
(CTA).
The package is being developed and tested by members of the CTA consortium and
is a spin-off of the analog sub-process of the
`pipeline protopype <https://cta-observatory.github.io/protopipe/>`_.

Its main features are currently to,

  * find the best cutoff in gammaness/score, to discriminate between signal
    and background, as well as the angular cut to obtain the best sensitivity
    for a given amount of observation time and a given template for the
    source of interest (:ref:`perf`)
  * compute the instrument response functions, effective area,
    point spread function and energy resolution (:ref:`perf`)
  * estimate the sensitivity of the array (:ref:`perf`),

with plans to extend its capabilities to reach the requirements of the
future observatory.

.. Should we add the following or is it too soon? --->
.. Event though the initial efforts are focused on CTA, it is potentially possible
.. to extend the capabilities of `pyirf` to other IACTs instruments as well.

The source code is hosted on a `GitHub repository <https://github.com/cta-observatory/pyirf>`__, to
which this documentation is linked.

.. warning::
  This is not yet stable code, so expect large and rapid changes.

.. _pyirf_intro:
.. toctree::
  :caption: Overview
  :maxdepth: 1

  install/index
  usage/index
  contribute/index
  changelog
  AUTHORS

.. _pyirf_structure:
.. toctree::
  :caption: Structure
  :maxdepth: 1

  io/index
  resources/index
  perf/index
  scripts/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
