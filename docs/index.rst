.. meta::
    :github_url: https://github.com/cta-observatory/pyirf

Welcome to pyirf's documentation!
=================================

`pyirf` is a prototype for the generation of Instrument Response Functions (IRFs)
for the `Cherenkov Telescope Array <https://www.cta-observatory.org/>`__
(CTA).
The package is being developed and tested by members of the CTA consortium and
is a spin-off of the analog sub-process of the
`pipeline protopype <https://cta-observatory.github.io/protopipe/>`_.

Its main features are currently to

  * find the best cutoff in gammaness/score, to discriminate between signal
    and background, as well as the angular cut to obtain the best sensitivity
    for a given amount of observation time and a given template for the
    source of interest (:ref:`cut_optimization`)
  * compute the instrument response functions, effective area,
    point spread function and energy resolution (:ref:`irf`)
  * estimate the sensitivity of the array (:ref:`sensitivity`),

with plans to extend its capabilities to reach the requirements of the
future observatory.

.. Should we add the following or is it too soon? --->
.. Event though the initial efforts are focused on CTA, it is potentially possible
.. to extend the capabilities of `pyirf` to other IACTs instruments as well.

The source code is hosted on a `GitHub repository <https://github.com/cta-observatory/pyirf>`__, to
which this documentation is linked.

.. warning::
  This is not yet stable code, so expect large and rapid changes.

Citing this software
--------------------

.. |doilatest| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4740755.svg
  :target: https://doi.org/10.5281/zenodo.4740755
.. |doi_v0.5.0| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4748994.svg
  :target: https://doi.org/10.5281/zenodo.4748994
.. |doi_v0.4.0| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4304466.svg
  :target: https://doi.org/10.5281/zenodo.4304466

If you use a released version of this software for a publication,
please cite it by using the corresponding DOI.

- latest : |doilatest|
- v0.5.0 : |doi_v0.5.0|
- v0.4.0 : |doi_v0.4.0|

.. toctree::
  :maxdepth: 1
  :caption: Overview
  :name: _pyirf_intro

  install
  introduction
  examples
  notebooks/index
  contribute
  changelog
  AUTHORS


.. toctree::
  :maxdepth: 1
  :caption: API Documentation
  :name: _pyirf_api_docs

  irf/index
  sensitivity
  benchmarks/index
  cuts
  cut_optimization
  simulation
  spectral
  statistics
  binning
  io/index
  interpolation
  gammapy
  utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
