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
If you use this software for a publication, please cite our publication and the software
DOI published to Zenodo that corresponds to the version of the software you are using:

- latest : |doilatest|
- all versions: `Please visit Zenodo <https://zenodo.org/search?q=parent.id%3A4740755&f=allversions%3Atrue&l=list&p=1&s=10&sort=version>`_ and select the correct version

At this point, our latest publication is the `2023 ICRC proceeding <https://doi.org/10.22323/1.444.0618>`_, which you can
cite using the following bibtex entry, especially if using functionalities from ``pyirf.interpolation``:

.. code::

   @inproceedings{pyirf-icrc-2023,
     author = {Dominik, Rune Michael and Linhoff, Maximilian and Sitarek, Julian},
     title = {Interpolation of Instrument Response Functions for the Cherenkov Telescope Array in the Context of pyirf},
     usera = {for the CTA Consortium},
     doi = {10.22323/1.444.0703},
     booktitle = {Proceedings, 38th International Cosmic Ray Conference},
     year=2023,
     volume={444},
     number={618},
     location={Nagoya, Japan},
   }


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


.. |doilatest| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4740755.svg
  :target: https://doi.org/10.5281/zenodo.4740755
