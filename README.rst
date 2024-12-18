==========================================
pyirf |ci| |codacy| |coverage| |doilatest|
==========================================

.. |ci| image:: https://github.com/cta-observatory/pyirf/workflows/CI/badge.svg?branch=main
  :target: https://github.com/cta-observatory/pyirf/actions?query=workflow%3ACI+branch%3Amain
.. |codacy| image:: https://app.codacy.com/project/badge/Grade/669fef80d3d54070960e66351477e383
  :target: https://www.codacy.com/gh/cta-observatory/pyirf/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cta-observatory/pyirf&amp;utm_campaign=Badge_Grade
.. |coverage| image:: https://codecov.io/gh/cta-observatory/pyirf/branch/main/graph/badge.svg
  :target: https://codecov.io/gh/cta-observatory/pyirf
.. |doilatest| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4740755.svg
  :target: https://doi.org/10.5281/zenodo.4740755

*pyirf* is a python library for the generation of Instrument Response
Functions (IRFs) and sensitivities for the
`Cherenkov Telescope Array (CTA) <https://www.cta-observatory.org/>`_ .

Thanks to its simple input/output and modular function-based structure,
it can be used to process also data from other Imaging Atmospheric
Cherenkov Telescopes (IACTs).

- **Source code:** https://github.com/cta-observatory/pyirf
- **Documentation:** https://cta-observatory.github.io/pyirf/

Citing this software
--------------------
If you use a released version of this software for a publication,
please cite it by using the corresponding DOI:

- latest : |doilatest|
- all versions: `Please visit Zenodo <https://zenodo.org/record/5833284>`_ and select the correct version

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
