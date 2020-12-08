==========================================
pyirf |ci| |codacy| |coverage| |doilatest|
==========================================

.. |ci| image:: https://github.com/cta-observatory/pyirf/workflows/CI/badge.svg?branch=master
  :target: https://github.com/cta-observatory/pyirf/actions?query=workflow%3ACI+branch%3Amaster
.. |codacy| image:: https://app.codacy.com/project/badge/Grade/669fef80d3d54070960e66351477e383
  :target: https://www.codacy.com/gh/cta-observatory/pyirf/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cta-observatory/pyirf&amp;utm_campaign=Badge_Grade
.. |coverage| image:: https://codecov.io/gh/cta-observatory/pyirf/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/cta-observatory/pyirf
.. |doilatest| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4304466.svg
  :target: https://doi.org/10.5281/zenodo.4304466


*pyirf* is a python3-based library for the generation of Instrument Response
Functions (IRFs) and sensitivities for the
`Cherenkov Telescope Array (CTA) <https://www.cta-observatory.org/>`_ .

Thanks to its simple input/output and modular function-based structure,
it can be potentially used to process also data from other Imaging Atmospheric
Cherenkov Telescopes (IACTs).

- **Source code:** https://github.com/cta-observatory/pyirf
- **Documentation:** https://cta-observatory.github.io/pyirf/

Citing this software
--------------------
If you use a released version of this software for a publication,
please cite it by using the corresponding DOI:

- v0.4.0 : |doilatest|
