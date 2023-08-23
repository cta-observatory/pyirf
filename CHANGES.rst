Pyirf v0.10.0 (2023-08-23)
==========================

This release contains an important bug fix for the energy dispersion computation,
it was wrongly normalized before.

API Changes
-----------

- In prior versions of pyirf, the energy dispersion matrix was normalized to a
  sum of 1 over the migration axis.
  This is wrong, the correct normalization is to an integral of 1, which is fixed now.

  The internal API of the interpolation functions had to be adapted to take in additional
  keywords, mainly the bin edges and the kind of normalization (standard or solid angle cone sections). [`#250 <https://github.com/cta-observatory/pyirf/pull/250>`__]

- Replace single ``viewcone`` argument of ``SimulationInfo`` with
  ``viewcone_min`` and ``viewcone_max``, e.g. to correctly enable
  ring wobble simulations. [`#239 <https://github.com/cta-observatory/pyirf/pull/239>`__]


Bug Fixes
---------

- See above on the energy dispersion change.


New Features
------------

- Add option to specify which containment to use for angular resolution. [`#234 <https://github.com/cta-observatory/pyirf/pull/234>`__]



pyirf 0.9.0 (2023-07-19)
========================


API Changes
-----------

- Change the interpolation API to top-level estimator classes that instantiate
  inter- and extrapolator objects. Drops the ``interpolate_xyz`` functions
  originally used to interpolate a xyz IRF component in favour of a ``XYZEstimator``
  class. Moves data checks from intepolator to estimator classes.

  Direct usage of interpolator objects is now discuraged, use estimator objects instead. [`#228 <https://github.com/cta-observatory/pyirf/pull/228>`__]


Bug Fixes
---------

- Correctly fill n_events in ``angular_resolution``, was always 0 before. [`#231 <https://github.com/cta-observatory/pyirf/pull/231>`__]

- Remove condition that relative sensitivity must be > 1.
  This condition was added by error and resulted in returning
  nan if the flux needed to fulfill the conditions is larger than
  the reference flux used to weight the events. [`#241 <https://github.com/cta-observatory/pyirf/pull/241>`__]


New Features
------------

- Add moment morphing as second interpolation method able to handle discretized PDF 
  components of IRFs. [`#229 <https://github.com/cta-observatory/pyirf/pull/229>`__]

- Add a base structure for extrapolators similar to the interpolation case
  as well as a first extrapolator for parametrized components, extrapolating from the
  nearest simplex in one or two dimensions. [`#236 <https://github.com/cta-observatory/pyirf/pull/236>`__]

- Add an extrapolator for discretized PDF components, extrapolating from the
  nearest simplex in one or two dimensions utilizing the same approach moment morphing
  interpolation uses. [`#237 <https://github.com/cta-observatory/pyirf/pull/237>`__]

- Add a ``DiscretePDFNearestNeighborSearcher`` and a ``ParametrizedNearestNeighborSearcher`` to support nearest neighbor approaches 
  as alternatives to inter-/ and extrapolation [`#232 <https://github.com/cta-observatory/pyirf/pull/232>`__]



Maintenance
-----------

- Drop python 3.8 support in accordance with `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_ [`#243 <https://github.com/cta-observatory/pyirf/pull/243>`__]



pyirf 0.8.1 (2023-03-16)
========================


New Features
------------

- Migrating the interpolation methods from ``pyirf.interpolation`` to interpolator 
  objects, allowing for later inheritance for new algorithms and reusability. [`#210 <https://github.com/cta-observatory/pyirf/pull/210>`__]


Maintenance
-----------

- Add and enable ``towncrier`` in CI. [`#207 <https://github.com/cta-observatory/pyirf/pull/207>`__]

- Add a fixture containing three IRFs from `the prod5 IRF data-release <https://zenodo.org/record/5499840>`_
  for unit testing. Specifically the fixture contains the contents of:

   - Prod5-North-20deg-AverageAz-4LSTs.180000s-v0.1.fits.gz.
   - Prod5-North-40deg-AverageAz-4LSTs.180000s-v0.1.fits.gz
   - Prod5-North-60deg-AverageAz-4LSTs.180000s-v0.1.fits.gz

   The user has to download these irfs to ``irfs/`` using ``download_irfs.py``,
   github's CI does so automatically and caches them for convenience. [`#211 <https://github.com/cta-observatory/pyirf/pull/211>`__]


Older releases
==============

For releases between v0.4.1 and v0.8.1, please refer to `the GitHub releases page <https://github.com/cta-observatory/pyirf/releases>`_.


.. _pyirf_0p4p1_release:

`0.4.1 <https://github.com/cta-observatory/pyirf/releases/tag/v0.4.1>`__ (2021-03-22)
-------------------------------------------------------------------------------------

Summary
+++++++

- Released March 22nd, 2021
- 1 Contributors

Contributors
++++++++++++

- Maximilian Nöthe


Merged Pull Requests
++++++++++++++++++++

- `#135 <https://github.com/cta-observatory/pyirf/pull/135>`_ Add functions to convert pyirf results to the corresponding gammapy classes
- `#137 <https://github.com/cta-observatory/pyirf/pull/137>`_ Add example notebook for calculating point-lile IRFs from the FACT open data


.. _pyirf_0p4p0_release:

`0.4.0 <https://github.com/cta-observatory/pyirf/releases/tag/v0.4.0>`__ (2020-11-09)
-------------------------------------------------------------------------------------

Summary
+++++++

- Released November 11th, 2020
- 2 Contributors

Contributors
++++++++++++

In order of number of commits:

- Maximilian Nöthe
- Michele Peresano


Description
+++++++++++

This release is an important update that introduces three
changes in the cut optimization, background estimation and sensitivity calculation.

Together, these changes bring the calculated sensitivities much closer to the ones calculated by
EventDisplay.

* Scale the relative flux calculated to reach the target sensitivity
  up if the requirements on the minimum number of signal events are not met.
  Essentially, instead of always calculating the flux that
  yields ``target_sensitivity`` and then checking if the two other conditions are met,
  we increase the required flux to meet the other requirements.
  This can result in new sensitivities where before pyirf would report no sensitivities,
  and report better sensitivities everywhere where the event number conditions where not
  met before at the target significance.
  The best sensitivity now is the lowest flux that just barely satisfies all
  requirements (so is at the minimum requirement of one of the three).

* Differentiate between `reco_source_fov_offset` and `true_source_fov_offset`,
  using the former for background rates and the latter for everything concerning
  signal events.

* Change ``optimize_gh_cut`` to do the optimization in terms of efficiency and
  limit this efficiency to max. 80 % in the EventDisplay comparison.


Smaller improvements also include:

* It is now possible to include a ``particle_type`` column in the event lists,
  which will result in additionally reporting all event counts also per ``particle_type``.
  E.g. if ``particle_type`` is included in the background table consisting of both
  electrons and protons, ``estimate_background`` will not only report ``n_background(_weighted)``
  but also ``n_electron(_weighted)`` and ``n_proton(_weighted)``

* ``relative_sensitivity`` now supports vectorized application and broadcasting
  of inputs, as previously wrongly advertized in the docstring.


Related news
++++++++++++

GammaPy ``0.18.0`` was released and includes fixes for IRF axis orders.
The output of ``pyirf`` in GADF fits format can now be read by gammapy without
problems.
The workarounds for installing GammaPy is also no longer needed.


Merged Pull Requests
++++++++++++++++++++

Feature changes
"""""""""""""""

- `#110 <https://github.com/cta-observatory/pyirf/pull/110>`_ Optimize cuts in efficiency steps with maximum efficiency of 80% for EventDisplay comparison
- `#104 <https://github.com/cta-observatory/pyirf/pull/104>`_ Scale flux for conditions, differenatiate reco and true source_fov_offset
- `#108 <https://github.com/cta-observatory/pyirf/pull/108>`_ Add counts / weighted counts per particle type
- `#107 <https://github.com/cta-observatory/pyirf/pull/107>`_ Small update to installation instructions
- `#106 <https://github.com/cta-observatory/pyirf/pull/106>`_ Use vectorize for relative_sensitivity

Project maintenance
"""""""""""""""""""

- `#102 <https://github.com/cta-observatory/pyirf/pull/102>`_ Require astropy >= 4.0.2
- `#100 <https://github.com/cta-observatory/pyirf/pull/100>`_ Fix deploy condition in travis yml


.. _pyirf_0p3p0_release:

`0.3.0 <https://github.com/cta-observatory/pyirf/releases/tag/v0.3.0>`__ (2020-10-05)
-------------------------------------------------------------------------------------

Summary
+++++++

- Released October 5th, 2020
- 5 Contributors

Contributors
++++++++++++

In order of number of commits:

- Maximilian Nöthe
- Michele Peresano
- Noah Biederbeck
- Lukas Nickel
- Gaia Verna


Description
+++++++++++

This release is the result of the IRF sprint week in September 2020.
Many bug fixes and improvements were made to the code.

As the target for the sprint week was to reproduce the approach of ``EventDisplay`` and
the resulting IRFs, one scheme of cut optimization is implemented.
The ``examples/calculate_eventdisplay_irfs.py`` should follow the approach
of ``EventDisplay`` closely and shows what is currently implemented in ``pyirf``.
In the central and upper energy range, ``pyirf`` now reproduces the ``EventDisplay`` sensitivity
exactly, the lower energy bins still show some disagreement.
The cut optimization seems not yet to be the same as EventDisplay's and will be further investigated.
This example could be used as a starting point if you also want to do cut optimization for best sensitivity.


At least one version of each IRF is now implemented and can be stored in the GADF format.
Computation of full-enclosure IRFs should be possible but is of now not yet tested
on a reference dataset.


Merged Pull Requests
++++++++++++++++++++

- `#97 <https://github.com/cta-observatory/pyirf/pull/97>`_ Store correct signal amount, store information on which checks failed for sensitivity bins (Maximilian Nöthe)
- `#96 <https://github.com/cta-observatory/pyirf/pull/96>`_ Add integration test (Michele Peresano)
- `#98 <https://github.com/cta-observatory/pyirf/pull/98>`_ Remove option point_like for psf (Maximilian Nöthe)
- `#95 <https://github.com/cta-observatory/pyirf/pull/95>`_ Cut updates (Maximilian Nöthe)
- `#91 <https://github.com/cta-observatory/pyirf/pull/91>`_ Fix conditions to take relative sensitivity into account, fixes #90 (Maximilian Nöthe)
- `#89 <https://github.com/cta-observatory/pyirf/pull/89>`_ Fix brentq returning the lower bound of 0 for flat li ma function (Maximilian Nöthe)
- `#85 <https://github.com/cta-observatory/pyirf/pull/85>`_ Improve comparison to EventDisplay (Maximilian Nöthe)
- `#75 <https://github.com/cta-observatory/pyirf/pull/75>`_ Add a function to check a table for required cols / units (Maximilian Nöthe)
- `#86 <https://github.com/cta-observatory/pyirf/pull/86>`_ Fix Li & Ma significance for n_off = 0 (Maximilian Nöthe)
- `#76 <https://github.com/cta-observatory/pyirf/pull/76>`_ Feature resample histogram (Noah Biederbeck, Lukas Nickel)
- `#79 <https://github.com/cta-observatory/pyirf/pull/79>`_ Fix integration of power law pdf in simulations.py (Gaia Verna)
- `#80 <https://github.com/cta-observatory/pyirf/pull/80>`_ Estimate unique runs taking pointing pos into account (Maximilian Nöthe)
- `#71 <https://github.com/cta-observatory/pyirf/pull/71>`_ Background estimation (Maximilian Nöthe)
- `#78 <https://github.com/cta-observatory/pyirf/pull/78>`_ Change argument order in create_rad_max_hdu (Lukas Nickel)
- `#77 <https://github.com/cta-observatory/pyirf/pull/77>`_ Calculate optimized cut on only the events surviving gh separation (Maximilian Nöthe)
- `#68 <https://github.com/cta-observatory/pyirf/pull/68>`_ Effective area 2d (Maximilian Nöthe)
- `#67 <https://github.com/cta-observatory/pyirf/pull/67>`_ Add method integrating sim. events in FOV bins (Maximilian Nöthe)
- `#63 <https://github.com/cta-observatory/pyirf/pull/63>`_ Verify hdus using ogadf-schema (Maximilian Nöthe)
- `#58 <https://github.com/cta-observatory/pyirf/pull/58>`_ Implement Background2d (Maximilian Nöthe)
- `#52 <https://github.com/cta-observatory/pyirf/pull/52>`_ Add sections about tests, coverage and building docs to docs (Maximilian Nöthe)
- `#46 <https://github.com/cta-observatory/pyirf/pull/46>`_ Add PyPI deploy and metadata (Maximilian Nöthe)


.. _pyirf_0p2p0_release:

`0.2.0 <https://github.com/cta-observatory/pyirf/releases/tag/v0.2.0>`__ (2020-09-27)
-------------------------------------------------------------------------------------

Summary
+++++++

- Released September 27th, 2020
- 4 Contributors

Contributors
++++++++++++

In order of number of commits:

-  Maximilian Nöthe
- Michele Peresano
- Lukas Nickel
- Hugo van Kemenade


Description
+++++++++++

For this version, pyirf's API was completely rewritten from scratch,
merging code from several projects (pyirf, pyfact, fact-project/irf) to provide a library to compute IACT
IRFs and sensitivity and store them in the GADF data format.

The class based API using a configuration file was replaced by a finer grained
function based API.

Implemented are point-like IRFs and sensitivity.

This release was the starting point for the IRF sprint week in September 2020,
where the refactoring continued.


Merged Pull Requests
++++++++++++++++++++

- `#36 <https://github.com/cta-observatory/pyirf/pull/36>`_ Start refactoring pyirf (Maximilian Nöthe, Michele Peresano, Lukas Nickel)
- `#35 <https://github.com/cta-observatory/pyirf/pull/35>`_ Cleanup example notebook (Maximilian Nöthe, Michele Peresano, Lukas Nickel)
- `#37 <https://github.com/cta-observatory/pyirf/pull/37>`_ Move to python >= 3.6 (Hugo van Kemenade)



.. _pyirf_0p1p0_release:

`0.1.0 <https://github.com/cta-observatory/pyirf/releases/tag/v0.1.0>`__ (2020-09-16)
-------------------------------------------------------------------------------------

This is a pre-release.

- Released September 16th, 2020


.. _pyirf_0p1p0alpha_prerelease:

`0.1.0-alpha <https://github.com/cta-observatory/pyirf/releases/tag/v0.1.0-alpha>`__ (2020-05-27)
-------------------------------------------------------------------------------------------------

Summary
+++++++

This is a pre-release.

- Released May 27th, 2020
- 3 contributors

Description
+++++++++++

- Started basic maintenance
- Started refactoring
- First tests with CTA-LST data

Contributors
++++++++++++

In alphabetical order by last name:

- Lea Jouvin
- Michele Peresano
- Thomas Vuillaume
