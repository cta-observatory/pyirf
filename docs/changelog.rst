.. _changelog:

Changelog
=========

We use a one-line description of every pull request.


.. _pyirf_0p8p0_release:

`0.8.0 <https://github.com/cta-observatory/pyirf/releases/tag/v0.8.0>`__ (2022-11-16)
-------------------------------------------------------------------------------------

Summary
+++++++

- Released November 16th, 2022
- 2 Contributors

Contributors
++++++++++++

- Rune Michael Dominik
- Maximilian Nöthe

Description
+++++++++++

This release updates the optional gammapy dependency to the 1.x, after the recent gammapy 1.0 release.
Basic support for interpolating rad max tables was added as well as some improvements for dealing with
diffuse gamma-ray simulations.

merged pull requests
++++++++++++++++++++

- `#202 https://github.com/cta-observatory/pyirf/pull/202`_ Update gammapy requirement to 1.x
- `#180 https://github.com/cta-observatory/pyirf/pull/180`_ Interpolate RAD_MAX tables
- `#199 https://github.com/cta-observatory/pyirf/pull/199`_ Improve under/overflow and nan handling in binning related functions
- `#200 https://github.com/cta-observatory/pyirf/pull/200`_ Fix unit handling of powerlaw
- `#197 https://github.com/cta-observatory/pyirf/pull/197`_ Improvements for calculating sensitivity on diffuse gammas
- `#193 https://github.com/cta-observatory/pyirf/pull/193`_ Fix ang res unit handling, fixes #192
- `#189 https://github.com/cta-observatory/pyirf/pull/189`_ Ignore under/overflow events in table operations
- `#188 https://github.com/cta-observatory/pyirf/pull/188`_ Include endpoint in create_bins_per_decade if it matches the regular spacing, fixes #187


.. _pyirf_0p7p0_release:

`0.7.0 <https://github.com/cta-observatory/pyirf/releases/tag/v0.7.0>`__ (2022-04-14)
-------------------------------------------------------------------------------------

Summary
+++++++

- Released April 4th, 2022
- 2 Contributors

Contributors
++++++++++++

- Rune Michael Dominik
- Maximilian Nöthe

Description
+++++++++++


The main feature of this release is new methods for interpolation of
pdf-like IRFs, i.e. energy dispersion and psf table.
These methods replace the methods not suitable for interpolation of probability
densities introduced in pyirf 0.5.0.

This release also adds support for astropy 5.0.

merged pull requests
++++++++++++++++++++

- `#174 <https://github.com/cta-observatory/pyirf/pull/174>`_ Adapted quantile interpolation
- `#177 <https://github.com/cta-observatory/pyirf/pull/177>`_ Add interpolate_psf_table to ``__all__``
- `#175 <https://github.com/cta-observatory/pyirf/pull/175>`_ Allow and test astropy=5


.. _pyirf_0p6p0_release:

`0.6.0 <https://github.com/cta-observatory/pyirf/releases/tag/v0.6.0>`__ (2021-01-10)
-------------------------------------------------------------------------------------

Summary
+++++++

- Released January 10th, 2022
- 5 Contributors

Contributors
++++++++++++

- Maximilian Nöthe
- Michael Punch
- Michele Peresano
- Julian Sitarek
- Gernot Maier

Description
+++++++++++

This release is mainly a maintenance release with few changes to the functionality.
The main feature is the update to gammapy 0.19 in the tests and the utility functions
to get gammapy IRF classes from pyirf output.

In the future, we will more directly interface with and use gammapy, now that
the science tool decision was taken.


merged pull requests
++++++++++++++++++++

- `#164 <https://github.com/cta-observatory/pyirf/pull/164>`_ Update to gammapy 0.19
- `#168 <https://github.com/cta-observatory/pyirf/pull/168>`_ Enable intersphinx, use to link to gammapy docs
- `#171 <https://github.com/cta-observatory/pyirf/pull/171>`_ Add release-drafter action
- `#172 <https://github.com/cta-observatory/pyirf/pull/172>`_ Replace outdated link to redmine by xwiki link
- `#165 <https://github.com/cta-observatory/pyirf/pull/165>`_ Do not require private DL2 event display output anymore for unit tests
- `#162 <https://github.com/cta-observatory/pyirf/pull/162>`_ Refactor hist normalization, remove assert from library code
- `#160 <https://github.com/cta-observatory/pyirf/pull/160>`_ Add missing docs pages
- `#156 <https://github.com/cta-observatory/pyirf/pull/156>`_ Interpolate psf
- `#159 <https://github.com/cta-observatory/pyirf/pull/159>`_ GADF URL corrections
- `#154 <https://github.com/cta-observatory/pyirf/pull/154>`_ Fill energy and/or angular resolution tables with NaNs if input events table is empty


.. _pyirf_0p5p0_release:

`0.5.0 <https://github.com/cta-observatory/pyirf/releases/tag/v0.5.0>`__ (2021-05-05)
-------------------------------------------------------------------------------------

Summary
+++++++

- Released May 5th, 2021
- 4 Contributors

Contributors
++++++++++++

- Julian Sitarek
- Maximilian Nöthe
- Michele Peresano
- Abelardo Moralejo Olaizola

Description
+++++++++++

Main new feature in this release are functions to interpolate grids of IRFs
to, e.g. for different pointing directions, to new IRFs.
Supported at the moment are effective area and energy dispersion.

We also added a function to compute energy bias and resolution from the
energy dispersion IRF and a new spectrum for cosmic rays: the DAMPE combined
proton and helium spectrum.

The other pull requests are mainly maintenance and a small bugfix.


merged pull requests
++++++++++++++++++++

- `#149 <https://github.com/cta-observatory/pyirf/pull/149>`_ Interpolation docs
- `#141 <https://github.com/cta-observatory/pyirf/pull/141>`_ Interpolate IRFs
- `#144 <https://github.com/cta-observatory/pyirf/pull/144>`_ Add function to compute bias and resolution from energy dispersion
- `#145 <https://github.com/cta-observatory/pyirf/pull/145>`_ Proton+Helium spectrum from DAMPE 2019 ICRC proceeding
- `#148 <https://github.com/cta-observatory/pyirf/pull/148>`_ Use setuptools_scm for versioning
- `#147 <https://github.com/cta-observatory/pyirf/pull/147>`_ Fix benchmark functions for events outside given bins
- `#138 <https://github.com/cta-observatory/pyirf/pull/138>`_ Fix name of deploy build
- `#143 <https://github.com/cta-observatory/pyirf/pull/143>`_ Fix zenodo json
- `#139 <https://github.com/cta-observatory/pyirf/pull/139>`_ Fix how theta cut is calculated in EventDisplay comparison
- `#140 <https://github.com/cta-observatory/pyirf/pull/140>`_ uproot4 -> uproot


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
