.. _changelog:

Changelog
=========

We use a one-line description of every pull request.

.. to obtain the merged PRs since a specific release, e.g. v0.2.0 use
.. `$ git log --merges --first-parent master --oneline  master...v0.2.0`
.. to obtain the contributor, use
.. `$ git shortlog -sne master...v0.2.0

.. RELEASE TEMPLATE
..
.. `X.Y.Z < github link >`__ (Month Day, YEAR)
.. -------------------------------------------
..
.. Summary
.. +++++++
..
.. - Released Month Day, YEAR
.. - N contributors
..
.. **Description**
..
.. . . .
..
.. **Contributors:**
..
.. In alphabetical order by last name:
..
.. - . . .
..
.. Pull Requests
.. +++++++++++++
..
.. - [#XXX] TITLE (AUTHOR)


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
