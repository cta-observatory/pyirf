.. _contribute:

How to contribute
=================

.. toctree::
   :hidden:

   repo.rst

Development procedure
---------------------

A common way to add your contribution would look like this:

1. install *pyirf* in developer mode (:ref:`developer`)
2. start using it!
3. when you find something that is wrong or missing

  - go to the Projects or Issues tab of the GitHub repository (:ref:`repo`) and check if
    it is already popped out
  - in general it is always better to anticipate a PR with a new issue and link the two

4. branch from the master branch and start working on the code
5. create a PR from your branch to the project's official repository

This will trigger a number of operations on the Continuous Integration (CI)
pipeline, which are related to the quality of pushed modifications:

- documentation
- unit-testing
- benchmarks

So your PR should always come with:
 - a unit-test for each (at least) new method, function or class you introduce,
 - same for docstrings
 - execution (locally on your machine, for the moment) of the benchmarks

Please, at the time of your first contribution, add your first and last
name together with your contact email in the `AUTHORS.rst` file that you find
in the documentation folder of the project.

Further details
---------------

- Unit-tests are supposed to cover the whole code and all its possibilities
  in terms of cases, arguments, ecc.. This is ensured by a check on their
  *coverage* which we should always aim to maximize and keep stable (ideally to 100%)
- Benchmarks instead check for the quality and performance of the results,
  they come as notebooks stored for the moment under the *notebooks* folder
- These guidelines are necessarely quite general in terms of code quality,
  please have a look also to the
  `ctapipe development guidelines <https://cta-observatory.github.io/ctapipe/development/index.html>`_
- for what concerns CTA IRFs have a look to the
  `CTA IRF working group (internal) <https://forge.in2p3.fr/projects/instrument-response-functions/wiki>`_

Benchmarks
-----------

- `Comparison with EventDisplay <../notebooks/comparison_with_EventDisplay.ipynb>`__ | *comparison_with_EventDisplay.ipynb*
