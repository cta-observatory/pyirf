.. _contribute:

How to contribute
=================


Issue Tracker
-------------

We use the `GitHub issue tracker <https://github.com/cta-observatory/pyirf>`__.

If you found a bug or you are missing a feature, please check the existing
issues and then open a new one or contribute to the existing issue.

Development procedure
---------------------


We use the standard `GitHub workflow <https://guides.github.com/introduction/flow/>`__.

If you are not part of the ``cta-observatory`` organization,
you need to fork the repository to contribute.
See the `GitHub tutorial on forks <https://docs.github.com/en/github/getting-started-with-github/fork-a-repo>`__ if you are unsure how to do this.

#. When you find something that is wrong or missing

    - Go to the issue tracker  and check if an issue already exists for your bug or feature
    - In general it is always better to anticipate a PR with a new issue and link the two

#. To work on a bug fix or new feature, create a new branch, add commits and open your pull request

   - If you think your pull request is good to go and ready to be reviewed,
     you can directly open it as normal pull request.

   - You can also open it as a “Draft Pull Request”, if you are not yet finished
     but want to get early feedback on your ideas.

   - Especially when working on a bug, it makes sense to first add a new
     test that fails due to the bug and in a later commit add the fix showing
     that the test is then passing.
     This helps understanding the bug and will prevent it from reappearing later.

#. Wait for review comments and then implement or discuss requested changes.


We use `Travis CI <https://travis-ci.com/github/cta-observatory/pyirf>`__ to
run the unit tests and documentation building automatically for every pull request.
Passing unit tests and coverage of the changed code are required for all pull requests.

Further details
---------------

Please also have a look at the

- ``ctapipe`` `development guidelines <https://cta-observatory.github.io/ctapipe/development/index.html>`__
- The `Open Gamma-Ray Astronomy data formats <https://gamma-astro-data-formats.readthedocs.io/en/latest/>`__
  which also describe the IRF formats and their definitions.
- `CTA IRF working group wiki (internal) <https://forge.in2p3.fr/projects/instrument-response-functions/wiki>`__

Benchmarks
----------

- `Comparison with EventDisplay <../notebooks/comparison_with_EventDisplay.ipynb>`__ | *comparison_with_EventDisplay.ipynb*
