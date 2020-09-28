.. _contribute:

How to contribute
=================


Issue Tracker
-------------

We use the `GitHub issue tracker <https://github.com/cta-observatory/pyirf>`__
for individual issues and the `GitHub Projects page <https://github.com/cta-observatory/pyirf/projects>`_ can give you a quick overview.

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


Running the tests and looking at coverage
-----------------------------------------

For more immediate feedback, you should run the tests locally before pushing,
as builds on travis take quite long.

To run the tests locally, make sure you have the `tests` extras installed and then
run

.. code:: bash

    $ pytest -v


To also inspect the coverage, run

.. code:: bash

    $ pytest --cov=pyirf --cov-report=html -v

This will create a coverage report in html form in the ``htmlcov`` directory,
which you can serve locally using

.. code:: bash

    $ python -m http.server -d htmlcov

After this, you can view the report in your browser by visiting the url printed
to the terminal.


Building the documentation
--------------------------

This documentation uses sphinx and restructured text.
For an Introduction, see the `Sphinx documentation <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.

To build the docs locally, enter the ``docs`` directory and call:

.. code:: bash

    make html

Some changes require a full remake of the documentation, for that call

.. code:: bash

    make clean html

If you created or deleted file or submodule, you also need to remove the
``api`` directory, it will be regenerated automatically.

Make sure the docs are built without warnings from sphinx, as these
will be treated as errors in the build in the CI system as they most often
result in broken styling.

To look at the docs, use

.. code:: bash

    $ python -m http.server _build/html

and visit the printed URL in your browser.



Further details
---------------

Please also have a look at the

- ``ctapipe`` `development guidelines <https://cta-observatory.github.io/ctapipe/development/index.html>`__
- The `Open Gamma-Ray Astronomy data formats <https://gamma-astro-data-formats.readthedocs.io/en/latest/>`__
  which also describe the IRF formats and their definitions.
- ``ctools`` `documentation page on IRFs <http://cta.irap.omp.eu/ctools/users/user_manual/irf_cta.html>`__
- `CTA IRF working group wiki (internal) <https://forge.in2p3.fr/projects/instrument-response-functions/wiki>`__

- `CTA IRF Description Document for Prod3b (internal) <https://gitlab.cta-observatory.org/cta-consortium/aswg/documentation/internal_reports/irfs-reports/prod3b-irf-description>`__


Benchmarks
----------

- :doc:`notebooks/comparison_with_EventDisplay`
