.. _developer:

Installation for developers
===========================

If you want to use *pyirf* and also contribute to its development, follow these steps:

  1. Fork the official `repository <https://github.com/cta-observatory/pyirf>`_ has explained `here <https://help.github.com/en/articles/fork-a-repo>`__ (follow all the instructions)
  2. now your local copy is linked to your remote repository (**origin**) and the official one (**upstream**)
  3. create a dedicated environment with ``conda env create -f environment.yml``
  4. activate it with ``conda activate pyirf``
  5. install *pyirf* itself in developer mode with ``pip install -e .``

In this way, you will always use the version of the source code on which you
are working.

Next steps:

 * start using *pyirf* (:ref:`usage`),
 * for bugs and new features, please contribute to the project (:ref:`contribute`).
