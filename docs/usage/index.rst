.. _usage:

Workflow and usage
==================

.. toctree::
    :maxdepth: 1
    :hidden:

    EventDisplay
    lstchain_irf
    protopipe
    
You should have a working installation of pyirf (:ref:`install`).

For bugs and new features, please contribute to the project (:ref:`contribute`).

How to?
-------

In order to use *pyirf*, you need lists of events at the
DL2 level, e.g. events with a minimal number of information:

 * Direction
 * True energy
 * Reconstructed energy
 * Score/gammaness

In general a number of event lists are needed in order to estimate
the performance of the instruments:

 * Gamma-rays, considered as signal
 * Protons, considered as a source of diffuse background
 * Electrons, considered as a source of diffuse background

 At the moment we support the following pipelines:

  * LSTchain (:ref:`lstchain_irf`),
  * EventDisplay (:ref:`EventDisplay`).

  .. * protopipe (:ref:`protopipe`).
