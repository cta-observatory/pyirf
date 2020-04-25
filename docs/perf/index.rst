.. _perf:

====
perf
====

Introduction
============
The perf module contains classes that are used to estimate the performance of the
instrument. There are tools to handle the determination of the best-cutoffs
to separate gamma and the background (protons + electrons), to produce the
instrument response functions (IRFs) and to estimate the sensitivity.

The following responses are computed:

 * Effective area as a function of true energy
 * Migration matrix as a function of true and reconstructed energy
 * Point spread function computed with a 68 % radius containment as a function of reconstructed energy
 * Background rate as a function of reconstructed energy

The point-like source sensitivity is estimated with the gammapy_
library. We describe below how to estimate the performance of the instruments
and we describe in details how it is done.

How to?
=======
In order to estimate performance, you need lists of events at the
DL2 level, e.g. events with a minimal number of information:

 * Direction
 * True energy
 * Reconstructed energy
 * Score/gammaness

Three different tables of events, in HDF5_ format, are needed in order to estimate
the performance of the instruments:

 * Gamma-rays, considered as signal
 * Protons, considered as a source of diffuse background
 * Electrons, considered as a source of diffuse background

A single script called `make_performance.py` is used to estimate the performance:

.. code-block:: bash

    >$ ./make_performance.py --help
    usage: make_performance.py [-h] --config_file CONFIG_FILE [--wave | --tail]

An example configuration file is shown below with some comments:

.. code-block:: yaml

    general:
     # Directory with input data file
     indir: '/Users/julien/Documents/WorkingDir/Tools/python/protopipe/ana/prod_full_array_north_zen20_az0_complete/output/dl2/'
     # Template name for input file
     template_input_file: 'dl2_{}_{}_merged.h5'  # will be filled with mode and particle type
     # Directory for output files
     outdir: '/Users/julien/Documents/WorkingDir/Tools/python/protopipe/ana/prod_full_array_north_zen20_az0_complete/output/perf_mult3/'
     # Output table name
     output_table_name: 'table_best_cutoff'

    analysis:
     # Additional cut on data
     cut_on_data: 'NTels_reco >= 3'
     # Theta square cut optimisation (opti, fixed, r68)
     thsq_opt:
      type: 'r68'
      value: 0.2  # In degree, necessary for type fixed
     # Normalisation between ON and OFF regions
     alpha: 0.2
     # Observation time to estimate best cuts corresponding to best sensitivity
     obs_time:
      value: 50
      unit: 'h'
     min_sigma: 5  # minimal number of sigma
     min_excess: 10  # minimal number of excess events (nsig > min_excess)
     bkg_syst: 0.05  # percentage of bkg sytematics (nsig > bkg_syst * n_bkg)
     # Binning in reco energy (bkg rate, migration matrix)
     ereco_binning:  # TeV
      emin: 0.012589254
      emax: 199.52623
      nbin: 21
     # Binning for true energy (eff area, migration matrix, PSF)
     etrue_binning:  # TeV
      emin: 0.019952623
      emax: 199.52623
      nbin: 42

    # Information about simulation. In the future, everything should be store
    # in the input files (as meta data and as histogram)
    particle_information:
     # Simulated gamma-rays
     gamma:
      n_events_per_file: 1000000  #  number of files, 10**5 * 10
      e_min: 0.003  # energy min in TeV
      e_max: 330  # energy max in TeV
      gen_radius: 1400  # maximal impact parameter in meter
      diff_cone: 0  # diffuse cone, 0 or point-like, in degree
      gen_gamma: 2  # spectral index for input spectra
     # Simulated protons
     proton:
      n_events_per_file: 4000000  #  number of files, 2 * 10**5 * 20
      e_min: 0.004  # energy min in TeV
      e_max: 600  # energy max in TeV
      gen_radius: 1900  # maximal impact parameter in meter
      diff_cone: 10  # diffuse cone, 0 or point-like, in degree
      gen_gamma: 2  # spectral index for input spectra
      offset_cut: 1.  # maximum offset to consider particles
     # Simulated electrons
     electron:
      n_events_per_file: 2000000  #  number of files, 10**5 * 20
      e_min: 0.003  # energy min in TeV
      e_max: 330  # energy max in TeV
      gen_radius: 1900  # maximal impact parameter in meter
      diff_cone: 10  # diffuse cone, 0 or point-like, in degree
      gen_gamma: 2  # spectral index for input spectra
      offset_cut: 1.  # maximum offset to consider particles

    column_definition:
     # Column name for true energy
     mc_energy: 'mc_energy'
     # Column name for reconstructed energy
     reco_energy: 'reco_energy'
     # Column name for the angular distance in the camera between the true
     # position and the reconstructed position
     angular_distance_to_the_src: 'xi'
     # Column name for classification output
     classification_output:
      name: 'gammaness'
      range: [0, 1]  # needed to bin data and for diagnostic plots

Best cutoffs determination
=================================
The criteria to determine the best cut off which has been use up to now
is to obtain the minimal flux with a :math:`5\sigma` detection
in a given observation time for a Crab-like source template.
In order to find the cuts, a preliminary step is to weight the events according
to what has been measured in real life.
Both the weighting of the events and the determination of best cutoffs
are done with the `CutsOptimisation` class.
The application of the cuts and the generation of diagnostic plots
are handled respectively by the `CutsApplicator` and the `CutsDiagnostic`
classes.

Weighting of events
-------------------
The simulations are generated with a given spectral index, typically of 2 to get
high statistics at high energy. We thus need to flatten the spectral distribution
of the particle and then correct it to match reality. This is done
by computing a weight :math:`w(E)`, which is a function of true energy, for each particle.
It can be expressed as the multiplication of the ratios of fluxes and
the ratios of observation time, each ratio being defined as the division between
the 'observation quantity' and the Monte-Carlo quantity:

.. math::

   w(E)  &=  \frac{\phi_{\text{Obs}}}{\phi_{\text{MC}}} \times \frac{T_{\text{Obs}}}{T_{\text{MC}}} \\
              &=  A_\text{MC} \times I_\theta \times E^\Gamma \times I_E \times T_\text{Obs} \times \phi_\text{Obs}(E)/ N_\text{MC}

where the different quantities are defined as follow:

 * :math:`A_\mathrm{MC}`: MC generator area
 * :math:`I_\theta = 2 \pi (1-\cos\theta)`: angular phase space factor for diffuse flux (:math:`I_\theta = 1` for point-sources)
 * :math:`E^\Gamma`: accounts for the fact that the MC events have been drawn with an :math:`E^{-\Gamma}` spectrum
 * :math:`\Gamma`: spectral index of the MC generator
 * :math:`I_E = \int_{E_\text{min}}^{E_\text{max}} E^{-\Gamma} dE = (E_\text{max}^{(1-\Gamma)} - E_\text{min}^{(1-\Gamma)}) / (1-\Gamma)`: energy phase space factor
 * :math:`T_\text{Obs}`: assumed observation time
 * :math:`N_\text{MC}`: number of generated MC events
 * :math:`\phi_\text{Obs}(E)`: expected differential flux to be matched

The differential diffuse spectrum of the cosmic-rays comes from
`K. Bernlöhr et al. (2013) <http://adsabs.harvard.edu/abs/2013APh....43..171B>`_.
Concerning the gamma-rays, the Crab Nebula spectrum is usually took from `HEGRA
measurements <https://arxiv.org/abs/astro-ph/0407118>`_.

Best cutoffs
------------
Since the gamma/hadron separation power vary a lot with energy, the
best cutoffs to separate the gamma-rays and the background will be determined for
different bins in reconstructed energy. Those energy intervals are typically
chosen to get 5 bins per decade in energy.

Since we are dealing with point-like sources, a cut on the angular distance
between the event position and the position of the source is done. Here the
user have the choice to optimise the angular cut as a function of energy (MARS-like),
to use the point-spread function with a 68 % containment radius (EvtDisplay-like),
or to use a fixed angular cut. Up to now, no optimisation is done for the minimal
event multiplicity for an event to be took into account in the analysis (MARS-like).
A fixed cut on the multiplicity is done.

For each energy bin and for a given angular cut, the following procedure is done
to compute the minimal flux reachable:

 1. Correct the number of protons and electrons to match the region of interest
    define by the angular cut, e.g. the ON region
 2. Compute the gamma and the background (protons + electrons) efficiencies as
    a function of the score/gammaness (fine binning)
 3. Compute the lowest flux reachable in a given observation time and with a
    detection level of :math:`5\sigma` according to the `Li & Ma (1983)
    <http://adsabs.harvard.edu/abs/1983ApJ...272..317L>`_ formula (17). Scale
    the flux by the corresponding minimal number of photons if one of those
    requirements is not met:

     * :math:`N_\text{excess} \geq N_\text{min}`
     * :math:`N_\text{excess} \geq \text{syst}_\text{bkg} \times N_\text{bkg}`
 4. Select the cutoff and the angular cut which give the lowest flux

To look for the minimal flux, the score/gammaness are sampled according to
fixed value in background efficiencies, 15 values between 0.05 and 0.5 by step
of 0.05, as in the MARS analysis for CTA. We do not go below 0.05 since we
want some robustness against fluctuations.

In the two requirements, the number of excess is defined by
:math:`N_\text{excess}=N_\text{ON} - \alpha \times N_\text{OFF}`, :math:`\alpha`
is the normalisation between the ON and the OFF regions, :math:`N_\text{bkg}`
is the number of background in the ON regions and :math:`\text{syst}_\text{bkg}`
is the systematics on the number of background events. Typical values for
:math:`\alpha`, :math:`N_\text{min}` and :math:`\text{syst}_\text{bkg}` are 1/5,
10 and 5 %, respectively.

The final results of the procedure is a FITS table containing the results of the
optimisation for each energy bin such as, the minimal and maximal energy range of
the bin, the best cutoff, the best angular cut, with the corresponding excess,
background, etc.

Cutoffs application
-------------------
A dedicated class, called `CutsApplicator`, is in charge to apply the cuts
to the different event lists. Each event will be flagged according to the
different cuts it will pass, e.g. score/gammaness and angular cuts.
The output tables will be further processed when the user will generate IRFs.

Diagnostics
-----------
Several diagnostic plots are generated during the procedure.
For each energy bin both the efficiencies and the rates as a function
of the score/gammaness, as well as characteristics of the bin, are automatically
generated.
The efficiencies and the angular cuts are all also plotted against the
reconstructed energy in order to control the optimisation procedure
(e.g. background free regions, evolution of background efficiencies
with the angular cut, etc.).

Responses of the instrument
===========================

The instrument response functions characterise the performance of the instrument.
In addition, there are needed to estimate the sensitivity of the array.
A proposition for the CTA IRF data format is available
`here <https://gamma-astro-data-formats.readthedocs.io/>`_.
The IRF are stored as an HDU (Header Data Unit) list in a FITS
(Flexible Image Transport System) file.
Up to now we only considered analyses built with ON-axis gamma-ray simulations
and dedicated to the study of point-like sources.
We do not have offset dependency on the IRF for the moment and thus do not have
axes corresponding to offset bins.
Except for the migration matrix for which we hacked a bit the generation of the
EnergyDispersion object, since it expects offset axes, everything goes pretty
much smoothly.

Responses
---------

Effective area
^^^^^^^^^^^^^^
The collection area, which is proportional to the gamma-ray efficiency
of detection, is computed as a function of the true energy. The events which
are considered are the one passing the threshold of the best cutoff plus
the angular cuts.

Energy migration matrix
^^^^^^^^^^^^^^^^^^^^^^^
The migration matrix, ratio of the reconstructed energy over the true energy
as a function of the true energy, is computed with the events passing the
threshold of the best cutoff plus the angular cuts.
In order to be able to use the energy dispersion with Gammapy_ to compute
the sensitvity we artificially created fake offset bins.
I guess that Gammapy_ should be able to reaf IRF with single offset.

Background
^^^^^^^^^^
The question to consider whether the bakground is an IRF or not. Since here it
is needed to estimate the sensitivity of the instrument we consider it is included
in the IRFs.
Here a simple HDU containing the background (protons + electrons) rate as a
function of the reconstructed energy is generated.
The events which are considered are the one passing the threshold of
the best cutoff and the angular cuts.

Point spread function
^^^^^^^^^^^^^^^^^^^^^
Here we do not really need the PSF to compute the sensitivity, since the angular
cuts are already applied to the effective area, the energy migration matrix
and the background.
I chose to represent the PSF with a containment radius of 68 % as a function
of reconstructed energy as a simple HDU.
The events which are considered are the one passing the threshold of
the best cutoff.

We should generate the recommended IRF, e.g. parametrised as what? Apparently
there are multiple solutions
(see `here, <https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/psf/index.html>`_).

Angular cut values
^^^^^^^^^^^^^^^^^^
To be implemented: `<https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/point_like/index.html>`_

Sensitivity
-----------
The sensitivity is computed with the Gammapy software.

What could be improved?
=======================
 * `Data format for IRFs <https://gamma-astro-data-formats.readthedocs.io/>`_
 * Propagation and reading SIMTEL informations (meta-data, histograms)
   directly in the DL2
 * Implement optimisation on the number of telescopes to consider an event
 *

Reference/API
=============

.. automodapi::
   :no-inheritance-diagram:

.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
.. _Gammapy: https://gammapy.org/
.. _data format: https://gamma-astro-data-formats.readthedocs.io/
