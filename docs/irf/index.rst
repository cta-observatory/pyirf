.. _irf:

Instrument Response Functions
=============================


Effective Area
--------------

The collection area, which is proportional to the gamma-ray efficiency
of detection, is computed as a function of the true energy. The events which
are considered are the ones passing the threshold of the best cutoff plus
the angular cuts.

Energy Dispersion Matrix
------------------------

The energy dispersion matrix, ratio of the reconstructed energy over the true energy
as a function of the true energy, is computed with the events passing the
threshold of the best cutoff plus the angular cuts.

The corresponding energy migration matrix can be build from the dispersion matrix.


Point Spread Function
---------------------

The PSF describes the probability of measuring a gamma ray
of a given true energy and true position at a reconstructed position.


Reference/API
-------------

.. automodapi:: pyirf.irf
   :no-inheritance-diagram:
