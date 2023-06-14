.. _interpolation:

Interpolation of IRFs
=====================

This module contains functions to interpolate from a set of IRFs for different
conditions to a new IRF. Implementations of interpolation algorithms exist as interpolator
classes and are applied by top-level estimator classes to IRF components. 
Direct usage of the interpolator classes is discuraged, as only the estimator classes 
check the data for consistency.

This can e.g. be used to interpolate IRFs for zenith angles of 20° and 40°
to 30°.

Most methods support an arbitrary number of interpolation dimensions although it 
is strongly advised to limit those for resonable results.


Reference/API
-------------

.. automodapi:: pyirf.interpolation
