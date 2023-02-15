.. _interpolation:

Interpolation of IRFs
=====================

This module contains functions to interpolate from a set of IRFs for different
conditions to a new IRF. While ``pyirf.interpolators`` contains the implementations 
of interpolation algorithms, ``pyirf.interpolation`` applies them to IRF components.

This can e.g. be used to interpolate IRFs for zenith angles of 20° and 40°
to 30°.

The functions support an arbitrary number of interpolation dimensions.


Reference/API
-------------

.. automodapi:: pyirf.interpolation
   :no-inheritance-diagram:

.. automodapi:: pyirf.interpolators
