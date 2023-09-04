.. _interpolation:

Interpolation and Extrapolation of IRFs
=======================================

.. currentmodule:: pyirf.interpolation 

This module contains functions to inter- or extrapolate from a set of IRFs for different
conditions to a new IRF. Implementations of interpolation and extrapolation algorithms 
exist as interpolator and extrapolator classes and are applied by top-level estimator 
classes to IRF components. 
Direct usage of the inter- and extrapolator classes is discouraged, as only the estimator classes 
check the data for consistency.

Most methods support an arbitrary number of interpolation dimensions although it 
is strongly advised to limit those for reasonable results.
The herein provided functionalities can e.g. be used to interpolate IRFs for zenith 
angles of 20° and 40° to 30°.


IRF Component Estimator Classes
-------------------------------

.. autosummary::
   :nosignatures:

   EffectiveAreaEstimator       Estimate AEFF tables.
   RadMaxEstimator              Estimate RadMax tables.
   EnergyDispersionEstimator    Estimate 2D EDISPs.
   PSFTableEstimator            Estimate PSF tables.

Each of them is either tailored to parametrized or discrete PDF components though 
inheritance from the respective base class 

.. autosummary::
   :nosignatures:

   ParametrizedComponentEstimator   Parametrized components 
   DiscretePDFComponentEstimator    Discrete PDF components
 

Inter- and Extrapolation Classes
--------------------------------

This module provides inter- and extrapolation classes that can be 
plugged into the estimator classes. 
Not all of these classes support arbitrary grid-dimensions where the grid 
in this context is the grid of e.g. observation parameters like zenith angle and 
magnetic field inclination (this would be a 2D grid) on which template IRFs exist 
and are meant to be inter- or extrapolated.

For parametrized components (Effective Areas and Rad-Max tables) these classes are:

=============================================    ==================  ============    ==================================================================================================
**Name**                                         **Component Type**  **Grid-Dim**    **Note**
=============================================    ==================  ============    ==================================================================================================
:any:`GridDataInterpolator`                      Interpolation       Arbitrary       See also :any:`scipy.interpolate.griddata`.
:any:`ParametrizedNearestSimplexExtrapolator`    Extrapolation       1D or 2D        Linear (1D) or baryzentric (2D) extension outside the grid's convex hull from the nearest simplex.
:any:`ParametrizedNearestNeighborSearcher`       Nearest Neighbor    Arbitrary       Nearest neighbor finder usable instead of inter- and/or extrapolation. 
=============================================    ==================  ============    ==================================================================================================

For components represented by discretized PDFs (PSF and EDISP tables) these classes are:

=============================================    ==================  ============    ==============================================================================
**Name**                                         **Component Type**  **Grid-Dim**    **Note**
=============================================    ==================  ============    ==============================================================================
:any:`QuantileInterpolator`                      Interpolation       Arbitrary       Adaption of [Hol+13]_ and [Rea99]_ to discretized PDFs.
:any:`MomentMorphInterpolator`                   Interpolation       1D or 2D        Adaption of [Baa+15]_ to discretized PDFs.
:any:`MomentMorphNearestSimplexExtrapolator`     Extrapolation       1D or 2D        Extension of [Baa+15]_ beyond the grid's convex hull from the nearest simplex.
:any:`DiscretePDFNearestNeighborSearcher`        Nearest Neighbor    Arbitrary       Nearest neighbor finder usable instead of inter- and/or extrapolation. 
=============================================    ==================  ============    ==============================================================================

.. [Hol+13] B. E. Hollister and A. T. Pang (2013). Interpolation of Non-Gaussian Probability Distributions for Ensemble Visualization.
    https://engineering.ucsc.edu/sites/default/files/technical-reports/UCSC-SOE-13-13.pdf
.. [Rea99] A. L. Read (1999). Linear Interpolation of Histograms.
    Nucl. Instrum. Methods Phys. Res. A 425, 357-360. https://doi.org/10.1016/S0168-9002(98)01347-3
.. [Baa+15] M. Baak, S. Gadatsch, R. Harrington and W. Verkerke (2015). Interpolation between 
    multi-dimensional histograms using a new non-linear moment morphing method
    Nucl. Instrum. Methods Phys. Res. A 771, 39-48. https://doi.org/10.1016/j.nima.2014.10.033


Usage Example
-------------

To create a estimator class for an IRF component not yet implemented, one can simply 
inherit from respective base class.
Consider an example, where one is interested in an estimator for simple Gaussians.
As this is already the scope of the ``DiscretePDFComponentEstimator`` base class and 
for the sake of this demonstration, let the Gaussians come with some 
units attached that need handling:

.. code-block:: python

   import astropy.units as u

   from pyirf.interpolation import DiscretePDFComponentEstimator, MomentMorphInterpolator

   class GaussianEstimatior(DiscretePDFComponentEstimator):
      @u.quantity_input(gaussians=u.m)
      def __init__(
         self,
         grid_points,
         bin_edges,
         gaussians,
         interpolator_cls=MomentMorphInterpolator,
         interpolator_kwargs=None,
         extrapolator_cls=None,
         extrapolator_kwargs=None,
      ):
         if interpolator_kwargs is None:
            interpolator_kwargs = {}

         if extrapolator_kwargs is None:
            extrapolator_kwargs = {}

         self.unit = gaussians.unit

         super().__init__(
            grid_points=grid_points,
            bin_edges=bin_edges,
            binned_pdf=gaussians.to_value(u.m),
            interpolator_cls=interpolator_cls,
            interpolator_kwargs=interpolator_kwargs,
            extrapolator_cls=extrapolator_cls,
            extrapolator_kwargs=extrapolator_kwargs,
         )

      def __call__(self, target_point):
         res = super().__call__(target_point)

         # Return result with correct unit
         return u.Quantity(res, u.m, copy=False).to(self.unit)

This new estimator class can now be used just like any other estimator class already 
implemented in ``pyirf.interpolation``. 
While the ``extrapolator_cls`` argument can be empty when creating an instance of 
``GaussianEstimator``, effectively disabling extrapolation and raising an error in 
case it would be needed regardless, assume the desired extrapolation method to be 
``MomentMorphNearestSimplexExtrapolator``:

.. code-block:: python

   import numpy as np

   from scipy.stats import norm
   from pyirf.interpolation import MomentMorphNearestSimplexExtrapolator 

   bins = np.linspace(-10, 10, 51)
   grid = np.array([[1], [2], [3]])

   gaussians = np.array([np.diff(norm(loc=x, scale=1/x).cdf(bins))/np.diff(bins) for x in grid])

   estimator = GaussianEstimatior(
      grid_points = grid,
      bin_edges = bins,
      gaussians = gaussians * u.m,
      interpolator_cls = MomentMorphInterpolator,
      extrapolator_cls = MomentMorphNearestSimplexExtrapolator
   )

This estimator object can now easily be used to estimate Gaussians at arbitrary target points:

.. code-block:: python

   targets = np.array([[0.9], [1.5]])

   results = u.Quantity([estimator(target).squeeze() for target in targets])


Helper Classes
--------------

.. autosummary::
   :nosignatures:

   PDFNormalization


Base Classes
------------

.. autosummary::
   :nosignatures:

   BaseComponentEstimator
   BaseInterpolator
   ParametrizedInterpolator
   DiscretePDFInterpolator
   BaseExtrapolator
   ParametrizedExtrapolator
   DiscretePDFExtrapolator
   BaseNearestNeighborSearcher


