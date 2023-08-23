import astropy.units as u
import numpy as np

__all__ = [
    'SimulatedEventsInfo',
]


class SimulatedEventsInfo:
    """
    Information about all simulated events, for calculating event weights.

    Attributes
    ----------
    n_showers: int
        Total number of simulated showers. If reuse was used, this
        should already include the reuse.
    energy_min: u.Quantity[energy]
        Lower limit of the simulated energy range
    energy_max: u.Quantity[energy]
        Upper limit of the simulated energy range
    max_impact: u.Quantity[length]
        Maximum simulated impact parameter
    spectral_index: float
        Spectral Index of the simulated power law with sign included.
    viewcone_min: u.Quantity[angle]
        Inner angle of the viewcone
    viewcone_max: u.Quantity[angle]
        Outer angle of the viewcone
    """

    __slots__ = (
        "n_showers",
        "energy_min",
        "energy_max",
        "max_impact",
        "spectral_index",
        "viewcone_min",
        "viewcone_max",
    )

    @u.quantity_input(
        energy_min=u.TeV, energy_max=u.TeV, max_impact=u.m, viewcone_min=u.deg, viewcone_max=u.deg
    )
    def __init__(
        self, n_showers, energy_min, energy_max, max_impact, spectral_index, viewcone_min, viewcone_max,
    ):
        #: Total number of simulated showers, if reuse was used, this must
        #: already include reuse
        self.n_showers = n_showers
        #: Lower limit of the simulated energy range
        self.energy_min = energy_min
        #: Upper limit of the simulated energy range
        self.energy_max = energy_max
        #: Maximum simualted impact radius
        self.max_impact = max_impact
        #: Spectral index of the simulated power law with sign included
        self.spectral_index = spectral_index
        #: Inner viewcone angle
        self.viewcone_min = viewcone_min
        #: Outer viewcone angle
        self.viewcone_max = viewcone_max

        if spectral_index > -1:
            raise ValueError("spectral index must be <= -1")

    @u.quantity_input(energy_bins=u.TeV)
    def calculate_n_showers_per_energy(self, energy_bins):
        """
        Calculate number of showers that were simulated in the given energy intervals

        This assumes the events were generated and from a powerlaw
        like CORSIKA simulates events.

        Parameters
        ----------
        energy_bins: astropy.units.Quantity[energy]
            The interval edges for which to calculate the number of simulated showers

        Returns
        -------
        n_showers: numpy.ndarray
            The expected number of events inside each of the ``energy_bins``.
            This is a floating point number.
            The actual numbers will follow a poissionian distribution around this
            expected value.
        """
        bins = energy_bins
        e_low = bins[:-1]
        e_high = bins[1:]
        e_min = self.energy_min
        e_max = self.energy_max

        integral = _powerlaw_pdf_integral(
            self.spectral_index, e_low, e_high, e_min, e_max
        )

        integral[e_high <= e_min] = 0
        integral[e_low >= e_max] = 0

        mask = (e_high > e_max) & (e_low < e_max)
        integral[mask] = _powerlaw_pdf_integral(
            self.spectral_index, e_low[mask], e_max, e_min, e_max
        )

        mask = (e_high > e_min) & (e_low < e_min)
        integral[mask] = _powerlaw_pdf_integral(
            self.spectral_index, e_min, e_high[mask], e_min, e_max
        )

        return self.n_showers * integral

    def calculate_n_showers_per_fov(self, fov_bins):
        """
        Calculate number of showers that were simulated in the given fov bins.

        This assumes the events were generated uniformly distributed per solid angle,
        like CORSIKA simulates events with the VIEWCONE option.

        Parameters
        ----------
        fov_bins: astropy.units.Quantity[angle]
            The FOV bin edges for which to calculate the number of simulated showers

        Returns
        -------
        n_showers: numpy.ndarray(ndim=2)
            The expected number of events inside each of the ``fov_bins``.
            This is a floating point number.
            The actual numbers will follow a poissionian distribution around this
            expected value.
        """
        fov_bins = fov_bins
        fov_low = fov_bins[:-1]
        fov_high = fov_bins[1:]
        fov_integral = _viewcone_pdf_integral(self.viewcone_min, self.viewcone_max, fov_low, fov_high)
        return self.n_showers * fov_integral

    @u.quantity_input(energy_bins=u.TeV, fov_bins=u.deg)
    def calculate_n_showers_per_energy_and_fov(self, energy_bins, fov_bins):
        """
        Calculate number of showers that were simulated in the given
        energy and fov bins.

        This assumes the events were generated uniformly distributed per solid angle,
        and from a powerlaw in energy like CORSIKA simulates events.

        Parameters
        ----------
        energy_bins: astropy.units.Quantity[energy]
            The energy bin edges for which to calculate the number of simulated showers
        fov_bins: astropy.units.Quantity[angle]
            The FOV bin edges for which to calculate the number of simulated showers

        Returns
        -------
        n_showers: numpy.ndarray(ndim=2)
            The expected number of events inside each of the
            ``energy_bins`` and ``fov_bins``.
            Dimension (n_energy_bins, n_fov_bins)
            This is a floating point number.
            The actual numbers will follow a poissionian distribution around this
            expected value.
        """
        # energy distribution and fov distribution are independent in CORSIKA,
        # so just multiply both distributions.
        e_integral = self.calculate_n_showers_per_energy(energy_bins)
        fov_integral = self.calculate_n_showers_per_fov(fov_bins)
        return e_integral[:, np.newaxis] * fov_integral / self.n_showers

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_showers={self.n_showers}, "
            f"energy_min={self.energy_min:.3f}, "
            f"energy_max={self.energy_max:.2f}, "
            f"spectral_index={self.spectral_index:.1f}, "
            f"max_impact={self.max_impact:.2f}, "
            f"viewcone_min={self.viewcone_min}"
            f"viewcone_max={self.viewcone_max}"
            ")"
        )


def _powerlaw_pdf_integral(index, e_low, e_high, e_min, e_max):
    # strip units, make sure all in the same unit
    e_low = e_low.to_value(u.TeV)
    e_high = e_high.to_value(u.TeV)
    e_min = e_min.to_value(u.TeV)
    e_max = e_max.to_value(u.TeV)

    int_index = index + 1
    normalization = 1 / (e_max ** int_index - e_min ** int_index)
    e_term = e_high ** int_index - e_low ** int_index
    return e_term * normalization


def _viewcone_pdf_integral(viewcone_min, viewcone_max, fov_low, fov_high):
    """
    CORSIKA draws particles in the viewcone uniform per solid angle between
    viewcone_min and viewcone_max, the associated pdf is:

    pdf(theta, theta_min, theta_max) = sin(theta) / (cos(theta_min) - cos(theta_max))
    """
    scalar = np.asanyarray(fov_low).ndim == 0

    fov_low = np.atleast_1d(fov_low)
    fov_high = np.atleast_1d(fov_high)

    if (viewcone_max - viewcone_min).value == 0:
        raise ValueError("Only supported for diffuse simulations")
    else:
        norm = 1 / (np.cos(viewcone_min) - np.cos(viewcone_max))

    inside = (fov_high >= viewcone_min) & (fov_low <= viewcone_max)

    integral = np.zeros(fov_low.shape)
    lower = np.where(fov_low[inside] > viewcone_min, fov_low[inside], viewcone_min)
    upper = np.where(fov_high[inside] < viewcone_max, fov_high[inside], viewcone_max)

    integral[inside] = np.cos(lower) - np.cos(upper)
    integral *= norm

    if scalar:
        return np.squeeze(integral)
    return integral
