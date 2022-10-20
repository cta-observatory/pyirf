"""
Functions and classes for calculating spectral weights
"""
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from pkg_resources import resource_filename
from astropy.table import QTable

from .utils import cone_solid_angle

#: Unit of a point source flux
#:
#: Number of particles per Energy, time and area
POINT_SOURCE_FLUX_UNIT = (1 / u.TeV / u.s / u.m ** 2).unit

#: Unit of a diffuse flux
#:
#: Number of particles per Energy, time, area and solid_angle
DIFFUSE_FLUX_UNIT = POINT_SOURCE_FLUX_UNIT / u.sr


__all__ = [
    "POINT_SOURCE_FLUX_UNIT",
    "DIFFUSE_FLUX_UNIT",
    "calculate_event_weights",
    "PowerLaw",
    "LogParabola",
    "PowerLawWithExponentialGaussian",
    "CRAB_HEGRA",
    "CRAB_MAGIC_JHEAP2015",
    "PDG_ALL_PARTICLE",
    "IRFDOC_PROTON_SPECTRUM",
    "IRFDOC_ELECTRON_SPECTRUM",
    "TableInterpolationSpectrum",
    "DAMPE_P_He_SPECTRUM",
]


@u.quantity_input(true_energy=u.TeV)
def calculate_event_weights(true_energy, target_spectrum, simulated_spectrum):
    r"""
    Calculate event weights

    Events with a certain ``simulated_spectrum`` are reweighted to ``target_spectrum``.

    .. math::
        w_i = \frac{\Phi_\text{Target}(E_i)}{\Phi_\text{Simulation}(E_i)}

    Parameters
    ----------
    true_energy: astropy.units.Quantity[energy]
        True energy of the event
    target_spectrum: callable
        The target spectrum. Must be a allable with signature (energy) -> flux
    simulated_spectrum: callable
        The simulated spectrum. Must be a callable with signature (energy) -> flux

    Returns
    -------
    weights: numpy.ndarray
        Weights for each event
    """
    return (target_spectrum(true_energy) / simulated_spectrum(true_energy)).to_value(
        u.one
    )


class PowerLaw:
    r"""
    A power law with normalization, reference energy and index.
    Index includes the sign:

    .. math::

        \Phi(E, \Phi_0, \gamma, E_\text{ref}) =
        \Phi_0 \left(\frac{E}{E_\text{ref}}\right)^{\gamma}

    Attributes
    ----------
    normalization: astropy.units.Quantity[flux]
        :math:`\Phi_0`,
    index: float
        :math:`\gamma`
    e_ref: astropy.units.Quantity[energy]
        :math:`E_\text{ref}`
    """

    @u.quantity_input(
        normalization=[DIFFUSE_FLUX_UNIT, POINT_SOURCE_FLUX_UNIT], e_ref=u.TeV
    )
    def __init__(self, normalization, index, e_ref=1 * u.TeV):
        """Create a new PowerLaw spectrum"""
        if index > 0:
            raise ValueError(f'Index must be < 0, got {index}')

        self.normalization = normalization
        self.index = index
        self.e_ref = e_ref

    @u.quantity_input(energy=u.TeV)
    def __call__(self, energy):
        e = (energy / self.e_ref).to_value(u.one)
        return self.normalization * e**self.index

    @classmethod
    @u.quantity_input(obstime=u.hour, e_ref=u.TeV)
    def from_simulation(cls, simulated_event_info, obstime, e_ref=1 * u.TeV):
        """
        Calculate the flux normalization for simulated events drawn
        from a power law for a certain observation time.
        """
        e_min = simulated_event_info.energy_min
        e_max = simulated_event_info.energy_max
        index = simulated_event_info.spectral_index
        n_showers = simulated_event_info.n_showers
        viewcone = simulated_event_info.viewcone

        if viewcone.value > 0:
            solid_angle = 2 * np.pi * (1 - np.cos(viewcone)) * u.sr
            unit = DIFFUSE_FLUX_UNIT
        else:
            solid_angle = 1
            unit = POINT_SOURCE_FLUX_UNIT

        A = np.pi * simulated_event_info.max_impact ** 2

        delta = e_max ** (index + 1) - e_min ** (index + 1)
        nom = (index + 1) * e_ref ** index * n_showers
        denom = (A * obstime * solid_angle) * delta

        return cls(normalization=(nom / denom).to(unit), index=index, e_ref=e_ref,)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.normalization} * (E / {self.e_ref})**{self.index})"

    @u.quantity_input(inner=u.rad, outer=u.rad)
    def integrate_cone(self, inner, outer):
        """Integrate this powerlaw over solid angle in the given cone

        Parameters
        ----------
        inner : astropy.units.Quantity[angle]
            inner opening angle of cone
        outer : astropy.units.Quantity[angle]
            outer opening angle of cone

        Returns
        -------
        integrated : PowerLaw
            A new powerlaw instance with new normalization with the integration
            result.
        """
        if not self.normalization.unit.is_equivalent(DIFFUSE_FLUX_UNIT):
            raise ValueError("Can only integrate a diffuse flux over solid angle")

        solid_angle = cone_solid_angle(outer) - cone_solid_angle(inner)

        return PowerLaw(
            normalization=self.normalization * solid_angle,
            index=self.index,
            e_ref=self.e_ref,
        )



class LogParabola:
    r"""
    A log parabola flux parameterization.

    .. math::

        \Phi(E, \Phi_0, \alpha, \beta, E_\text{ref}) =
        \Phi_0 \left(
            \frac{E}{E_\text{ref}}
        \right)^{\alpha + \beta \cdot \log_{10}(E / E_\text{ref})}

    Attributes
    ----------
    normalization: astropy.units.Quantity[flux]
        :math:`\Phi_0`,
    a: float
        :math:`\alpha`
    b: float
        :math:`\beta`
    e_ref: astropy.units.Quantity[energy]
        :math:`E_\text{ref}`
    """

    @u.quantity_input(
        normalization=[DIFFUSE_FLUX_UNIT, POINT_SOURCE_FLUX_UNIT], e_ref=u.TeV
    )
    def __init__(self, normalization, a, b, e_ref=1 * u.TeV):
        """Create a new LogParabola spectrum"""
        self.normalization = normalization
        self.a = a
        self.b = b
        self.e_ref = e_ref

    @u.quantity_input(energy=u.TeV)
    def __call__(self, energy):
        e = (energy / self.e_ref).to_value(u.one)
        return self.normalization * e ** (self.a + self.b * np.log10(e))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.normalization} * (E / {self.e_ref})**({self.a} + {self.b} * log10(E / {self.e_ref}))"


class PowerLawWithExponentialGaussian(PowerLaw):
    r"""
    A power law with an additional Gaussian bump.
    Beware that the Gaussian is not normalized!

    .. math::

        \Phi(E, \Phi_0, \gamma, f, \mu, \sigma, E_\text{ref}) =
        \Phi_0 \left(
            \frac{E}{E_\text{ref}}
        \right)^{\gamma}
        \cdot \left(
            1 + f \cdot
            \left(
                \exp\left(
                    \operatorname{Gauss}(\log_{10}(E / E_\text{ref}), \mu, \sigma)
                \right) - 1
            \right)
        \right)

    Where :math:`\operatorname{Gauss}` is the unnormalized Gaussian distribution:

    .. math::
        \operatorname{Gauss}(x, \mu, \sigma) = \exp\left(
            -\frac{1}{2} \left(\frac{x - \mu}{\sigma}\right)^2
        \right)

    Attributes
    ----------
    normalization: astropy.units.Quantity[flux]
        :math:`\Phi_0`,
    a: float
        :math:`\alpha`
    b: float
        :math:`\beta`
    e_ref: astropy.units.Quantity[energy]
        :math:`E_\text{ref}`
    """

    @u.quantity_input(
        normalization=[DIFFUSE_FLUX_UNIT, POINT_SOURCE_FLUX_UNIT], e_ref=u.TeV
    )
    def __init__(self, normalization, index, e_ref, f, mu, sigma):
        """Create a new PowerLawWithExponentialGaussian spectrum"""
        super().__init__(normalization=normalization, index=index, e_ref=e_ref)
        self.f = f
        self.mu = mu
        self.sigma = sigma

    @u.quantity_input(energy=u.TeV)
    def __call__(self, energy):
        power = super().__call__(energy)
        log10_e = np.log10(energy / self.e_ref)
        # ROOT's TMath::Gauss does not add the normalization
        # this is missing from the IRFDocs
        # the code used for the plot can be found here:
        # https://gitlab.cta-observatory.org/cta-consortium/aswg/irfs-macros/cosmic-rays-spectra/-/blob/master/electron_spectrum.C#L508
        gauss = np.exp(-0.5 * ((log10_e - self.mu) / self.sigma) ** 2)
        return power * (1 + self.f * (np.exp(gauss) - 1))

    def __repr__(self):
        s = super().__repr__()
        gauss = f"Gauss(log10(E / {self.e_ref}), {self.mu}, {self.sigma})"
        return s[:-1] + f" * (1 + {self.f} * (exp({gauss}) - 1))"


class TableInterpolationSpectrum:
    """
    Interpolate flux points to obtain a spectrum.

    By default, flux is interpolated linearly in log-log space.
    """

    def __init__(
        self, energy, flux, log_energy=True, log_flux=True, reference_energy=1 * u.TeV
    ):
        """Create a new TableInterpolationSpectrum spectrum"""
        self.energy = energy
        self.flux = flux
        self.flux_unit = flux.unit
        self.log_energy = log_energy
        self.log_flux = log_flux
        self.reference_energy = reference_energy

        x = (energy / reference_energy).to_value(u.one)
        y = flux.to_value(self.flux_unit)

        if log_energy:
            x = np.log10(x)

        if log_flux:
            y = np.log10(y)

        self.interp = interp1d(x, y, bounds_error=False, fill_value="extrapolate")

    def __call__(self, energy):

        x = (energy / self.reference_energy).to_value(u.one)

        if self.log_energy:
            x = np.log10(x)

        y = self.interp(x)

        if self.log_flux:
            y = 10 ** y

        return u.Quantity(y, self.flux_unit, copy=False)

    @classmethod
    def from_table(
        cls, table: QTable, log_energy=True, log_flux=True, reference_energy=1 * u.TeV
    ):
        return cls(
            table["energy"],
            table["flux"],
            log_energy=log_energy,
            log_flux=log_flux,
            reference_energy=reference_energy,
        )

    @classmethod
    def from_file(
        cls, path, log_energy=True, log_flux=True, reference_energy=1 * u.TeV
    ):
        return cls.from_table(
            QTable.read(path),
            log_energy=log_energy,
            log_flux=log_flux,
            reference_energy=reference_energy,
        )


#: Power Law parametrization of the Crab Nebula spectrum as published by HEGRA
#:
#: From "The Crab Nebula and Pulsar between 500 GeV and 80 TeV: Observations with the HEGRA stereoscopic air Cherenkov telescopes",
#: Aharonian et al, 2004, ApJ 614.2
#: doi.org/10.1086/423931
CRAB_HEGRA = PowerLaw(
    normalization=2.83e-11 / (u.TeV * u.cm ** 2 * u.s), index=-2.62, e_ref=1 * u.TeV,
)

#: Log-Parabola parametrization of the Crab Nebula spectrum as published by MAGIC
#:
#: From "Measurement of the Crab Nebula spectrum over three decades in energy with the MAGIC telescopes",
#: Aleksìc et al., 2015, JHEAP
#: https://doi.org/10.1016/j.jheap.2015.01.002
CRAB_MAGIC_JHEAP2015 = LogParabola(
    normalization=3.23e-11 / (u.TeV * u.cm ** 2 * u.s), a=-2.47, b=-0.24,
)


#: All particle spectrum
#:
#: (30.2) from "The Review of Particle Physics (2020)"
#: https://pdg.lbl.gov/2020/reviews/rpp2020-rev-cosmic-rays.pdf
PDG_ALL_PARTICLE = PowerLaw(
    normalization=1.8e4 / (u.GeV * u.m ** 2 * u.s * u.sr), index=-2.7, e_ref=1 * u.GeV,
)

#: Proton spectrum definition defined in the CTA Prod3b IRF Document
#:
#: From "Description of CTA Instrument Response Functions (Production 3b Simulation)", section 4.3.1
#: https://gitlab.cta-observatory.org/cta-consortium/aswg/documentation/internal_reports/irfs-reports/prod3b-irf-description
IRFDOC_PROTON_SPECTRUM = PowerLaw(
    normalization=9.8e-6 / (u.cm ** 2 * u.s * u.TeV * u.sr),
    index=-2.62,
    e_ref=1 * u.TeV,
)

#: Electron spectrum definition defined in the CTA Prod3b IRF Document
#:
#: From "Description of CTA Instrument Response Functions (Production 3b Simulation)", section 4.3.1
#: https://gitlab.cta-observatory.org/cta-consortium/aswg/documentation/internal_reports/irfs-reports/prod3b-irf-description
IRFDOC_ELECTRON_SPECTRUM = PowerLawWithExponentialGaussian(
    normalization=2.385e-9 / (u.TeV * u.cm ** 2 * u.s * u.sr),
    index=-3.43,
    e_ref=1 * u.TeV,
    mu=-0.101,
    sigma=0.741,
    f=1.950,
)

#: Proton + Helium interpolated from DAMPE measurements
#:
#: Datapoints obtained from obtained from:
#: https://inspirehep.net/files/62efc8374ffced58ea7e3a333bfa1217
#: Points are from DAMPE, up to  8 TeV.
#: For higher energies we assume a
#: flattening of the dF/dE*E^2.7 more or less in the middle of the large
#: spread of the available data reported on the same proceeding.
DAMPE_P_He_SPECTRUM = TableInterpolationSpectrum.from_file(
    resource_filename("pyirf", "resources/dampe_p+he.ecsv")
)
