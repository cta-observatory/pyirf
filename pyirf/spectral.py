'''
Functions and classes for calculating spectral weights
'''
import astropy.units as u
import numpy as np
from scipy.stats import norm


POINT_SOURCE_FLUX_UNIT = (1 / u.TeV / u.s / u.m**2).unit
FLUX_UNIT = POINT_SOURCE_FLUX_UNIT / u.sr


@u.quantity_input(true_energy=u.TeV)
def calculate_event_weights(true_energy, target_spectrum, simulated_spectrum):
    return (
        target_spectrum(true_energy) / simulated_spectrum(true_energy)
    ).to_value(u.one)


class PowerLaw:
    @u.quantity_input(
        flux_normalization=[FLUX_UNIT, POINT_SOURCE_FLUX_UNIT],
        e_ref=u.TeV
    )
    def __init__(self, flux_normalization, spectral_index, e_ref=1 * u.TeV):
        self.flux_normalization = flux_normalization
        self.spectral_index = spectral_index
        self.e_ref = e_ref

    @u.quantity_input(energy=u.TeV)
    def __call__(self, energy):
        return (
            self.flux_normalization
            * (energy / self.e_ref) ** self.spectral_index
        )

    @classmethod
    @u.quantity_input(obstime=u.hour, e_ref=u.TeV)
    def from_simulation(
        cls, simulated_event_info, obstime, e_ref=1 * u.TeV
    ):
        '''
        Calculate the flux normalization for simulated events drawn
        from a power law for a certain observation time.
        '''
        e_min = simulated_event_info.energy_min
        e_max = simulated_event_info.energy_max
        spectral_index = simulated_event_info.spectral_index
        n_showers = simulated_event_info.n_showers
        viewcone = simulated_event_info.viewcone

        if viewcone.value > 0:
            solid_angle = 2 * np.pi * (1 - np.cos(viewcone)) * u.sr
        else:
            solid_angle = 1

        A = np.pi * simulated_event_info.max_impact**2

        delta = e_max**(spectral_index + 1) - e_min**(spectral_index + 1)
        nom = (spectral_index + 1) * e_ref**spectral_index * n_showers
        denom = (A * obstime * solid_angle) * delta

        return cls(
            flux_normalization=nom / denom,
            spectral_index=spectral_index,
            e_ref=e_ref,
        )

    def __repr__(self):
        return f'{self.__class__.__name__}({self.flux_normalization} * (E / {self.e_ref})**{self.spectral_index}'


class LogParabola:
    @u.quantity_input(
        flux_normalization=[FLUX_UNIT, POINT_SOURCE_FLUX_UNIT],
        e_ref=u.TeV
    )
    def __init__(self, flux_normalization, a, b, e_ref=1 * u.TeV):
        self.flux_normalization = flux_normalization
        self.a = a
        self.b = b
        self.e_ref = e_ref

    @u.quantity_input(energy=u.TeV)
    def __call__(self, energy):
        e = (energy / self.e_ref).to_value(u.one)
        return self.flux_normalization * e**(self.a + self.b * np.log10(e))


class PowerLawWithExponentialGaussian(PowerLaw):

    @u.quantity_input(
        flux_normalization=[FLUX_UNIT, POINT_SOURCE_FLUX_UNIT],
        e_ref=u.TeV
    )
    def __init__(self, flux_normalization, spectral_index, e_ref, f, mu, sigma):
        super().__init__(
            flux_normalization=flux_normalization,
            spectral_index=spectral_index,
            e_ref=e_ref
        )
        self.f = f
        self.mu = mu
        self.sigma = sigma

    @u.quantity_input(energy=u.TeV)
    def __call__(self, energy):
        power = super()(energy)
        log10_e = np.log10(energy / self.e_ref)
        gauss = norm.pdf(log10_e, self.mu, self.sigma)
        return power * (1 + self.f * gauss)


# From "The Crab Nebula and Pulsar between 500 GeV and 80 TeV: Observations with the HEGRA stereoscopic air Cherenkov telescopes",
# Aharonian et al, 2004, ApJ 614.2
# doi.org/10.1086/423931
CRAB_HEGRA = PowerLaw(
    flux_normalization=2.83e-11 / (u.TeV * u.cm**2 * u.s),
    spectral_index=-2.62,
    e_ref=1 * u.TeV,
)

# From "Measurement of the Crab Nebula spectrum over three decades in energy with the MAGIC telescopes",
#Aleksìc et al., 2015, JHEAP
# https://doi.org/10.1016/j.jheap.2015.01.002
CRAB_MAGIC_JHEAP2015 = LogParabola(
    flux_normalization=3.23e-11 / (u.TeV * u.cm**2 * u.s),
    a=-2.47,
    b=-0.24,
)

PDG_ALL_PARTICLE = PowerLaw(
    flux_normalization=1.8e4 * FLUX_UNIT,
    spectral_index=-2.7,
    e_ref=1 * u.GeV,
)

# From "Description of CTA Instrument Response Functions (Production 3b Simulation)"
# section 4.3.1
IRFDOC_PROTON_SPECTRUM = PowerLaw(
    flux_normalization=9.8e-6 / (u.cm**2 * u.s * u.TeV * u.sr),
    spectral_index=-2.62,
    e_ref=1 * u.TeV,
)

# section 4.3.2
IRFDOC_ELECTRON_SPECTRUM = PowerLawWithExponentialGaussian(
    flux_normalization=2.385e-9 / (u.TeV * u.cm**2 * u.s * u.sr),
    spectral_index=-3.43,
    e_ref=1 * u.TeV,
    mu=-0.101,
    sigma=0.741,
    f=1.950,
)
