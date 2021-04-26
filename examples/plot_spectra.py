import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.stats import norm

from pyirf.spectral import (
    DAMPE_P_He_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
    IRFDOC_PROTON_SPECTRUM,
    PDG_ALL_PARTICLE,
    CRAB_HEGRA,
    CRAB_MAGIC_JHEAP2015,
    POINT_SOURCE_FLUX_UNIT,
    DIFFUSE_FLUX_UNIT,
)


cr_spectra = {
    "PDG All Particle Spectrum": PDG_ALL_PARTICLE,
    "ATIC Proton Fit (from IRF Document)": IRFDOC_PROTON_SPECTRUM,
    "DAMPE p + He Table Interpolation": DAMPE_P_He_SPECTRUM
}


if __name__ == "__main__":

    energy = np.geomspace(0.001, 300, 1000) * u.TeV

    plt.figure(constrained_layout=True)
    plt.title("Crab Nebula Flux")
    plt.plot(
        energy.to_value(u.TeV),
        CRAB_HEGRA(energy).to_value(POINT_SOURCE_FLUX_UNIT),
        label="HEGRA",
    )
    plt.plot(
        energy.to_value(u.TeV),
        CRAB_MAGIC_JHEAP2015(energy).to_value(POINT_SOURCE_FLUX_UNIT),
        label="MAGIC JHEAP 2015",
    )

    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("E / TeV")
    plt.ylabel(f'Flux / ({POINT_SOURCE_FLUX_UNIT.to_string("latex")})')

    plt.figure(constrained_layout=True)
    plt.title("Cosmic Ray Flux")

    for label, spectrum in cr_spectra.items():
        unit = energy.unit ** 2 * DIFFUSE_FLUX_UNIT
        plt.plot(
            energy.to_value(u.TeV),
            (spectrum(energy) * energy ** 2).to_value(unit),
            label=label,
        )

    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$E \,\,/\,\, \mathrm{TeV}$")
    plt.ylabel(rf'$E^{2} \cdot \Phi \,\,/\,\,$ ({unit.to_string("latex")})')

    energy = np.geomspace(0.006, 10, 1000) * u.TeV
    plt.figure(constrained_layout=True)
    plt.title("Electron Flux")

    unit = u.TeV ** 2 / u.m ** 2 / u.s / u.sr
    plt.plot(
        energy.to_value(u.TeV),
        (energy ** 3 * IRFDOC_ELECTRON_SPECTRUM(energy)).to_value(unit),
        label="IFAE 2013 (from IRF Document)",
    )

    plt.legend()
    plt.xscale("log")
    plt.xlim(5e-3, 10)
    plt.ylim(1e-5, 0.25e-3)
    plt.xlabel(r"$E \,\,/\,\, \mathrm{TeV}$")
    plt.ylabel(rf'$E^3 \cdot \Phi \,\,/\,\,$ ({unit.to_string("latex")})')
    plt.grid()

    plt.show()
