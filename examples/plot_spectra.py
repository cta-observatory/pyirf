import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from pyirf.spectral import (
    IRFDOC_ELECTRON_SPECTRUM,
    IRFDOC_PROTON_SPECTRUM,
    PDG_ALL_PARTICLE,
    CRAB_HEGRA,
    CRAB_MAGIC_JHEAP2015,
    POINT_SOURCE_FLUX_UNIT,
    FLUX_UNIT,
)


cr_spectra = {
    'PDG All Particle Spectrum': PDG_ALL_PARTICLE,
    'ATIC Proton Fit (from IRF Document)': IRFDOC_PROTON_SPECTRUM,
    'Electron Spectrum (from IRFall  Document)': IRFDOC_ELECTRON_SPECTRUM,
}


if __name__ == '__main__':

    energy = np.geomspace(0.01, 100, 1000) * u.TeV

    plt.figure(constrained_layout=True)
    plt.title('Crab Nebula Flux')
    plt.plot(
        energy.to_value(u.TeV),
        CRAB_HEGRA(energy).to_value(POINT_SOURCE_FLUX_UNIT),
        label='HEGRA',
    )
    plt.plot(
        energy.to_value(u.TeV),
        CRAB_MAGIC_JHEAP2015(energy).to_value(POINT_SOURCE_FLUX_UNIT),
        label='MAGIC JHEAP 2015'
    )

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f'E / TeV')
    plt.ylabel(f'Flux / ({POINT_SOURCE_FLUX_UNIT.to_string("latex")})')

    plt.figure(constrained_layout=True)
    plt.title('Cosmic Ray Flux')

    for label, spectrum in cr_spectra.items():

        plt.plot(
            energy.to_value(u.TeV),
            spectrum(energy).to_value(FLUX_UNIT),
            label=label,
        )

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f'E / TeV')
    plt.ylabel(f'Flux / ({FLUX_UNIT.to_string("latex")})')

    plt.show()
