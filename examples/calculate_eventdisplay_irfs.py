'''
Example for using pyirf to calculate IRFS and sensitivity from EventDisplay DL2 fits
files produced from the root output by this script:

https://github.com/Eventdisplay/Converters/blob/master/DL2/generate_DL2_file.py
'''
import astropy.units as u
from pyirf.io.eventdisplay import read_eventdisplay_fits

from pyirf.spectral import PowerLaw, CRAB_HEGRA, IRFDOC_PROTON_SPECTRUM, calculate_event_weights


def main():
    # read gammas
    gammas, gamma_info = read_eventdisplay_fits('data/gamma_onSource.S.3HB9-FD_ID0.eff-0.fits')
    simulated_spectrum = PowerLaw.from_simulation(gamma_info, 50 * u.hour)
    gammas['weight'] = calculate_event_weights(
        true_energy=gammas['true_energy'],
        target_spectrum=CRAB_HEGRA,
        simulated_spectrum=simulated_spectrum,
    )

    # read protons
    protons, proton_info = read_eventdisplay_fits('data/proton_onSource.S.3HB9-FD_ID0.eff-0.fits')
    simulated_spectrum = PowerLaw.from_simulation(proton_info, 50 * u.hour)
    protons['weight'] = calculate_event_weights(
        true_energy=protons['true_energy'],
        target_spectrum=IRFDOC_PROTON_SPECTRUM,
        simulated_spectrum=simulated_spectrum,
    )

    # perform cut optimization

    # calculate IRFs for the best cuts

    # write OGADF output file


if __name__ == '__main__':
    main()
