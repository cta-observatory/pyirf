'''
Example for using pyirf to calculate IRFS and sensitivity from EventDisplay DL2 fits
files produced from the root output by this script:

https://github.com/Eventdisplay/Converters/blob/master/DL2/generate_DL2_file.py
'''
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation
from pyirf.io.eventdisplay import read_eventdisplay_fits
from pyirf.binning import create_bins_per_decade, add_overflow_bins, calculate_bin_indices, create_histogram_table
from pyirf.sensitiviy import calculate_sensitivity
import numpy as np

from pyirf.spectral import PowerLaw, CRAB_HEGRA, IRFDOC_PROTON_SPECTRUM, calculate_event_weights

T_OBS = 50 * u.hour


def main():
    # read gammas
    gammas, gamma_info = read_eventdisplay_fits('data/gamma_onSource.S.3HB9-FD_ID0.eff-0.fits')
    simulated_spectrum = PowerLaw.from_simulation(gamma_info, T_OBS)
    gammas['weight'] = calculate_event_weights(
        true_energy=gammas['true_energy'],
        target_spectrum=CRAB_HEGRA,
        simulated_spectrum=simulated_spectrum,
    )


    # read protons
    protons, proton_info = read_eventdisplay_fits('data/proton_onSource.S.3HB9-FD_ID0.eff-0.fits')
    simulated_spectrum = PowerLaw.from_simulation(proton_info, T_OBS)
    protons['weight'] = calculate_event_weights(
        true_energy=protons['true_energy'],
        target_spectrum=IRFDOC_PROTON_SPECTRUM,
        simulated_spectrum=simulated_spectrum,
    )

    # sensitivity binning
    bins_e_reco = add_overflow_bins(create_bins_per_decade(1e-2 * u.TeV, 200 * u.TeV, 5))

    # calculate theta (angular distance from source pos to reco pos)

    for tab in (gammas, protons):
        tab['theta'] = angular_separation(
            tab['true_az'], tab['true_alt'],
            tab['reco_az'], tab['reco_alt'],
        )
        tab['bin_reco_energy'] = calculate_bin_indices(
            tab['reco_energy'], bins_e_reco
        )

    theta_cut = np.percentile(gammas['theta'], 68)
    print(f'Using theta cut: {theta_cut.to(u.deg):.2f}')
    gh_cut = 0.0

    for tab in (gammas, protons):
        tab['selected'] = (tab['gh_score'] > gh_cut) & (tab['theta'] < theta_cut)

    print(f'Remaining gammas: {np.count_nonzero(gammas["selected"])} of {len(gammas)}')
    print(f'Remaining protons: {np.count_nonzero(protons["selected"])} of {len(protons)}')

    signal = create_histogram_table(gammas[gammas['selected']], bins_e_reco, 'reco_energy')
    background = create_histogram_table(protons[protons['selected']], bins_e_reco, 'reco_energy')

    sensitivity = calculate_sensitivity(signal, background, 1, T_OBS)
    sensitivity['flux_sensitivity'] = sensitivity['relative_sensitivity'] * CRAB_HEGRA(sensitivity['reco_energy_center'])

    print(sensitivity)

    # calculate sensitivity for best cuts

    # calculate IRFs for the best cuts

    # write OGADF output file


if __name__ == '__main__':
    main()
