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
from pyirf.cuts import is_selected
import numpy as np
from astropy import table

from pyirf.spectral import PowerLaw, CRAB_HEGRA, IRFDOC_PROTON_SPECTRUM, calculate_event_weights, IRFDOC_ELECTRON_SPECTRUM

T_OBS = 50 * u.hour


particles = {
    'gamma': {
        'file': 'data/gamma_onSource.S.3HB9-FD_ID0.eff-0.fits',
        'target_spectrum': CRAB_HEGRA,
    },
    'proton': {
        'file': 'data/proton_onSource.S.3HB9-FD_ID0.eff-0.fits',
        'target_spectrum': IRFDOC_PROTON_SPECTRUM,
    },
    'electron': {
        'file': 'data/electron_onSource.S.3HB9-FD_ID0.eff-0.fits',
        'target_spectrum': IRFDOC_ELECTRON_SPECTRUM,
    },
}


def main():
    for p in particles.values():
        p['events'], p['simulation_info'] = read_eventdisplay_fits(p['file'])
        p['simulated_spectrum'] = PowerLaw.from_simulation(p['simulation_info'], T_OBS)
        p['events']['weight'] = calculate_event_weights(
            p['events']['true_energy'], p['target_spectrum'], p['simulated_spectrum']
        )

    # sensitivity binning
    bins_e_reco = add_overflow_bins(create_bins_per_decade(
        10**-1.9 * u.TeV, 10**2.31 * u.TeV, bins_per_decade=5
    ))

    # calculate theta (angular distance from source pos to reco pos)

    for p in particles.values():
        tab = p['events']
        tab['theta'] = angular_separation(
            tab['true_az'], tab['true_alt'],
            tab['reco_az'], tab['reco_alt'],
        )
        tab['bin_reco_energy'] = calculate_bin_indices(
            tab['reco_energy'], bins_e_reco
        )

    cuts = {
        'theta': {
            'operator': 'le',
            'cut_values': np.percentile(particles['gamma']['events']['theta'], 68),
        },
        'gh_score': {
            'operator': 'ge',
            'cut_values': 0.0,
        }
    }
    print("Using the cuts:", cuts)

    for k, p in particles.items():
        tab = p['events']
        tab['selected'] = is_selected(tab, cuts, tab['bin_reco_energy'])
        print(f'Remaining {k}s: {np.count_nonzero(tab["selected"])} of {len(tab)}')

    gammas = particles['gamma']['events']
    signal = gammas[gammas['selected']]
    signal_hist = create_histogram_table(signal, bins_e_reco, 'reco_energy')

    background = table.vstack([
        particles['proton']['events'],
        particles['electron']['events']
    ])
    background_hist = create_histogram_table(
        background[background['selected']], bins_e_reco, 'reco_energy'
    )

    sensitivity = calculate_sensitivity(signal_hist, background_hist, 1, T_OBS)
    sensitivity['flux_sensitivity'] = sensitivity['relative_sensitivity'] * CRAB_HEGRA(sensitivity['reco_energy_center'])

    sensitivity.meta['EXTNAME'] = 'SENSITIVITY'
    sensitivity.write('sensitivity.fits', overwrite=True)

    # calculate sensitivity for best cuts

    # calculate IRFs for the best cuts

    # write OGADF output file


if __name__ == '__main__':
    main()
