'''
Example for using pyirf to calculate IRFS and sensitivity from EventDisplay DL2 fits
files produced from the root output by this script:

https://github.com/Eventdisplay/Converters/blob/master/DL2/generate_DL2_file.py
'''
import logging
import operator

import numpy as np
from astropy import table
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation

from pyirf.io.eventdisplay import read_eventdisplay_fits
from pyirf.binning import create_bins_per_decade, add_overflow_bins
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from astropy.io import fits

from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)
from pyirf.cut_optimization import optimize_gh_cut


log = logging.getLogger('pyirf')


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
    logging.basicConfig(level=logging.DEBUG)

    for k, p in particles.items():
        log.info(f'Simulated {k.title()} Events:')

        p['events'], p['simulation_info'] = read_eventdisplay_fits(p['file'])
        p['simulated_spectrum'] = PowerLaw.from_simulation(p['simulation_info'], T_OBS)
        p['events']['weight'] = calculate_event_weights(
            p['events']['true_energy'], p['target_spectrum'], p['simulated_spectrum']
        )

        log.info(p['simulation_info'])
        log.info('')

    # calculate theta (angular distance from source pos to reco pos)

    for p in particles.values():
        tab = p['events']
        tab['theta'] = angular_separation(
            tab['true_az'], tab['true_alt'],
            tab['reco_az'], tab['reco_alt'],
        )

    gammas = particles['gamma']['events']

    gh_cut = 0.0
    log.info(f'Using fixed G/H cut of {gh_cut} to calculate theta cuts')

    # event display uses much finer bins for the theta cut than
    # for the sensitivity
    theta_bins = add_overflow_bins(create_bins_per_decade(
        10**(-1.9) * u.TeV,
        10**2.3005 * u.TeV,
        100,
    ))

    # theta cut is 68 percent containmente of the gammas
    # for now with a fixed global, unoptimized score cut
    mask_theta_cuts = gammas['gh_score'] >= gh_cut
    theta_cuts = calculate_percentile_cut(
        gammas['theta'][mask_theta_cuts],
        gammas['reco_energy'][mask_theta_cuts],
        bins=theta_bins,
        min_value=0.05 * u.deg,
        fill_value=np.nan * u.deg,
        percentile=68,
    )

    # evaluate the theta cut
    for p in particles.values():
        tab = p['events']
        tab['selected_theta'] = evaluate_binned_cut(
            tab['theta'],
            tab['reco_energy'],
            theta_cuts,
            operator.le,
        )

    # background table composed of both electrons and protons
    background = table.vstack([
        particles['proton']['events'],
        particles['electron']['events']
    ])

    # same bins as event display uses
    sensitivity_bins = add_overflow_bins(create_bins_per_decade(
        10**-1.9 * u.TeV, 10**2.31 * u.TeV, bins_per_decade=5
    ))
    sensitivity, gh_cuts = optimize_gh_cut(
        gammas[gammas['selected_theta']],
        background[background['selected_theta']],
        bins=sensitivity_bins,
        cut_values=np.arange(-1.0, 1.005, 0.05),
        op=operator.ge,
    )

    sensitivity['flux_sensitivity'] = sensitivity['relative_sensitivity'] * CRAB_HEGRA(sensitivity['reco_energy_center'])

    # calculate IRFs for the best cuts

    # write OGADF output file
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(sensitivity, name='SENSITIVITY'),
        fits.BinTableHDU(theta_cuts, name='THETA_CUTS'),
        fits.BinTableHDU(gh_cuts, name='GH_CUTS'),
    ]
    fits.HDUList(hdus).writeto('sensitivity.fits.gz', overwrite=True)


if __name__ == '__main__':
    main()
