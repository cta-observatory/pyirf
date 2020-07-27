#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
import pkg_resources
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.coordinates.angle_utilities import angular_separation
from gammapy.spectrum import cosmic_ray_flux, CrabSpectrum
import ctaplot
from copy import deepcopy

from pyirf.io.io import load_config, get_simu_info
from pyirf.perf import (CutsOptimisation,
                        CutsDiagnostic,
                        CutsApplicator,
                        IrfMaker,
                        SensitivityMaker,
                        )


from gammapy.irf import EnergyDispersion2D




def read_and_update_dl2(filepath, tel_id=1, filters=['intensity > 0']):
    """
    read DL2 data from lstchain file and update it to be compliant with irf Maker
    """
    dl2_params_lstcam_key = 'dl2/event/telescope/parameters/LST_LSTCam'  # lstchain DL2 files
    data = pd.read_hdf(filepath, key=dl2_params_lstcam_key)
    data = deepcopy(data.query(f'tel_id == {tel_id}'))
    for filter in filters:
        data = deepcopy(data.query(filter))

    # angles are in degrees in protopipe
    data['xi'] = pd.Series(angular_separation(data.reco_az.values * u.rad,
                                              data.reco_alt.values * u.rad,
                                              data.mc_az.values * u.rad,
                                              data.mc_alt.values * u.rad,
                                              ).to(u.deg).value,
                           index=data.index)

    data['offset'] = pd.Series(angular_separation(data.reco_az.values * u.rad,
                                                  data.reco_alt.values * u.rad,
                                                  data.mc_az_tel.values * u.rad,
                                                  data.mc_alt_tel.values * u.rad,
                                                  ).to(u.deg).value,
                               index=data.index)

    for key in ['mc_alt', 'mc_az', 'reco_alt', 'reco_az', 'mc_alt_tel', 'mc_az_tel']:
        data[key] = np.rad2deg(data[key])

    return data


def main(args):
    paths = {}
    paths['gamma'] = args.dl2_gamma_filename
    paths['proton'] = args.dl2_proton_filename
    paths['electron'] = args.dl2_electron_filename

    # Read configuration file
    cfg = load_config(args.config_file)
    # cfg = configuration()

    cfg['analysis']['obs_time'] = {}
    cfg['analysis']['obs_time']['unit'] = u.h
    cfg['analysis']['obs_time']['value'] = args.obs_time

    cfg['general']['outdir'] = args.outdir

    # Create output directory if necessary
    outdir = os.path.join(cfg['general']['outdir'], 'irf_ThSq_{}_Time{:.2f}{}'.format(
        cfg['analysis']['thsq_opt']['type'],
        cfg['analysis']['obs_time']['value'],
        cfg['analysis']['obs_time']['unit'])
                          )
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load data
    particles = ['gamma', 'electron', 'proton']
    evt_dict = dict()  # Contain DL2 file for each type of particle
    for particle in particles:
        infile = paths[particle]
        evt_dict[particle] = read_and_update_dl2(infile)
        cfg = get_simu_info(infile, particle, config=cfg)

    # Apply offset cut to proton and electron
    for particle in ['electron', 'proton']:
        evt_dict[particle] = evt_dict[particle].query('offset <= {}'.format(
            cfg['particle_information'][particle]['offset_cut'])
        )

    # Add required data in configuration file for future computation
    for particle in particles:
        # cfg['particle_information'][particle]['n_files'] = \
        #     len(np.unique(evt_dict[particle]['obs_id']))
        cfg['particle_information'][particle]['n_simulated'] = \
            cfg['particle_information'][particle]['n_files'] * cfg['particle_information'][particle][
                'n_events_per_file']

    # Define model for the particles
    model_dict = {'gamma': CrabSpectrum('hegra').model,
                  'proton': cosmic_ray_flux,
                  'electron': cosmic_ray_flux}

    # Reco energy binning
    cfg_binning = cfg['analysis']['ereco_binning']
    # ereco = np.logspace(np.log10(cfg_binning['emin']),
    #                     np.log10(cfg_binning['emax']),
    #                     cfg_binning['nbin'] + 1) * u.TeV
    ereco = ctaplot.ana.irf_cta().E_bin * u.TeV

    # Handle theta square cut optimisation
    # (compute 68 % containment radius PSF if necessary)
    thsq_opt_type = cfg['analysis']['thsq_opt']['type']
    print(thsq_opt_type)
    # if thsq_opt_type in 'fixed':
    #     thsq_values = np.array([cfg['analysis']['thsq_opt']['value']]) * u.deg
    #     print('Using fixed theta cut: {}'.format(thsq_values))
    # elif thsq_opt_type in 'opti':
    #     thsq_values = np.arange(0.05, 0.40, 0.01) * u.deg
    #     print('Optimising theta cut for: {}'.format(thsq_values))
    if thsq_opt_type != 'r68':
        raise ValueError("only r68 supported at the moment")
    elif thsq_opt_type in 'r68':
        print('Using R68% theta cut')
        print('Computing...')
        cfg_binning = cfg['analysis']['ereco_binning']
        ereco = np.logspace(np.log10(cfg_binning['emin']),
                            np.log10(cfg_binning['emax']),
                            cfg_binning['nbin'] + 1) * u.TeV
        ereco = ctaplot.ana.irf_cta().E_bin * u.TeV
        radius = 68

        thsq_values = list()
        for ibin in range(len(ereco) - 1):
            emin = ereco[ibin]
            emax = ereco[ibin + 1]

            energy_query = 'reco_energy > {} and reco_energy <= {}'.format(
                emin.value, emax.value
            )
            data = evt_dict['gamma'].query(energy_query).copy()

            min_stat = 0
            if len(data) <= min_stat:
                print('  ==> Not enough statistics:')
                print('To be handled...')
                thsq_values.append(0.3)
                continue

            psf = np.percentile(data['offset'], radius)
            psf_err = psf / np.sqrt(len(data))

            thsq_values.append(psf)
        thsq_values = np.array(thsq_values) * u.deg
        # Set 0.05 as a lower value
        idx = np.where(thsq_values.value < 0.05)
        thsq_values[idx] = 0.05 * u.deg
        print('Using theta cut: {}'.format(thsq_values))

    # Cuts optimisation
    print('### Finding best cuts...')
    cut_optimiser = CutsOptimisation(
        config=cfg,
        evt_dict=evt_dict,
        verbose_level=0
    )

    # Weight events
    print('- Weighting events...')
    cut_optimiser.weight_events(
        model_dict=model_dict,
        colname_mc_energy=cfg['column_definition']['mc_energy']
    )

    # Find best cutoff to reach best sensitivity
    print('- Estimating cutoffs...')
    cut_optimiser.find_best_cutoff(energy_values=ereco, angular_values=thsq_values)

    # Save results and auxiliary data for diagnostic
    print('- Saving results to disk...')
    cut_optimiser.write_results(
        outdir, '{}.fits'.format(cfg['general']['output_table_name']),
        format='fits'
    )

    # Cuts diagnostic
    print('### Building cut diagnostics...')
    cut_diagnostic = CutsDiagnostic(config=cfg, indir=outdir)
    cut_diagnostic.plot_optimisation_summary()
    cut_diagnostic.plot_diagnostics()

    # Apply cuts and save data
    print('### Applying cuts to data...')
    cut_applicator = CutsApplicator(config=cfg, evt_dict=evt_dict, outdir=outdir)
    cut_applicator.apply_cuts()

    # Irf Maker
    print('### Building IRF...')
    irf_maker = IrfMaker(config=cfg, evt_dict=evt_dict, outdir=outdir)
    irf_maker.build_irf()

    # Sensitivity maker
    print('### Estimating sensitivity...')
    sensitivity_maker = SensitivityMaker(config=cfg, outdir=outdir)
    sensitivity_maker.load_irf()
    sensitivity_maker.estimate_sensitivity()


def plot_sensitivity(irf_filename, ax=None, **kwargs):
    """
    Plot the sensitivity

    Parameters
    ----------
    irf_filename: path
    ax:
    kwargs:

    Returns
    -------
    ax
    """
    ax = ctaplot.plot_sensitivity_cta_performance('north', color='black', ax=ax)

    with fits.open(irf_filename) as irf:
        t = irf['SENSITIVITY']
        elo = t.data['ENERG_LO']
        ehi = t.data['ENERG_HI']
        energy = (elo + ehi) / 2.
        sens = t.data['SENSITIVITY']

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.errorbar(energy, sens,
                xerr=(ehi - elo) / 2.,
                **kwargs
                )

    ax.legend(fontsize=17)
    ax.grid(which='both')
    return ax


def plot_angular_resolution(irf_filename, ax=None, **kwargs):
    """
    Plot angular resolution from an IRF file

    Parameters
    ----------
    irf_filename
    ax
    kwargs

    Returns
    -------

    """

    ax = ctaplot.plot_angular_resolution_cta_performance('north', color='black', ax=ax)

    with fits.open(irf_filename) as irf:
        psf_hdu = irf['POINT SPREAD FUNCTION']
        e_lo = psf_hdu.data['ENERG_LO']
        e_hi = psf_hdu.data['ENERG_HI']
        energy = (e_lo + e_hi) / 2.
        psf = psf_hdu.data['PSF68']

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.errorbar(energy, psf,
                xerr=(e_hi - e_lo) / 2.,
                **kwargs,
                )

    ax.legend(fontsize=17)
    ax.grid(which='both')
    return ax


def plot_energy_resolution_hdf(gamma_filename, ax=None, **kwargs):
    """
    Plot angular resolution from an IRF file

    Parameters
    ----------
    irf_filename
    ax
    kwargs

    Returns
    -------

    """
    data = pd.read_hdf(gamma_filename)

    ax = ctaplot.plot_angular_resolution_cta_performance('north', color='black', label='CTA North', ax=ax)
    ax = ctaplot.plot_energy_resolution(data.mc_energy, data.reco_energy, ax=ax, **kwargs)
    ax.grid(which='both')
    ax.set_title('Energy resoluton', fontsize=18)
    ax.legend()
    return ax


def plot_energy_resolution(irf_file, ax=None, **kwargs):
    """
    Plot angular resolution from an IRF file
    Parameters
    ----------
    irf_filename
    ax
    kwargs
    Returns
    -------
    """

    e2d = EnergyDispersion2D.read(irf_file, hdu='ENERGY DISPERSION')
    edisp = e2d.to_energy_dispersion('0 deg')

    energy_bin = np.logspace(-1.5, 1, 15)
    e = np.sqrt(energy_bin[1:] * energy_bin[:-1])
    xerr = (e - energy_bin[:-1], energy_bin[1:] - e)
    r = edisp.get_resolution(e)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.errorbar(e, r, xerr=xerr, **kwargs)
    ax.set_xscale('log')
    ax.grid(True, which='both')
    ax.set_title('Energy resoluton')
    ax.set_xlabel('Energy [TeV]')
    ax.legend()
    return ax


def plot_background_rate(irf_filename, ax=None, **kwargs):
    """

    Returns
    -------

    """
    from ctaplot.io.dataset import load_any_resource

    ax = plt.gca() if ax is None else ax

    bkg = load_any_resource('CTA-Performance-prod3b-v2-North-20deg-50h-BackgroundSqdeg.txt')
    ax.loglog((bkg[0] + bkg[1]) / 2., bkg[2], label='CTA performances North', color='black')

    with fits.open(irf_filename) as irf:
        elo = irf['BACKGROUND'].data['ENERG_LO']
        ehi = irf['BACKGROUND'].data['ENERG_HI']
        energy = (elo + ehi) / 2.
        bkg = irf['BACKGROUND'].data['BGD']

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.errorbar(energy, bkg,
                xerr=(ehi - elo) / 2.,
                **kwargs
                )

    ax.legend(fontsize=17)
    ax.grid(which='both')
    return ax


def plot_effective_area(irf_filename, ax=None, **kwargs):
    """

    Parameters
    ----------
    irf_filename
    ax
    kwargs

    Returns
    -------

    """

    ax = ctaplot.plot_effective_area_cta_performance('north', color='black', ax=ax)

    with fits.open(irf_filename) as irf:
        elo = irf['SPECRESP'].data['ENERG_LO']
        ehi = irf['SPECRESP'].data['ENERG_HI']
        energy = (elo + ehi) / 2.
        eff_area = irf['SPECRESP'].data['SPECRESP']
        eff_area_no_cut = irf['SPECRESP (NO CUTS)'].data['SPECRESP (NO CUTS)']

    if 'label' not in kwargs:
        kwargs['label'] = 'Effective area [m2]'
    else:
        user_label = kwargs['label']
        kwargs['label'] = f'{user_label}'

    ax.loglog(energy, eff_area, **kwargs)

    kwargs['label'] = f"{kwargs['label']} (no cuts)"
    kwargs['linestyle'] = '--'

    ax.loglog(energy, eff_area_no_cut, **kwargs)

    ax.legend(fontsize=17)
    ax.grid(which='both')
    return ax


if __name__ == '__main__':

    # performance_default_config = pkg_resources.resource_filename('pyirf', 'resources/performance.yml')
    performance_default_config = os.path.join(os.path.dirname(__file__), "../resources/performance.yml")

    parser = argparse.ArgumentParser(description='Make performance files')

    parser.add_argument(
        '--obs_time',
        dest='obs_time',
        type=float,
        default=50,
        help='Observation time in hours'
    )

    parser.add_argument('--dl2_gamma', '-g',
                        dest='dl2_gamma_filename',
                        type=str,
                        required=True,
                        help='path to the gamma dl2 file'
                        )

    parser.add_argument('--dl2_proton', '-p',
                        dest='dl2_proton_filename',
                        type=str,
                        required=True,
                        help='path to the proton dl2 file'
                        )

    parser.add_argument('--dl2_electron', '-e',
                        dest='dl2_electron_filename',
                        type=str,
                        required=True,
                        help='path to the electron dl2 file'
                        )

    parser.add_argument('--outdir', '-o',
                        dest='outdir',
                        type=str,
                        default='.',
                        help="Output directory"
                        )

    parser.add_argument('--conf', '-c',
                        dest='config_file',
                        type=str,
                        default=performance_default_config,
                        help="Optional. Path to a config file."
                             " If none is given, the standard performance config is used"
                        )

    args = parser.parse_args()

    main(args)

    irf_filename = os.path.join(args.outdir, 'irf_ThSq_r68_Time50.00h/irf.fits.gz')
    fig_output = os.path.join(args.outdir, 'irf_ThSq_r68_Time50.00h/')

    fig, ax = plt.subplots(figsize=(12, 7))
    ax = plot_angular_resolution(irf_filename, ax=ax, label='LST1 (lstchain)')
    fig.savefig(os.path.join(fig_output, 'angular_resolution.png'), dpi=200, fmt='png')

    fig, ax = plt.subplots(figsize=(12, 7))
    ax = plot_background_rate(irf_filename, ax=ax, label='LST1 (lstchain)')
    fig.savefig(os.path.join(fig_output, 'background_rate.png'), dpi=200, fmt='png')

    fig, ax = plt.subplots(figsize=(12, 7))
    ax = plot_effective_area(irf_filename, ax=ax, label='LST1 (lstchain)')
    fig.savefig(os.path.join(fig_output, 'effective_area.png'), dpi=200, fmt='png')

    fig, ax = plt.subplots(figsize=(12, 7))
    ax = plot_sensitivity(irf_filename, ax=ax, label='LST1 (lstchain)')
    fig.savefig(os.path.join(fig_output, 'sensitivity.png'), dpi=200, fmt='png')

    gamma_filename = os.path.join(args.outdir, 'irf_ThSq_r68_Time50.00h/gamma_processed.h5')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax = plot_energy_resolution(irf_filename, ax=ax, label='LST1 (lstchain)')
    fig.savefig(os.path.join(fig_output, 'energy_resolution.png'), dpi=200, fmt='png')
