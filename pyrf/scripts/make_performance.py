#!/usr/bin/env python

import os
import astropy.units as u
import argparse
import pandas as pd
import numpy as np

from gammapy.spectrum import cosmic_ray_flux, CrabSpectrum

from pyrf.io.io import load_config
from pyrf.perf import (CutsOptimisation,
                       CutsDiagnostic,
                       CutsApplicator,
                       IrfMaker,
                       SensitivityMaker,
                       )


def main():
    # Read arguments
    parser = argparse.ArgumentParser(description='Make performance files')
    parser.add_argument('--config_file', type=str, required=True, help='')
    parser.add_argument(
        '--obs_time',
        type=str,
        required=True,
        help='Observation time, should be given as a string, value and astropy unit separated by an empty space'
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--wave', dest="mode", action='store_const',
                            const="wave", default="tail",
                            help="if set, use wavelet cleaning")
    mode_group.add_argument('--tail', dest="mode", action='store_const',
                            const="tail",
                            help="if set, use tail cleaning, otherwise wavelets")
    args = parser.parse_args()

    # Read configuration file
    cfg = load_config(args.config_file)

    # Add obs. time in configuration file
    str_obs_time = args.obs_time.split()
    cfg['analysis']['obs_time'] = {'value': float(str_obs_time[0]), 'unit': str(str_obs_time[-1])}

    # Create output directory if necessary
    outdir = os.path.join(cfg['general']['outdir'], 'irf_{}_ThSq_{}_Time{:.2f}{}'.format(
        args.mode,
        cfg['analysis']['thsq_opt']['type'],
        cfg['analysis']['obs_time']['value'],
        cfg['analysis']['obs_time']['unit'])
    )
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    indir = cfg['general']['indir']
    template_input_file = cfg['general']['template_input_file']

    # Load data
    particles = ['gamma', 'electron', 'proton']
    evt_dict = dict()  # Contain DL2 file for each type of particle
    for particle in particles:
        # template looks like dl2_{}_{}_merged.h5
        infile = os.path.join(indir, template_input_file.format(args.mode, particle))
        evt_dict[particle] = pd.read_hdf(infile, key='reco_events')

    # Apply offset cut to proton and electron
    for particle in ['electron', 'proton']:
        # print('Initial stat: {} {}'.format(len(evt_dict[particle]), particle))
        evt_dict[particle] = evt_dict[particle].query('offset <= {}'.format(
            cfg['particle_information'][particle]['offset_cut'])
        )

    # Add required data in configuration file for future computation
    for particle in particles:
        cfg['particle_information'][particle]['n_files'] = \
            len(np.unique(evt_dict[particle]['obs_id']))
        cfg['particle_information'][particle]['n_simulated'] = \
            cfg['particle_information'][particle]['n_files'] * cfg['particle_information'][particle]['n_events_per_file']

    # Define model for the particles
    model_dict = {'gamma': CrabSpectrum('hegra').model,
                  'proton': cosmic_ray_flux,
                  'electron': cosmic_ray_flux}

    # Reco energy binning
    cfg_binning = cfg['analysis']['ereco_binning']
    ereco = np.logspace(np.log10(cfg_binning['emin']),
                        np.log10(cfg_binning['emax']),
                        cfg_binning['nbin'] + 1) * u.TeV

    # Handle theta square cut optimisation
    # (compute 68 % containment radius PSF if necessary)
    thsq_opt_type = cfg['analysis']['thsq_opt']['type']
    if thsq_opt_type in 'fixed':
        thsq_values = np.array([cfg['analysis']['thsq_opt']['value']]) * u.deg
        print('Using fixed theta cut: {}'.format(thsq_values))
    elif thsq_opt_type in 'opti':
        thsq_values = np.arange(0.05, 0.40, 0.01) * u.deg
        print('Optimising theta cut for: {}'.format(thsq_values))
    elif thsq_opt_type in 'r68':
        print('Using R68% theta cut')
        print('Computing...')
        cfg_binning = cfg['analysis']['ereco_binning']
        ereco = np.logspace(np.log10(cfg_binning['emin']),
                            np.log10(cfg_binning['emax']),
                            cfg_binning['nbin'] + 1) * u.TeV
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
                # import sys
                # sys.exit()

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


if __name__ == '__main__':
    main()