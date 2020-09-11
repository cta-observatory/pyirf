"""Script to produce DL3 data from DL2 data and a configuration file.

Is it initially thought as a clean start based on old code for reproducing
EventDisplay DL3 data based on the latest release of the GADF format.

Todo:
- make some config arguments also CLI ones like in ctapipe-stage1-process


"""

# =========================================================================
#                            MODULE IMPORTS
# =========================================================================

# PYTHON STANDARD LIBRARY

import argparse
import os

# THIRD-PARTY MODULES

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates.angle_utilities import angular_separation
from gammapy.spectrum import cosmic_ray_flux, CrabSpectrum  # UPDATE TO LATEST

# THIS PACKAGE

from pyirf.io.io import load_config, read_FITS
from pyirf.perf import (
    CutsOptimisation,
    CutsDiagnostic,
    CutsApplicator,
    IrfMaker,
    SensitivityMaker,
)


def main():

    # =========================================================================
    #                   READ INPUT FROM CLI AND CONFIGURATION FILE
    # =========================================================================

    # INPUT FROM CLI

    parser = argparse.ArgumentParser(description="Produce DL3 data from DL2.")

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="A configuration file like pyirf/resources/performance.yaml .",
    )

    parser.add_argument(
        "--obs_time",
        type=str,
        required=True,
        help="An observation time given as a string in astropy format e.g. '50h' or '30min'",
    )

    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        help="Name of the pipeline that has produced the DL2 files.",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Print debugging information."
    )

    args = parser.parse_args()

    # INPUT FROM THE CONFIGURATION FILE

    cfg = load_config(args.config_file)

    # Add obs. time to the configuration file
    obs_time = u.Quantity(args.obs_time)
    cfg["analysis"]["obs_time"] = {
        "value": obs_time.value,
        "unit": obs_time.unit.to_string("fits"),
    }

    # Get input directory
    indir = cfg["general"]["indir"]

    # Get template of the input file(s)
    template_input_file = cfg["general"]["template_input_file"]

    # Get output directory
    outdir = os.path.join(
        cfg["general"]["outdir"],
        "irf_{}_Time{}{}".format(
            args.pipeline,
            cfg["analysis"]["obs_time"]["value"],
            cfg["analysis"]["obs_time"]["unit"],
        ),
    )  # and create it if necessary
    os.makedirs(outdir, exist_ok=True)

    # =========================================================================
    #                   READ DL2 DATA AND STORE IT ACCORDING TO GADF
    # =========================================================================

    # Load FITS data
    particles = ["gamma", "electron", "proton"]
    evt_dict = dict()  # Contain DL2 file for each type of particle
    for particle in particles:
        if args.debug:
            print(f"Loading {particle} DL2 data...")
        infile = os.path.join(indir, template_input_file.format(particle))
        evt_dict[particle] = read_FITS(
            config=cfg, infile=infile, pipeline=args.pipeline, debug=args.debug
        )

    # =========================================================================
    #               PRELIMINARY OPERATIONS FOR SPECIFIC PIPELINES
    # =========================================================================

    # Some pipelines could provide some of the DL2 data in different ways
    # After this part, DL2 data is supposed to be equivalent, regardless
    # of the original pipeline.

    # Later we should move this out of here, perhaps under a "utils" module.

    if args.pipeline == "EventDisplay":

        # EventDisplay provides true and reconstructed directions, so we
        # calculate THETA here and we add it to the tables.

        for particle in particles:

            THETA = angular_separation(
                evt_dict[particle]["TRUE_AZ"],
                evt_dict[particle]["TRUE_ALT"],
                evt_dict[particle]["AZ"],
                evt_dict[particle]["ALT"],
            )  # in degrees

            # Add THETA column
            evt_dict[particle]["THETA"] = THETA

    # =========================================================================
    #                   REST OF THE OPERATIONS (TO BE REFACTORED)
    # =========================================================================

    # Apply offset cut to proton and electron
    for particle in ["electron", "proton"]:

        # There seems to be a problem in using pandas from FITS data
        # ValueError: Big-endian buffer not supported on little-endian compiler
        # I convert to astropy table....
        # should we use only those?

        evt_dict[particle] = Table.from_pandas(evt_dict[particle])

        if args.debug:
            print(particle)
            # print(evt_dict[particle].head(n=5))
            print(evt_dict[particle])

        # print('Initial stat: {} {}'.format(len(evt_dict[particle]), particle))

        mask_theta = (
            evt_dict[particle]["THETA"]
            < cfg["particle_information"][particle]["offset_cut"]
        )
        evt_dict[particle] = evt_dict[particle][mask_theta]
        # PANDAS EQUIVALENT
        # evt_dict[particle] = evt_dict[particle].query(
        #     "THETA <= {}".format(cfg["particle_information"][particle]["offset_cut"])
        # )

    # Add required data in configuration file for future computation
    for particle in particles:
        n_files = cfg["particle_information"][particle]["n_files"]
        print(f"{n_files} files for {particle}")
        cfg["particle_information"][particle]["n_files"] = len(
            np.unique(evt_dict[particle]["OBS_ID"])
        )
        cfg["particle_information"][particle]["n_simulated"] = (
            cfg["particle_information"][particle]["n_files"]
            * cfg["particle_information"][particle]["n_events_per_file"]
        )

    # Define model for the particles
    model_dict = {
        "gamma": CrabSpectrum("hegra").model,
        "proton": cosmic_ray_flux,
        "electron": cosmic_ray_flux,
    }

    # Reco energy binning
    cfg_binning = cfg["analysis"]["ereco_binning"]
    ereco = (
        np.logspace(
            np.log10(cfg_binning["emin"]),
            np.log10(cfg_binning["emax"]),
            cfg_binning["nbin"] + 1,
        )
        * u.TeV
    )

    # Handle theta square cut optimisation
    # (compute 68 % containment radius PSF if necessary)
    thsq_opt_type = cfg["analysis"]["thsq_opt"]["type"]
    if thsq_opt_type == "fixed":
        thsq_values = np.array([cfg["analysis"]["thsq_opt"]["value"]]) * u.deg
        print("Using fixed theta cut: {}".format(thsq_values))
    elif thsq_opt_type == "opti":
        thsq_values = np.arange(0.05, 0.40, 0.01) * u.deg
        print("Optimising theta cut for: {}".format(thsq_values))
    elif thsq_opt_type == "r68":
        print("Using R68% theta cut")
        print("Computing...")
        cfg_binning = cfg["analysis"]["ereco_binning"]
        ereco = (
            np.logspace(
                np.log10(cfg_binning["emin"]),
                np.log10(cfg_binning["emax"]),
                cfg_binning["nbin"] + 1,
            )
            * u.TeV
        )
        radius = 68

        thsq_values = list()

        # There seems to be a problem in using pandas from FITS data
        # ValueError: Big-endian buffer not supported on little-endian compiler

        # I convert to astropy table....
        # should we use only those?

        evt_dict["gamma"] = Table.from_pandas(evt_dict["gamma"])
        if args.debug:
            print("GAMMAS")
            # print(evt_dict["gamma"].head(n=5))
            print(evt_dict["gamma"])

        for ibin in range(len(ereco) - 1):
            emin = ereco[ibin]
            emax = ereco[ibin + 1]

            # PANDAS EQUIVALENT
            # energy_query = "reco_energy > {} and reco_energy <= {}".format(
            #     emin.value, emax.value
            # )
            # data = evt_dict["gamma"].query(energy_query).copy()

            mask_energy = (evt_dict["gamma"]["ENERGY"] > emin.value) & (
                evt_dict["gamma"]["ENERGY"] < emax.value
            )
            data = evt_dict["gamma"][mask_energy]

            min_stat = 0
            if len(data) <= min_stat:
                print("  ==> Not enough statistics:")
                print("To be handled...")
                thsq_values.append(0.3)
                continue
                # import sys
                # sys.exit()

            psf = np.percentile(data["THETA"], radius)
            # psf_err = psf / np.sqrt(len(data)) # not used after?

            thsq_values.append(psf)
        thsq_values = np.array(thsq_values) * u.deg
        # Set 0.05 as a lower value
        idx = np.where(thsq_values.value < 0.05)
        thsq_values[idx] = 0.05 * u.deg
        print("Using theta cut: {}".format(thsq_values))

    # Cuts optimisation
    print("### Finding best cuts...")
    cut_optimiser = CutsOptimisation(config=cfg, evt_dict=evt_dict, verbose_level=0)

    # Weight events
    print("- Weighting events...")
    cut_optimiser.weight_events(
        model_dict=model_dict,
        # colname_mc_energy=cfg["column_definition"]["TRUE_ENERGY"],
        colname_mc_energy="TRUE_ENERGY",
    )

    # Find best cutoff to reach best sensitivity
    print("- Estimating cutoffs...")
    cut_optimiser.find_best_cutoff(energy_values=ereco, angular_values=thsq_values)

    # Save results and auxiliary data for diagnostic
    print("- Saving results to disk...")
    cut_optimiser.write_results(
        outdir, "{}.fits".format(cfg["general"]["output_table_name"]), format="fits"
    )

    # Cuts diagnostic
    print("### Building cut diagnostics...")
    cut_diagnostic = CutsDiagnostic(config=cfg, indir=outdir)
    cut_diagnostic.plot_optimisation_summary()
    cut_diagnostic.plot_diagnostics()

    # Apply cuts and save data
    print("### Applying cuts to data...")
    cut_applicator = CutsApplicator(config=cfg, evt_dict=evt_dict, outdir=outdir)
    cut_applicator.apply_cuts(args.debug)

    # Irf Maker
    print("### Building IRF...")
    irf_maker = IrfMaker(config=cfg, evt_dict=evt_dict, outdir=outdir)
    irf_maker.build_irf(thsq_values)

    # Sensitivity maker
    print("### Estimating sensitivity...")
    sensitivity_maker = SensitivityMaker(config=cfg, outdir=outdir)
    sensitivity_maker.load_irf()
    sensitivity_maker.estimate_sensitivity()


if __name__ == "__main__":
    main()
