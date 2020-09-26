"""
Example for using pyirf to calculate IRFS and sensitivity from EventDisplay DL2 fits
files produced from the root output by this script:

https://github.com/Eventdisplay/Converters/blob/master/DL2/generate_DL2_file.py
"""
import logging
import operator

import numpy as np
from astropy import table
import astropy.units as u
from astropy.io import fits

from pyirf.io.eventdisplay import read_eventdisplay_fits
from pyirf.binning import (
    create_bins_per_decade,
    add_overflow_bins,
    create_histogram_table,
)
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.sensitivity import calculate_sensitivity
from pyirf.utils import calculate_theta, calculate_source_fov_offset
from pyirf.benchmarks import energy_bias_resolution, angular_resolution

from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)
from pyirf.cut_optimization import optimize_gh_cut

from pyirf.irf import (
    point_like_effective_area,
    energy_dispersion,
    psf_table,
)

from pyirf.io import (
    create_aeff2d_hdu,
    create_psf_table_hdu,
    create_energy_dispersion_hdu,
    create_rad_max_hdu,
)


log = logging.getLogger("pyirf")


T_OBS = 50 * u.hour

# scaling between on and off region.
# Make off region 5 times larger than on region for better
# background statistics
ALPHA = 0.05

# gh cut used for first calculation of the binned theta cuts
INITIAL_GH_CUT = 0.0

particles = {
    "gamma": {
        "file": "data/gamma_onSource.S.3HB9-FD_ID0.eff-0.fits.gz",
        "target_spectrum": CRAB_HEGRA,
    },
    "proton": {
        "file": "data/proton_onSource.S.3HB9-FD_ID0.eff-0.fits.gz",
        "target_spectrum": IRFDOC_PROTON_SPECTRUM,
    },
    "electron": {
        "file": "data/electron_onSource.S.3HB9-FD_ID0.eff-0.fits.gz",
        "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
    },
}


def get_bg_cuts(cuts, alpha):
    """Rescale the cut values to enlarge the background region"""
    cuts = cuts.copy()
    cuts["cut"] /= np.sqrt(alpha)
    return cuts


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pyirf").setLevel(logging.DEBUG)

    for k, p in particles.items():
        log.info(f"Simulated {k.title()} Events:")
        p["events"], p["simulation_info"] = read_eventdisplay_fits(p["file"])

        p["simulated_spectrum"] = PowerLaw.from_simulation(p["simulation_info"], T_OBS)
        p["events"]["weight"] = calculate_event_weights(
            p["events"]["true_energy"], p["target_spectrum"], p["simulated_spectrum"]
        )
        p["events"]["source_fov_offset"] = calculate_source_fov_offset(p["events"])
        # calculate theta / distance between reco and assuemd source positoin
        # we handle only ON observations here, so the assumed source pos
        # is the pointing position
        p["events"]["theta"] = calculate_theta(
            p["events"],
            assumed_source_az=p["events"]["pointing_az"],
            assumed_source_alt=p["events"]["pointing_alt"],
        )
        log.info(p["simulation_info"])
        log.info("")

    gammas = particles["gamma"]["events"]
    # background table composed of both electrons and protons
    background = table.vstack(
        [particles["proton"]["events"], particles["electron"]["events"]]
    )

    log.info(f"Using fixed G/H cut of {INITIAL_GH_CUT} to calculate theta cuts")

    # event display uses much finer bins for the theta cut than
    # for the sensitivity
    theta_bins = add_overflow_bins(
        create_bins_per_decade(10 ** (-1.9) * u.TeV, 10 ** 2.3005 * u.TeV, 50,)
    )

    # theta cut is 68 percent containmente of the gammas
    # for now with a fixed global, unoptimized score cut
    mask_theta_cuts = gammas["gh_score"] >= INITIAL_GH_CUT
    theta_cuts = calculate_percentile_cut(
        gammas["theta"][mask_theta_cuts],
        gammas["reco_energy"][mask_theta_cuts],
        bins=theta_bins,
        min_value=0.05 * u.deg,
        fill_value=np.nan * u.deg,
        percentile=68,
    )

    # evaluate the theta cut
    gammas["selected_theta"] = evaluate_binned_cut(
        gammas["theta"], gammas["reco_energy"], theta_cuts, operator.le
    )
    # we make the background region larger by a factor of ALPHA,
    # so the radius by sqrt(ALPHA) to get better statistics for the background
    theta_cuts_bg = get_bg_cuts(theta_cuts, ALPHA)
    background["selected_theta"] = evaluate_binned_cut(
        background["theta"], background["reco_energy"], theta_cuts_bg, operator.le
    )

    # same bins as event display uses
    sensitivity_bins = add_overflow_bins(
        create_bins_per_decade(
            10 ** -1.9 * u.TeV, 10 ** 2.31 * u.TeV, bins_per_decade=5
        )
    )

    log.info("Optimizing G/H separation cut for best sensitivity")
    sensitivity_step_2, gh_cuts = optimize_gh_cut(
        gammas[gammas["selected_theta"]],
        background[background["selected_theta"]],
        bins=sensitivity_bins,
        cut_values=np.arange(-1.0, 1.005, 0.05),
        op=operator.ge,
        alpha=ALPHA,
    )

    # now that we have the optimized gh cuts, we recalculate the theta
    # cut as 68 percent containment on the events surviving these cuts.
    for tab in (gammas, background):
        tab["selected_gh"] = evaluate_binned_cut(
            tab["gh_score"], tab["reco_energy"], gh_cuts, operator.ge
        )

    theta_cuts_opt = calculate_percentile_cut(
        gammas["theta"],
        gammas["reco_energy"],
        theta_bins,
        fill_value=np.nan * u.deg,
        percentile=68,
        min_value=0.05 * u.deg,
    )

    theta_cuts_opt_bg = get_bg_cuts(theta_cuts_opt, ALPHA)

    for tab, cuts in zip([gammas, background], [theta_cuts_opt, theta_cuts_opt_bg]):
        tab["selected_theta"] = evaluate_binned_cut(
            tab["theta"], tab["reco_energy"], cuts, operator.le
        )
        tab["selected"] = tab["selected_theta"] & tab["selected_gh"]

    signal_hist = create_histogram_table(
        gammas[gammas["selected"]], bins=sensitivity_bins
    )
    background_hist = create_histogram_table(
        background[background["selected"]], bins=sensitivity_bins
    )

    sensitivity = calculate_sensitivity(signal_hist, background_hist, alpha=ALPHA)

    # scale relative sensitivity by Crab flux to get the flux sensitivity
    for s in (sensitivity_step_2, sensitivity):
        s["flux_sensitivity"] = s["relative_sensitivity"] * CRAB_HEGRA(
            s["reco_energy_center"]
        )

    # write OGADF output file
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(sensitivity, name="SENSITIVITY"),
        fits.BinTableHDU(sensitivity_step_2, name="SENSITIVITY_STEP_2"),
        fits.BinTableHDU(theta_cuts, name="THETA_CUTS"),
        fits.BinTableHDU(theta_cuts_opt, name="THETA_CUTS_OPT"),
        fits.BinTableHDU(gh_cuts, name="GH_CUTS"),
    ]

    masks = {
        "": gammas["selected"],
        "_NO_CUTS": slice(None),
        "_ONLY_GH": gammas["selected_gh"],
        "_ONLY_THETA": gammas["selected_theta"],
    }

    # binnings for the irfs
    true_energy_bins = add_overflow_bins(
        create_bins_per_decade(10 ** -1.9 * u.TeV, 10 ** 2.31 * u.TeV, 10,)
    )
    fov_offset_bins = [0, 0.5] * u.deg
    source_offset_bins = np.arange(0, 1 + 1e-4, 1e-3) * u.deg
    energy_migration_bins = np.geomspace(0.2, 5, 200)

    for label, mask in masks.items():
        effective_area = point_like_effective_area(
            gammas[mask],
            particles["gamma"]["simulation_info"],
            true_energy_bins=true_energy_bins,
        )
        hdus.append(
            create_aeff2d_hdu(
                effective_area[..., np.newaxis],  # add one dimension for FOV offset
                true_energy_bins,
                fov_offset_bins,
                extname="EFFECTIVE_AREA" + label,
            )
        )
        edisp = energy_dispersion(
            gammas[mask],
            true_energy_bins=true_energy_bins,
            fov_offset_bins=fov_offset_bins,
            migration_bins=energy_migration_bins,
        )
        hdus.append(
            create_energy_dispersion_hdu(
                edisp,
                true_energy_bins=true_energy_bins,
                migration_bins=energy_migration_bins,
                fov_offset_bins=fov_offset_bins,
                extname="ENERGY_DISPERSION" + label,
            )
        )

    bias_resolution = energy_bias_resolution(
        gammas[gammas["selected"]], true_energy_bins,
    )
    ang_res = angular_resolution(gammas[gammas["selected_gh"]], true_energy_bins,)

    psf = psf_table(
        gammas[gammas["selected_gh"]],
        true_energy_bins,
        fov_offset_bins=fov_offset_bins,
        source_offset_bins=source_offset_bins,
    )

    hdus.append(
        create_psf_table_hdu(
            psf, true_energy_bins, source_offset_bins, fov_offset_bins,
        )
    )
    hdus.append(
        create_rad_max_hdu(
            theta_bins, fov_offset_bins, rad_max=theta_cuts_opt["cut"][:, np.newaxis]
        )
    )
    hdus.append(fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION"))
    hdus.append(fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION"))
    fits.HDUList(hdus).writeto("pyirf_eventdisplay.fits.gz", overwrite=True)


if __name__ == "__main__":
    main()
