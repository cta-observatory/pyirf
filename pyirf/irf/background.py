import astropy.units as u
import numpy as np

from ..utils import cone_solid_angle

#: Unit of the background rate IRF
BACKGROUND_UNIT = u.Unit('s-1 TeV-1 sr-1')


def background_2d(events, reco_energy_bins, fov_offset_bins, t_obs):
    """
    Calculate background rates in radially symmetric bins in the field of view.

    GADF documentation here:
    https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/bkg/index.html#bkg-2d

    Parameters
    ----------
    events: astropy.table.QTable
        DL2 events table of the selected background events.
        Needed columns for this function: `source_fov_offset`, `reco_energy`, `weight`
    reco_energy: astropy.units.Quantity[energy]
        The bins in reconstructed energy to be used for the IRF
    fov_offset_bins: astropy.units.Quantity[angle]
        The bins in the field of view offset to be used for the IRF
    t_obs: astropy.units.Quantity[time]
        Observation time. This must match with how the individual event
        weights are calculated.

    Returns
    -------
    bg_rate: astropy.units.Quantity
        The background rate as particles per energy, time and solid angle
        in the specified bins.

        Shape: (len(reco_energy_bins) - 1, len(fov_offset_bins) - 1)
    """

    hist, _, _ = np.histogram2d(
        events["reco_energy"].to_value(u.TeV),
        events["source_fov_offset"].to_value(u.deg),
        bins=[
            reco_energy_bins.to_value(u.TeV),
            fov_offset_bins.to_value(u.deg),
        ],
        weights=events['weight'],
    )

    # divide all energy bins by their width
    # hist has shape (n_energy, n_fov_offset) so we need to transpose and then back
    bin_width_energy = np.diff(reco_energy_bins)
    per_energy = (hist.T / bin_width_energy).T

    # divide by solid angle in each fov bin and the observation time
    bin_solid_angle = np.diff(cone_solid_angle(fov_offset_bins))
    bg_rate = per_energy / t_obs / bin_solid_angle

    return bg_rate.to(BACKGROUND_UNIT)
