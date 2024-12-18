from itertools import product
import numpy as np
from astropy.table import QTable
import astropy.units as u
from tqdm.auto import tqdm
import operator


from .cuts import evaluate_binned_cut_by_index, calculate_percentile_cut, evaluate_binned_cut
from .sensitivity import calculate_sensitivity, estimate_background
from .binning import create_histogram_table, bin_center, calculate_bin_indices


__all__ = [
    "optimize_gh_cut",
]


def optimize_cuts(
    signal,
    background,
    reco_energy_bins,
    multiplicity_cuts,
    gh_cut_efficiencies,
    theta_cut_efficiencies,
    fov_offset_min=0 * u.deg,
    fov_offset_max=1 * u.deg,
    alpha=1.0,
    progress=True,
    **kwargs
):
    """
    Optimize the gamma/hadronnes, theta and multiplicity cut in every energy bin of reconstructed energy
    for best sensitivity.

    Parameters
    ----------
    signal: astropy.table.QTable
        event list of simulated signal events.
        Required columns are `theta`, `reco_energy`, 'weight', `gh_score`
        No directional (theta) or gamma/hadron cut should already be applied.
    background: astropy.table.QTable
        event list of simulated background events.
        Required columns are `reco_source_fov_offset`, `reco_energy`,
        'weight', `gh_score`.
        No directional (theta) or gamma/hadron cut should already be applied.
    reco_energy_bins: astropy.units.Quantity[energy]
        Bins in reconstructed energy to use for sensitivity computation
    gh_cut_efficiencies: np.ndarray[float, ndim=1]
        The cut efficiencies to scan for best sensitivity.
    theta_cuts: astropy.table.QTable
        cut definition of the energy dependent theta cut,
        e.g. as created by ``calculate_percentile_cut``
    op: comparison function with signature f(a, b) -> bool
        The comparison function to use for the gamma hadron score.
        Returning true means an event passes the cut, so is not discarded.
        E.g. for gammaness-like score, use `operator.ge` (>=) and for a
        hadroness-like score use `operator.le` (<=).
    fov_offset_min: astropy.units.Quantity[angle]
        Minimum distance from the fov center for background events to be taken into account
    fov_offset_max: astropy.units.Quantity[angle]
        Maximum distance from the fov center for background events to be taken into account
    alpha: float
        Size ratio of off region / on region. Will be used to
        scale the background rate.
    progress: bool
        If True, show a progress bar during cut optimization
    **kwargs are passed to ``calculate_sensitivity``
    """
    gh_cut_efficiencies = np.asanyarray(gh_cut_efficiencies)
    gh_cut_percentiles = 100 * (1 - gh_cut_efficiencies)
    fill_value = signal['gh_score'].max()


    sensitivities = []
    cut_indicies = []
    n_theta_cuts = len(theta_cut_efficiencies)
    n_gh_cuts = len(gh_cut_efficiencies)
    n_cuts = len(multiplicity_cuts) * n_theta_cuts * n_gh_cuts

    signal_bin_index, signal_valid = calculate_bin_indices(
        signal['reco_energy'], reco_energy_bins,
    )
    background_bin_index, background_valid = calculate_bin_indices(
        background['reco_energy'], reco_energy_bins,
    )

    gh_cut_grid = []
    theta_cut_grid = []

    with tqdm(total=n_cuts, disable=not progress) as bar:
        for multiplicity_index, multiplicity_cut in enumerate(multiplicity_cuts):

            signal_mask_multiplicity = signal['multiplicity'] >= multiplicity_cut
            background_mask_multiplicity = background["multiplicity"] >= multiplicity_cut

            gh_cuts = calculate_percentile_cut(
                signal['gh_score'][signal_mask_multiplicity],
                signal['reco_energy'][signal_mask_multiplicity],
                bins=reco_energy_bins,
                fill_value=fill_value,
                percentile=gh_cut_percentiles,
            )
            gh_cut_grid.append(gh_cuts)

            theta_cuts = calculate_percentile_cut(
                signal['theta'][signal_mask_multiplicity],
                signal['reco_energy'][signal_mask_multiplicity],
                bins=reco_energy_bins,
                fill_value=0.3 * u.deg,
                max_value=0.3 * u.deg,
                min_value=0.02 * u.deg,
                percentile=100 * theta_cut_efficiencies,
            )
            theta_cut_grid.append(theta_cuts)


            for gh_index, theta_index in product(range(n_gh_cuts), range(n_theta_cuts)):

                # apply the current cuts
                theta_cut = theta_cuts.copy()
                theta_cut["cut"] = theta_cuts["cut"][:, theta_index]
                signal_mask_theta = evaluate_binned_cut_by_index(
                    signal["theta"], signal_bin_index, signal_valid, theta_cut, operator.le,
                )

                gh_cut = gh_cuts.copy()
                gh_cut["cut"] = gh_cuts["cut"][:, gh_index]
                signal_mask_gh = evaluate_binned_cut_by_index(
                    signal["gh_score"], signal_bin_index, signal_valid, gh_cut, operator.ge,
                )

                signal_selected = signal_mask_gh & signal_mask_theta & signal_mask_multiplicity

                background_mask_gh = evaluate_binned_cut_by_index(
                    background["gh_score"], background_bin_index, background_valid, gh_cut, operator.ge,
                )
                background_selected = background_mask_gh & background_mask_multiplicity

                # create the histograms
                signal_hist = create_histogram_table(
                    signal[signal_selected], reco_energy_bins, "reco_energy"
                )

                background_hist = estimate_background(
                    events=background[background_selected],
                    reco_energy_bins=reco_energy_bins,
                    theta_cuts=theta_cut,
                    alpha=alpha,
                    fov_offset_min=fov_offset_min,
                    fov_offset_max=fov_offset_max,
                )

                sensitivity = calculate_sensitivity(
                    signal_hist, background_hist, alpha=alpha,
                    **kwargs,
                )
                cut_indicies.append((multiplicity_index, theta_index, gh_index))
                sensitivities.append(sensitivity)
                bar.update(1)

    best_gh_cut = QTable()
    best_gh_cut["low"] = reco_energy_bins[0:-1]
    best_gh_cut["center"] = bin_center(reco_energy_bins)
    best_gh_cut["high"] = reco_energy_bins[1:]
    best_gh_cut["cut"] = np.nan
    best_gh_cut["efficiency"] = np.nan

    best_multiplicity_cut = QTable()
    best_multiplicity_cut["low"] = reco_energy_bins[0:-1]
    best_multiplicity_cut["center"] = bin_center(reco_energy_bins)
    best_multiplicity_cut["high"] = reco_energy_bins[1:]
    best_multiplicity_cut["cut"] = np.nan

    best_theta_cut = QTable()
    best_theta_cut["low"] = reco_energy_bins[0:-1]
    best_theta_cut["center"] = bin_center(reco_energy_bins)
    best_theta_cut["high"] = reco_energy_bins[1:]
    best_theta_cut["cut"] = np.nan
    best_theta_cut["cut"].unit = theta_cuts["cut"].unit
    best_theta_cut["efficiency"] = np.nan

    best_sensitivity = sensitivities[0].copy()
    for bin_id in range(len(reco_energy_bins) - 1):
        sensitivities_bin = [s["relative_sensitivity"][bin_id] for s in sensitivities]

        if not np.all(np.isnan(sensitivities_bin)):
            # nanargmin won't return the index of nan entries
            best = np.nanargmin(sensitivities_bin)
        else:
            # if all are invalid, just use the first one
            best = 0
        
        multiplicity_index, theta_index, gh_index = cut_indicies[best]

        best_sensitivity[bin_id] = sensitivities[best][bin_id]

        best_gh_cut["cut"][bin_id] = gh_cut_grid[multiplicity_index]["cut"][bin_id][gh_index]
        best_multiplicity_cut["cut"][bin_id] = multiplicity_cuts[multiplicity_index]
        best_theta_cut["cut"][bin_id] = theta_cut_grid[multiplicity_index]["cut"][bin_id][theta_index]

        best_gh_cut["efficiency"][bin_id] = gh_cut_efficiencies[gh_index]
        best_theta_cut["efficiency"][bin_id] = theta_cut_efficiencies[theta_index]

    return best_sensitivity, best_multiplicity_cut, best_theta_cut, best_gh_cut


def optimize_gh_cut(
    signal,
    background,
    reco_energy_bins,
    gh_cut_efficiencies,
    theta_cuts,
    op=operator.ge,
    fov_offset_min=0 * u.deg,
    fov_offset_max=1 * u.deg,
    alpha=1.0,
    progress=True,
    **kwargs
):
    """
    This procedure is EventDisplay-like, since it only applies a
    pre-computed theta cut and then optimizes only the gamma/hadron separation
    cut.

    Parameters
    ----------
    signal: astropy.table.QTable
        event list of simulated signal events.
        Required columns are `theta`, `reco_energy`, 'weight', `gh_score`
        No directional (theta) or gamma/hadron cut should already be applied.
    background: astropy.table.QTable
        event list of simulated background events.
        Required columns are `reco_source_fov_offset`, `reco_energy`,
        'weight', `gh_score`.
        No directional (theta) or gamma/hadron cut should already be applied.
    reco_energy_bins: astropy.units.Quantity[energy]
        Bins in reconstructed energy to use for sensitivity computation
    gh_cut_efficiencies: np.ndarray[float, ndim=1]
        The cut efficiencies to scan for best sensitivity.
    op: comparison function with signature f(a, b) -> bool
        The comparison function to use for the gamma hadron score.
        Returning true means an event passes the cut, so is not discarded.
        E.g. for gammaness-like score, use `operator.ge` (>=) and for a
        hadroness-like score use `operator.le` (<=).
    fov_offset_min: astropy.units.Quantity[angle]
        Minimum distance from the fov center for background events to be taken into account
    fov_offset_max: astropy.units.Quantity[angle]
        Maximum distance from the fov center for background events to be taken into account
    alpha: float
        Size ratio of off region / on region. Will be used to
        scale the background rate.
    progress: bool
        If True, show a progress bar during cut optimization
    **kwargs are passed to ``calculate_sensitivity``
    """

    # we apply each cut for all reco_energy_bins globally, calculate the
    # sensitivity and then lookup the best sensitivity for each
    # bin independently

    signal_selected_theta = evaluate_binned_cut(
        signal['theta'], signal['reco_energy'], theta_cuts,
        op=operator.le,
    )

    sensitivities = []
    # calculate necessary percentile needed for
    # ``calculate_percentile_cut`` with the correct efficiency.
    # Depends on the operator, since we need to invert the
    # efficiency if we compare using >=, since percentile is
    # defines as <=.
    gh_cut_efficiencies = np.asanyarray(gh_cut_efficiencies)
    if op(-1, 1):
        # if operator behaves like "<=", "<" etc:
        percentiles = 100 * gh_cut_efficiencies
        fill_value = signal['gh_score'].min()
    else:
        # operator behaves like ">=", ">"
        percentiles = 100 * (1 - gh_cut_efficiencies)
        fill_value = signal['gh_score'].max()

    gh_cuts = calculate_percentile_cut(
        signal['gh_score'], signal['reco_energy'],
        bins=reco_energy_bins,
        fill_value=fill_value,
        percentile=percentiles,
    )

    for col in tqdm(range(len(gh_cut_efficiencies)), disable=not progress):
        gh_cut = gh_cuts.copy()
        gh_cut["cut"] = gh_cuts["cut"][:, col]

        # apply the current cut
        signal_selected = evaluate_binned_cut(
            signal["gh_score"], signal["reco_energy"], gh_cut, op,
        ) & signal_selected_theta

        background_selected = evaluate_binned_cut(
            background["gh_score"], background["reco_energy"], gh_cut, op,
        )

        # create the histograms
        signal_hist = create_histogram_table(
            signal[signal_selected], reco_energy_bins, "reco_energy"
        )

        background_hist = estimate_background(
            events=background[background_selected],
            reco_energy_bins=reco_energy_bins,
            theta_cuts=theta_cuts,
            alpha=alpha,
            fov_offset_min=fov_offset_min,
            fov_offset_max=fov_offset_max,
        )

        sensitivity = calculate_sensitivity(
            signal_hist, background_hist, alpha=alpha,
            **kwargs,
        )
        sensitivities.append(sensitivity)

    best_cut_table = QTable()
    best_cut_table["low"] = reco_energy_bins[0:-1]
    best_cut_table["center"] = bin_center(reco_energy_bins)
    best_cut_table["high"] = reco_energy_bins[1:]
    best_cut_table["cut"] = np.nan
    best_cut_table["efficiency"] = np.nan

    best_sensitivity = sensitivities[0].copy()
    for bin_id in range(len(reco_energy_bins) - 1):
        sensitivities_bin = [s["relative_sensitivity"][bin_id] for s in sensitivities]

        if not np.all(np.isnan(sensitivities_bin)):
            # nanargmin won't return the index of nan entries
            best = np.nanargmin(sensitivities_bin)
        else:
            # if all are invalid, just use the first one
            best = 0

        best_sensitivity[bin_id] = sensitivities[best][bin_id]
        best_cut_table["cut"][bin_id] = gh_cuts["cut"][bin_id][best]
        best_cut_table["efficiency"][bin_id] = gh_cut_efficiencies[best]

    return best_sensitivity, best_cut_table
