import numpy as np
import astropy.units as u
from ..binning import create_histogram_table


@u.quantity_input(area=u.m**2)
def effective_area(selected_n, simulated_n, area):
    '''
    Calculate effective area for histograms of selected and total simulated events

    Parameters
    ----------
    n_selected: int or ``~numpy.ndarray``[int]
        The number of surviving (e.g. triggered, analysed, after cuts)
    n_simulated: int or ``~numpy.ndarray``[int]
        The total number of events simulated
    area: ``~astropy.units.Quantity``[area]
        Area in which particle's core position was simulated
    '''
    return (selected_n / simulated_n) * area


def point_like_effective_area(selected_events, simulation_info, true_energy_bins):
    '''
    Calculate effective area for the given set of DL2 events, simulation statistics
    and true energy bins.

    Parameters
    ----------
    selected_events: ``~astropy.table.QTable``
        DL2 events table, required columns for this function: `true_energy`.
    simulation_info: ``~pyirf.simulations.SimulatedEventsInfo``
        The overall statistics of the simulated events
    true_energy_bins: ``astropy.units.Quantity``[energy]
        The bin edges in which to calculate effective area.
    '''
    area = np.pi * simulation_info.max_impact**2

    hist_selected = create_histogram_table(selected_events, true_energy_bins, 'true_energy')
    hist_simulated = simulation_info.calculate_n_showers(true_energy_bins)

    return effective_area(hist_selected['n'], hist_simulated, area)
