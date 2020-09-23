import numpy as np
import astropy.units as u
from astropy.table import QTable
from ..binning import create_histogram_table


@u.quantity_input(area=u.m**2)
def effective_area(selected_n, simulated_n, area):
    return (selected_n / simulated_n) * area


def calculate_simulated_events(simulation_info, bins):
    bins = bins.to_value(u.TeV)
    e_low = bins[:-1]
    e_high = bins[1:]

    int_index = simulation_info.spectral_index + 1
    e_min = simulation_info.energy_min.to_value(u.TeV)
    e_max = simulation_info.energy_max.to_value(u.TeV)

    e_term = e_low**int_index - e_high**int_index
    normalization = int_index / (e_max**int_index - e_min**int_index)

    return simulation_info.n_showers * normalization * e_term


def point_like_effective_area(selected_events, simulation_info, true_energy_bins):
    area = np.pi * simulation_info.max_impact**2

    hist_selected = create_histogram_table(selected_events, true_energy_bins, 'true_energy')
    hist_simulated = calculate_simulated_events(simulation_info, true_energy_bins)

    area_table = QTable(hist_selected[['true_energy_' + k for k in ('low', 'high')]])
    area_table['effective_area'] = effective_area(
        hist_selected['n'], hist_simulated, area
    )

    return area_table
