import numpy as np
import astropy.units as u
from gammapy.maps import MapAxis
from gammapy.irf import EffectiveAreaTable2D

from ..binning import create_histogram_table


__all__ = [
    "effective_area",
    "effective_area_per_energy",
    "effective_area_per_energy_and_fov",
]


@u.quantity_input(area=u.m ** 2)
def effective_area(n_selected, n_simulated, area):
    """
    Calculate effective area for histograms of selected and total simulated events

    Parameters
    ----------
    n_selected: int or numpy.ndarray[int]
        The number of surviving (e.g. triggered, analysed, after cuts)
    n_simulated: int or numpy.ndarray[int]
        The total number of events simulated
    area: astropy.units.Quantity[area]
        Area in which particle's core position was simulated
    """
    return (n_selected / n_simulated) * area


def effective_area_per_energy(
    selected_events,
    simulation_info,
    true_energy_axis: MapAxis,
    fov_offset_axis: MapAxis,
):
    """
    Calculate effective area in bins of true energy.

    Parameters
    ----------
    selected_events: astropy.table.QTable
        DL2 events table, required columns for this function: `true_energy`.
    simulation_info: pyirf.simulations.SimulatedEventsInfo
        The overall statistics of the simulated events
    true_energy_axis: gammapy.maps.MapAxis
        The bin edges in which to calculate effective area.
    """
    area = np.pi * simulation_info.max_impact ** 2

    if fov_offset_axis.nbin != 1:
        raise ValueError('fov_offset_axis must only have a single bin')

    hist_selected = create_histogram_table(
        selected_events,
        true_energy_axis.edges,
        "true_energy"
    )
    hist_simulated = simulation_info.calculate_n_showers_per_energy(true_energy_axis.edges)

    aeff = effective_area(hist_selected["n"], hist_simulated, area)
    return EffectiveAreaTable2D(
        axes=[
            true_energy_axis,
            fov_offset_axis,
        ],
        data=aeff[:, np.newaxis],
    )


def effective_area_per_energy_and_fov(
    selected_events,
    simulation_info,
    true_energy_axis: MapAxis,
    fov_offset_axis: MapAxis,
):
    """
    Calculate effective area in bins of true energy and field of view offset.

    Parameters
    ----------
    selected_events: astropy.table.QTable
        DL2 events table, required columns for this function:
        - `true_energy`
        - `true_source_fov_offset`
    simulation_info: pyirf.simulations.SimulatedEventsInfo
        The overall statistics of the simulated events
    true_energy_bins: astropy.units.Quantity[energy]
        The true energy bin edges in which to calculate effective area.
    fov_offset_bins: astropy.units.Quantity[angle]
        The field of view radial bin edges in which to calculate effective area.
    """
    area = np.pi * simulation_info.max_impact ** 2

    hist_simulated = simulation_info.calculate_n_showers_per_energy_and_fov(
        true_energy_axis.edges,
        fov_offset_axis.edges,
    )

    hist_selected, _, _ = np.histogram2d(
        selected_events["true_energy"].to_value(u.TeV),
        selected_events["true_source_fov_offset"].to_value(u.deg),
        bins=[
            true_energy_axis.edges.to_value(u.TeV),
            fov_offset_axis.edges.to_value(u.deg),
        ],
    )

    aeff = effective_area(hist_selected, hist_simulated, area)
    return EffectiveAreaTable2D(
        axes=[
            true_energy_axis,
            fov_offset_axis,
        ],
        data=aeff,
    )
