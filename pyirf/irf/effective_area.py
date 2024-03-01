import numpy as np
import astropy.units as u
from ..binning import create_histogram_table


__all__ = [
    "effective_area",
    "effective_area_per_energy",
    "effective_area_per_energy_and_fov",
    "effective_area_3d_polar",
    "effective_area_3d_lonlat",
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


def effective_area_per_energy(selected_events, simulation_info, true_energy_bins):
    """
    Calculate effective area in bins of true energy.

    Parameters
    ----------
    selected_events: astropy.table.QTable
        DL2 events table, required columns for this function: `true_energy`.
    simulation_info: pyirf.simulations.SimulatedEventsInfo
        The overall statistics of the simulated events
    true_energy_bins: astropy.units.Quantity[energy]
        The bin edges in which to calculate effective area.
    """
    area = np.pi * simulation_info.max_impact ** 2

    hist_selected = create_histogram_table(
        selected_events, true_energy_bins, "true_energy"
    )
    hist_simulated = simulation_info.calculate_n_showers_per_energy(true_energy_bins)

    return effective_area(hist_selected["n"], hist_simulated, area)


def effective_area_per_energy_and_fov(
    selected_events, simulation_info, true_energy_bins, fov_offset_bins
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
        true_energy_bins, fov_offset_bins
    )

    hist_selected, _, _ = np.histogram2d(
        selected_events["true_energy"].to_value(u.TeV),
        selected_events["true_source_fov_offset"].to_value(u.deg),
        bins=[
            true_energy_bins.to_value(u.TeV),
            fov_offset_bins.to_value(u.deg),
        ],
    )

    return effective_area(hist_selected, hist_simulated, area)


def effective_area_3d_polar(
    selected_events,
    simulation_info,
    energy_bins,
    fov_offset_bins,
    fov_position_angle_bins,
):
    """
    Calculate effective area in bins of true energy, field of view offset, and field of view position angle.

    Parameters
    ----------
    selected_events: astropy.table.QTable
        DL2 events table, required columns for this function:
        - `true_energy`
        - `true_source_fov_offset`
        - `true_source_fov_position_angle`
    simulation_info: pyirf.simulations.SimulatedEventsInfo
        The overall statistics of the simulated events
    true_energy_bins: astropy.units.Quantity[energy]
        The true energy bin edges in which to calculate effective area.
    fov_offset_bins: astropy.units.Quantity[angle]
        The field of view radial bin edges in which to calculate effective area.
    fov_position_angle_bins: astropy.units.Quantity[radian]
        The field of view azimuthal bin edges in which to calculate effective area.
    """
    area = np.pi * simulation_info.max_impact**2

    hist_simulated = simulation_info.calculate_n_showers_3d_polar(
        energy_bins, fov_offset_bins, fov_position_angle_bins
    )

    hist_selected, _ = np.histogramdd(
        np.column_stack(
            [
                selected_events["true_energy"].to_value(u.TeV),
                selected_events["true_source_fov_offset"].to_value(u.deg),
                selected_events["true_source_fov_position_angle"].to_value(u.rad),
            ]
        ),
        bins=(
            energy_bins.to_value(u.TeV),
            fov_offset_bins.to_value(u.deg),
            fov_position_angle_bins.to_value(u.rad),
        ),
    )

    return effective_area(hist_selected, hist_simulated, area)


def effective_area_3d_lonlat(
    selected_events,
    simulation_info,
    energy_bins,
    fov_longitude_bins,
    fov_latitude_bins,
    subpixels=20,
):
    """
    Calculate effective area in bins of true energy, field of view longitude, and field of view latitude.

    Parameters
    ----------
    selected_events: astropy.table.QTable
        DL2 events table, required columns for this function:
        - `true_energy`
        - `true_source_fov_lon`
        - `true_source_fov_lat`
    simulation_info: pyirf.simulations.SimulatedEventsInfo
        The overall statistics of the simulated events
    true_energy_bins: astropy.units.Quantity[energy]
        The true energy bin edges in which to calculate effective area.
    fov_longitude_bins: astropy.units.Quantity[angle]
        The field of view longitude bin edges in which to calculate effective area.
    fov_latitude_bins: astropy.units.Quantity[angle]
        The field of view latitude bin edges in which to calculate effective area.
    """
    area = np.pi * simulation_info.max_impact**2

    hist_simulated = simulation_info.calculate_n_showers_3d_lonlat(
        energy_bins, fov_longitude_bins, fov_latitude_bins, subpixels=subpixels
    )

    selected_columns = np.column_stack(
            [
                selected_events["true_energy"].to_value(u.TeV),
                selected_events["true_source_fov_lon"].to_value(u.deg),
                selected_events["true_source_fov_lat"].to_value(u.deg),
            ]
        )
    bins = (
            energy_bins.to_value(u.TeV),
            fov_longitude_bins.to_value(u.deg),
            fov_latitude_bins.to_value(u.deg),
    )

    hist_selected, _ = np.histogramdd(selected_columns, bins=bins)

    return effective_area(hist_selected, hist_simulated, area)
