from astropy.table import QTable
import astropy.units as u

from ..simulations import SimulatedEventsInfo


COLUMN_MAP = {
    'obs_id': 'OBS_ID',
    'event_id': 'EVENT_ID',
    'true_energy': 'MC_ENERGY',
    'reco_energy': 'ENERGY',
    'true_alt': 'MC_ALT',
    'true_az': 'AZ',
    'reco_alt': 'ALT',
    'reco_az': 'AZ',
    'gh_score': 'GH_MVA',
    'multiplicity': 'MULTIP',
}


def read_eventdisplay_fits(infile):
    """
    Read an DL2 Fits file as produced by the DL2 converter from root
    here: https://github.com/Eventdisplay/Converters/blob/master/DL2/generate_DL2_file.py

    Returns
    -------
    events: astropy.QTable
        Astropy Table object containing the reconstructed events information.
    simulated_events: ``~pyirf.simulations.SimulatedEventsInfo``

    """
    events_table = QTable.read(infile, format='fits', hdu='EVENTS')
    sim_events = QTable.read(infile, format='fits', hdu='SIMULATED EVENTS')
    run_header = QTable.read(infile, format='fits', hdu='RUNHEADER')[0]

    events = QTable({
        new: events_table[old]
        for new, old in COLUMN_MAP.items()
    })

    sim_info = SimulatedEventsInfo(
        n_showers=sim_events['EVENTS'].sum(),
        energy_min=u.Quantity(run_header['E_range'][0], u.TeV),
        energy_max=u.Quantity(run_header['E_range'][1], u.TeV),
        max_impact=u.Quantity(run_header['core_range'][1], u.m),
        spectral_index=run_header['spectral_index'],
        viewcone=u.Quantity(run_header['viewcone'][1], u.deg),
    )

    return events, sim_info
