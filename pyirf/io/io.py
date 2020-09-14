"""
Set of classes and functions of input and output.

Proposal of general structure:
- a reader for each data format (in the end only FITS, but also HDF5 for now)
- a mapper that reads user-defined DL2 column names into GADF format
- there should be only one output format (we follow GADF).

Currently some column names are defined in the configuration file under the
section 'column_definition'.

"""


# PYTHON STANDARD LIBRARY
# import os

# THIRD-PARTY MODULES

from astropy.io import fits
import yaml

# import pkg_resources
from tables import open_file
import numpy as np
import pandas as pd

from ctapipe.io import HDF5TableReader
from ctapipe.io.containers import MCHeaderContainer


def load_config(name):
    """
    Load YAML configuration file.

    Parameters
    ----------
    name : str
        Path of the configuration file.

    Returns
    -------
    cfg : dict
        Dictionary containing all the configuration information.

    """
    try:
        with open(name, "r") as stream:
            cfg = yaml.load(stream, Loader=yaml.FullLoader)
    except FileNotFoundError as e:
        print(e)
        raise
    return cfg


def read_simu_info_hdf5(filename):
    """
    Read simu info from an hdf5 file

    Returns
    -------
    `ctapipe.containers.MCHeaderContainer`
    """

    with HDF5TableReader(filename) as reader:
        mcheader = reader.read("/simulation/run_config", MCHeaderContainer())
        mc = next(mcheader)

    return mc


def read_simu_info_merged_hdf5(filename):
    """
    Read simu info from a merged hdf5 file.

    Check that simu info are the same for all runs from merged file.
    Combine relevant simu info such as num_showers (sum).
    Note: works for a single run file as well.

    Parameters
    ----------
    filename: path to an hdf5 merged file

    Returns
    -------
    `ctapipe.containers.MCHeaderContainer`

    """
    with open_file(filename) as file:
        simu_info = file.root["simulation/run_config"]
        colnames = simu_info.colnames
        not_to_check = [
            "num_showers",
            "shower_prog_start",
            "detector_prog_start",
            "obs_id",
        ]
        for k in colnames:
            if k not in not_to_check:
                assert np.all(simu_info[:][k] == simu_info[0][k])
        num_showers = simu_info[:]["num_showers"].sum()

    combined_mcheader = read_simu_info_hdf5(filename)
    combined_mcheader["num_showers"] = num_showers
    return combined_mcheader


def get_simu_info(filepath, particle_name, config=None):
    """
    read simu info from file and return config
    """
    if config is None:
        config = {}

    if "particle_information" not in config:
        config["particle_information"] = {}
    if particle_name not in config["particle_information"]:
        config["particle_information"][particle_name] = {}
    cfg = config["particle_information"][particle_name]

    simu = read_simu_info_merged_hdf5(filepath)
    cfg["n_events_per_file"] = simu.num_showers * simu.shower_reuse
    cfg["n_files"] = 1
    cfg["e_min"] = simu.energy_range_min
    cfg["e_max"] = simu.energy_range_max
    cfg["gen_radius"] = simu.max_scatter_range
    cfg["diff_cone"] = simu.max_viewcone_radius
    cfg["gen_gamma"] = -simu.spectral_index

    return config


def internal_dataformat_mapper(debug=False, config=None):
    """Defines the format to be used internally after input.

    All readers should call this function to map input data from different
    formats.

    Parameters
    ----------
    config : dict
        Dictionary obtained from pyirf.io.load_config.
    debug  : bool
        If True, print some debugging information.

    Returns
    -------

    columns : dict
        Dictionary that maps user-defined DL2 quantities to the internal equivalent.

    """

    columns = {}

    for key in config["column_definition"]:
        columns[key] = config["column_definition"][key]

    if debug:
        print("Mapping to internal data format....")
        print(columns)

    return columns


def read_FITS(config=None, infile=None, pipeline="EventDisplay", debug=False):
    """
    Store contents of a FITS file into one or more astropy tables.

    Parameters
    ----------
    config : str
        Path of the DL2 file.
    infile : str
        Path of the DL2 file.
    debug  : bool
        If True, print some debugging information.

    Returns
    -------

    table : astropy.Table
        Astropy Table object containing the reconstructed events information.

    Notes
    -----
    For the moment this reader is specific to EventDisplay.

    If DL2 files in FITS format are supposed to have all the same structure,
    then this reader is fine; if not, this reader will become
    read_EventDisplay_FITS and others will follow.

    In general, though, for the the final FITS reader or any other specific one:

    - if GADF mandatory columns names are missing, only a warning is raised,
    - it is possible to add custom columns.

    """
    DL2data = dict()

    colnames = internal_dataformat_mapper(debug, config=config)

    # later differentiate between EVENTS, GTI & POINTING

    with fits.open(infile) as hdul:

        print(f"Found {len(hdul)} Header Data Units in {hdul.filename()}.")

        EVENTS = hdul[1]

        # map the keys

        for INTERNAL_key, USER_key in colnames.items():

            print(f"Checking if {INTERNAL_key} equivalent is defined...")

            # check for mispellings in the config file
            if USER_key in EVENTS.columns.names:
                if pipeline == "EventDisplay":
                    # for Event Display EVENTS is HDU 1
                    DL2data[INTERNAL_key] = EVENTS.data[USER_key]
                else:
                    print("WARNING : we support only EventDisplay for now!")
            else:
                print(f"WARNING : {USER_key} missing from DL2 data!")

    # Convert to pandas dataframe
    DL2data = pd.DataFrame.from_dict(DL2data)

    return DL2data


def write(cuts=None, irfs=None):
    """
    DL3 data writer.

    This should be writer for the DL3 data.
    For the moment it is just a dummy function for reference.
    Final format is still unclear, we are trying to follow the latest
    version of GADF [1]_.

    Notes
    -----

    .. [1] https://gamma-astro-data-formats.readthedocs.io/en/latest/

    """
    return None


# def get_resource(resource_name):
#     """ get the filename for a resource """
#     resource_path = os.path.join('resources', resource_name)
#     if not pkg_resources.resource_exists(__name__, resource_path):
#         raise FileNotFoundError(f"Couldn't find resource: {resource_name}")
#     else:
#         return pkg_resources.resource_filename(__name__, resource_path)
