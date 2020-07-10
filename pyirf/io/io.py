from tables import open_file
import numpy as np
from ctapipe.io import HDF5TableReader
from ctapipe.io.containers import MCHeaderContainer
import yaml
import pkg_resources
import os
from astropy.table import Table


def load_config(name):
    """Load YAML configuration file."""
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
    Check that simu info are the same for all runs from merged file
    Combine relevant simu info such as num_showers (sum)
    Note: works for a single run file as well

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


def get_simu_info(filepath, particle_name, config={}):
    """
    read simu info from file and return config
    """

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

    print(particle_name)
    print(cfg)

    return config


def read_EventDisplay(indir=None, infile=None):
    """Read DL2 files in FITS format from the EventDisplay analysis chain.

    Parameters
    ----------
    indir : str
        Path of the DL2 file.
    infile : str
        Name of the DL2 file.

    Returns
    -------

    table : astropy.Table
        Astropy Table object containing the reconstructed events information.

    """

    if (not indir) or (not infile):
        print("WARNING: missing input information!")
        print("Please, check that the folder exists and it contains the file.")
        exit()

    table = Table.read(f"{infile}", hdu=1)

    return table
def write(cuts=None, irfs=None):
    """Write DL3 data.

    This should be writer for the DL3 data!
    Format is still unclear, but nomenclature should follow GADF.

    We need at least the applied optimized cuts and the IRFs.

    """
    return None


# def get_resource(resource_name):
#     """ get the filename for a resource """
#     resource_path = os.path.join('resources', resource_name)
#     if not pkg_resources.resource_exists(__name__, resource_path):
#         raise FileNotFoundError(f"Couldn't find resource: {resource_name}")
#     else:
#         return pkg_resources.resource_filename(__name__, resource_path)
