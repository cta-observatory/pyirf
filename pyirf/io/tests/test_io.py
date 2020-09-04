"""Unit tests for input / output operations."""

from pkg_resources import resource_filename

from pyirf.io import load_config

# TO DO: test DL2 data in HDF5 and FITS format


def test_load_config():

    config_file = resource_filename("pyirf", "resources/config.yml")

    assert load_config(config_file) is not None
