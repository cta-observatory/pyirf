import glob
import sys
import subprocess
import pytest
import os
from pathlib import Path

# from ogadf_schema import AEFF_2D, EDISP_2D, PSF_TABLE, BKG_2D, RAD_MAX
from astropy.io import fits


def test_integration_test():

    ROOT_DIR = os.path.dirname(os.path.abspath("setup.py"))

    for example_script in glob.glob(f"{ROOT_DIR}/examples/calculate_*.py"):

        # run script and check that it doesn't crash
        subprocess.check_output(f"python {example_script}", shell=True)

        print("DEBUG")

        # check that the output file exists and it's not empty
        output_file_name = f"{ROOT_DIR}/pyirf_eventdisplay.fits.gz"
        output_file_fits = fits.open(output_file_name)

        print(*[repr(hdu.header["EXTNAME"]) for hdu in output_file_fits[1:]])

        assert os.path.isfile(output_file_name) and os.path.getsize(output_file_name)

        # check that the output file respects the OGADF schema
        # assert AEFF_2D.validate_hdu(f['EFFECTIVE AREA'], onerror='raise')
        # assert EDISP_2D.validate_hdu(f['ENERGY DISPERSION'], onerror='raise')
        # assert PSF_TABLE.validate_hdu(f['EFFECTIVE AREA'], onerror='raise')
        # assert BKG_2D.validate_hdu(f['EFFECTIVE AREA'], onerror='raise')
        # assert RAD_MAX.validate_hdu(f['EFFECTIVE AREA'], onerror='raise')
