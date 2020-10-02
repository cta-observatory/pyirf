import subprocess
import logging
import os
import sys
from astropy.io import fits
from ogadf_schema.irfs import AEFF_2D, EDISP_2D, PSF_TABLE, BKG_2D, RAD_MAX
from pathlib import Path
import pytest


@pytest.mark.integration
def test_eventdisplay_example(caplog):
    """Runs the EventDisplay example script and check its output."""

    ROOT_DIR = Path(__file__).parent.parent.parent

    # run script and check that it doesn't crash
    script_path = os.path.join(ROOT_DIR, "examples/calculate_eventdisplay_irfs.py")
    subprocess.check_output([sys.executable, script_path])

    # check that the output file exists and it's not empty
    outname = os.path.join(ROOT_DIR, "pyirf_eventdisplay.fits.gz")
    assert os.path.isfile(outname) and os.path.getsize(outname)

    # open FITS file
    outfile = fits.open(outname)

    # known errors
    caplog.set_level(logging.WARNING, logger="fits_schema")

    # check that each HDU respects the OGADF schema
    AEFF_2D.validate_hdu(outfile["EFFECTIVE_AREA"], onerror="log")
    EDISP_2D.validate_hdu(outfile["ENERGY_DISPERSION"], onerror="log")
    PSF_TABLE.validate_hdu(outfile["PSF"], onerror="log")
    BKG_2D.validate_hdu(outfile["BACKGROUND"], onerror="log")
    RAD_MAX.validate_hdu(outfile["RAD_MAX"], onerror="log")

    errors_to_ignore = {
        "Dimensionality of rows is 0, should be 1",
    }

    assert all(rec.message in errors_to_ignore for rec in caplog.records)
