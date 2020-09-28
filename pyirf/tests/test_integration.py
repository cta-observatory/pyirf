import glob
import subprocess
import logging
import os
from astropy.io import fits
from ogadf_schema.irfs import AEFF_2D, EDISP_2D, PSF_TABLE, BKG_2D, RAD_MAX


def test_integration_test(caplog):

    caplog.set_level(logging.INFO)

    ROOT_DIR = os.path.dirname(os.path.abspath("setup.py"))

    for example_script in glob.glob(f"{ROOT_DIR}/examples/calculate_*.py"):

        # run script and check that it doesn't crash
        subprocess.check_output(f"python {example_script}", shell=True)

        # check that the output file exists and it's not empty
        outname = f"{ROOT_DIR}/pyirf_eventdisplay.fits.gz"
        outfile = fits.open(outname)

        assert os.path.isfile(outname) and os.path.getsize(outname)

        # check that the output file respects the OGADF schema

        # known errors
        dimensionality_error = "Dimensionality of rows is 0, should be 1"

        AEFF_2D.validate_hdu(outfile["EFFECTIVE_AREA"], onerror="log")
        assert str([rec.message for rec in caplog.records]) == str(
            [dimensionality_error] * 2
        )
        EDISP_2D.validate_hdu(outfile["ENERGY_DISPERSION"], onerror="log")
        assert str([rec.message for rec in caplog.records]) == str(
            [dimensionality_error] * 4
        )
        PSF_TABLE.validate_hdu(outfile["PSF"], onerror="log")
        assert str([rec.message for rec in caplog.records]) == str(
            [dimensionality_error] * 6
        )
        BKG_2D.validate_hdu(outfile["BACKGROUND"], onerror="log")
        assert str([rec.message for rec in caplog.records]) == str(
            [dimensionality_error] * 6
        )
        RAD_MAX.validate_hdu(outfile["RAD_MAX"], onerror="log")
        assert str([rec.message for rec in caplog.records]) == str(
            [dimensionality_error] * 8
        )
