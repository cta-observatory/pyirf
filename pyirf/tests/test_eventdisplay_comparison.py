import numpy as np
import subprocess
import logging
import os
import uproot
import sys
from astropy.io import fits
from ogadf_schema.irfs import AEFF_2D, EDISP_2D, PSF_TABLE, BKG_2D, RAD_MAX
from pathlib import Path
import pytest
import astropy.units as u
from astropy.table import QTable


IRF_FILE = "DESY.d20180113.V3.ID0NIM2LST4MST4SST4SCMST4.prod3b-paranal20degs05b-NN.S.3HB9-FD.180000s.root"


@pytest.mark.integration
def test_eventdisplay_example(caplog):
    """Runs the EventDisplay example script and check its output."""

    ROOT_DIR = Path(__file__).parent.parent.parent

    # run script and check that it doesn't crash
    script_path = os.path.join(ROOT_DIR, "examples/calculate_eventdisplay_irfs.py")
    subprocess.check_output([sys.executable, script_path])

    # check that the output file exists and it's not empty
    outpath = ROOT_DIR / "pyirf_eventdisplay.fits.gz"
    assert outpath.is_file() and outpath.stat().st_size > 0

    # open FITS file
    output_hdul = fits.open(outpath)

    # known errors
    caplog.set_level(logging.WARNING, logger="fits_schema")

    # check that each HDU respects the OGADF schema
    AEFF_2D.validate_hdu(output_hdul["EFFECTIVE_AREA"], onerror="log")
    EDISP_2D.validate_hdu(output_hdul["ENERGY_DISPERSION"], onerror="log")
    PSF_TABLE.validate_hdu(output_hdul["PSF"], onerror="log")
    BKG_2D.validate_hdu(output_hdul["BACKGROUND"], onerror="log")
    RAD_MAX.validate_hdu(output_hdul["RAD_MAX"], onerror="log")

    errors_to_ignore = {
        # error due to astropy bug, which should be fixed in 4.0.2.
        # TODO: remove when we require astropy >= 4.0.2
        "Dimensionality of rows is 0, should be 1",
    }

    assert all(rec.message in errors_to_ignore for rec in caplog.records)

    # compare to reference

    f = uproot.open(ROOT_DIR / "data" / IRF_FILE)
    sensitivity_ed = f['DiffSens'] * u.Unit('erg s-1 cm-2')
    table = QTable.read(outpath, hdu='SENSITIVITY')
    sensitivity_pyirf = table['reco_energy_center']**2 * table['flux_sensitivity']

    ratio = (sensitivity_pyirf[1:-1] / sensitivity_ed[1:-1]).to_value(u.one)

    # TODO shrink margin when we get closer to prevent a regression
    assert np.all((ratio < 2) & (ratio > 0.45))
