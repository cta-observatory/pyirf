import pathlib
import re

import pytest
from astropy.units import Quantity

from gammapy.irf import load_irf_dict_from_file

PROD5_IRF_PATH = pathlib.Path(__file__).parent.parent / "irfs/"


@pytest.fixture(scope="session")
def prod5_irfs():
    if not PROD5_IRF_PATH.exists():
        pytest.fail(
            "Test IRF files missing, you need to download them using "
            "`python download_irfs.py` in pyirfs root directory."
        )

    # Get dict of {ZEN_PNT, IRF} pairs for each file in ./irfs
    irfs = {
        Quantity(re.search(r"\d{2}deg", str(irf_file)).group()): load_irf_dict_from_file(
            irf_file
        )
        for irf_file in PROD5_IRF_PATH.glob("*.fits.gz")
    }

    assert len(irfs) == 3
    for key in ["aeff", "psf", "edisp"]:
        for irf in irfs.values():
            assert key in irf

    # Sort dict by zenith angle
    return dict(sorted(irfs.items()))
