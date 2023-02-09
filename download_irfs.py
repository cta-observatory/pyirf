# coding: utf-8
import tarfile
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests


def download_irfs():
    r = requests.get(
        "https://zenodo.org/record/5499840/files/cta-prod5-zenodo-fitsonly-v0.1.zip?download=1"
    )
    r.raise_for_status()

    obstime = 50 * 3600

    tarball = (
        "fits/CTA-Performance-prod5-v0.1-North-LSTSubArray-{zenith:d}deg.FITS.tar.gz"
    )
    irf_file = "Prod5-North-{zenith:d}deg-AverageAz-4LSTs.{obstime}s-v0.1.fits.gz"

    output_dir = Path(__file__).parent / "irfs"
    output_dir.mkdir(exist_ok=True)

    for zenith in (20, 40, 60):
        with ZipFile(BytesIO(r.content)) as zipfile:
            with tarfile.open(
                fileobj=zipfile.open(tarball.format(zenith=zenith), mode="r")
            ) as tar:
                tar.extract(
                    irf_file.format(zenith=zenith, obstime=obstime), path=output_dir
                )


if __name__ == "__main__":
    download_irfs()
