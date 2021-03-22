from setuptools import setup, find_packages
import re

with open("pyirf/version.py") as f:
    __version__ = re.search('^__version__ = "(.*)"$', f.read()).group(1)


gammapy = "gammapy~=0.18"

extras_require = {
    "docs": [
        "sphinx",
        "sphinx_rtd_theme",
        "sphinx_automodapi",
        "numpydoc",
        "nbsphinx",
        "uproot4",
        "awkward1",
        "notebook",
        "tables",
        gammapy,
    ],
    "tests": [
        "pytest",
        "pytest-cov",
        "gammapy~=0.18",
        "ogadf-schema~=0.2.3",
        "uproot~=4.0",
        "awkward~=1.0",
    ],
    "gammapy": [
        gammapy,
    ]
}

extras_require["all"] = list(set(extras_require["tests"] + extras_require["docs"]))

setup(
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "astropy~=4.0,>=4.0.2",
        "matplotlib",
        "numpy>=1.18",
        "scipy",
        "tqdm",
    ],
    extras_require=extras_require,
)
