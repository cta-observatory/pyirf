from setuptools import setup, find_packages
import re

with open("pyirf/version.py") as f:
    __version__ = re.search('^__version__ = "(.*)"$', f.read()).group(1)

extras_require = {
    "docs": [
        "sphinx",
        "sphinx_rtd_theme",
        "sphinx_automodapi",
        "numpydoc",
        "nbsphinx",
        "uproot~=3.0",
    ],
    "tests": [
        "pytest",
        "pytest-cov",
        "gammapy~=0.17",
        "ogadf-schema~=0.2.3",
        "uproot~=3.0",
    ],
}

extras_require["all"] = list(set(extras_require["tests"] + extras_require["docs"]))

setup(
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "astropy~=4.0",
        "matplotlib",
        "numpy>=1.18",
        "pandas",
        "scipy",
        "tqdm",
        "tables",
    ],
    extras_require=extras_require,
)
