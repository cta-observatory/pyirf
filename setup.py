from setuptools import setup, find_packages
import re

with open("pyirf/version.py") as f:
    __version__ = re.search('^__version__ = "(.*)"$', f.read()).group(1)

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
        "tables",
        "gammapy~=0.8.0",
    ],
    extras_require={
        'docs': [
            'rinohtype',
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx_automodapi',
            'numpydoc',
            'nbsphinx'
        ]
    }
)
