from setuptools import setup, find_packages
import re

with open("pyirf/version.py") as f:
    __version__ = re.search('^__version__ = "(.*)"$', f.read()).group(1)

extras_require = {
    'docs': [
        'rinohtype',
        'sphinx',
        'sphinx_rtd_theme',
        'sphinx_automodapi',
        'numpydoc',
        'nbsphinx'
    ],
    'tests': [
        'pytest',
        'pytest-cov',
        'uproot',
    ],
}

extras_require['all'] = extras_require['tests'] + extras_require['docs']

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
        "gammapy~=0.17",
    ],
    extras_require=extras_require,
)
