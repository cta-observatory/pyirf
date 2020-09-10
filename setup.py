from setuptools import setup, find_packages
import re

with open('pyirf/version.py') as f:
    __version__ = re.search("^__version__ = '(.*)'$", f.read()).group(1)

setup(
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'astropy',
        'ctaplot~=0.5.0',
        'gammapy==0.8',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'tables',
        'ctapipe==0.7',
    ],
)
