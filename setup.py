from setuptools import setup, find_packages

exec(open('pyirf/version.py').read())

setup(
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'astropy',
        'ctaplot~=0.5.0',
        'gammapy==0.17',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'tables',
        'ctapipe==0.8',
    ],
)
