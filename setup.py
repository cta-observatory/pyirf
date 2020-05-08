from setuptools import setup, find_packages

exec(open('pyrf/version.py').read())

setup(
    version=__version__,
    packages=find_packages(),
)
