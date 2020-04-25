from setuptools import setup, find_packages
from pyrf import __version__
import os

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name="pyrf",
    version=__version__,
    description="Python IACT IRF builder",
    url="https://github.com/cta-observatory/pyrf",
    author="Julien Lefaucheur, Michele Peresano, Thomas Vuillaume",
    license="MIT",
    packages=find_packages(),
    package_data={},
    include_package_data=True,
    install_requires=[],
)