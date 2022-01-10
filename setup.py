from setuptools import setup, find_packages
import os



extras_require = {
    "docs": [
        "sphinx",
        "sphinx_rtd_theme",
        "sphinx_automodapi",
        "numpydoc",
        "nbsphinx",
        "uproot",
        "awkward1",
        "notebook",
        "tables",
    ],
    "tests": [
        "pytest",
        "pytest-cov",
        "ogadf-schema~=0.2.3",
        "uproot~=4.0",
        "awkward~=1.0",
    ],
}

extras_require["all"] = list(set(extras_require["tests"] + extras_require["docs"]))

setup(
    use_scm_version={"write_to": os.path.join("pyirf", "_version.py")},
    packages=find_packages(),
    install_requires=[
        "astropy~=4.0,>=4.0.2",
        "matplotlib",
        "numpy>=1.18",
        "scipy",
        "tqdm",
        "setuptools_scm",
        "gammapy~=0.19",
    ],
    include_package_data=True,
    extras_require=extras_require,
)
