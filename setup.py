from setuptools import setup, find_packages
import os


gammapy = "gammapy~=0.19"

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
        gammapy,
    ],
    "tests": [
        "pytest",
        "pytest-cov",
        gammapy,
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
    use_scm_version={"write_to": os.path.join("pyirf", "_version.py")},
    packages=find_packages(),
    install_requires=[
        "astropy~=4.0,>=4.0.2",
        "matplotlib",
        "numpy>=1.18",
        "scipy",
        "tqdm",
        "setuptools_scm",
    ],
    include_package_data=True,
    extras_require=extras_require,
)
