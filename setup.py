from setuptools import setup, find_packages
import os


gammapy = "gammapy~=1.0"

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
        "towncrier",
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
    ],
}
extras_require["dev"] = extras_require["tests"] + [
    "setuptools_scm",
]

all_extras = set()
for extra in extras_require.values():
    all_extras.update(extra)
extras_require["all"] = list(all_extras)

setup(
    use_scm_version={"write_to": os.path.join("pyirf", "_version.py")},
    packages=find_packages(exclude=['pyirf._dev_version']),
    install_requires=[
        "astropy>=4.0.2",
        "matplotlib",
        "numpy>=1.18",
        "scipy",
        "tqdm",
    ],
    include_package_data=True,
    extras_require=extras_require,
)
