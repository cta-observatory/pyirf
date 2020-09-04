from setuptools import setup, find_packages

exec(open("pyirf/version.py").read())

setup(
    version=__version__,
    packages=find_packages(),
    package_data={"pyirf": ["resources/config.yml"]},
    include_package_data=True,
    install_requires=[
        "astropy",
        "ctaplot~=0.5.0",
        "gammapy==0.8",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "tables",
        "ctapipe==0.7",
    ],
)
