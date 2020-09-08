import os
import numpy as np
import astropy.units as u
from astropy.table import Table, Column
from astropy.io import fits
import pandas as pd

from gammapy.utils.nddata import NDDataArray #, BinnedDataAxis
from gammapy.maps import MapAxis
from gammapy.maps.utils import edges_from_lo_hi
# from gammapy.utils.energy import EnergyBounds
from gammapy.irf import EffectiveAreaTable, EnergyDispersion2D
# from gammapy.spectrum import SensitivityEstimator
from gammapy.estimators import SensitivityEstimator

__all__ = ["IrfMaker", "SensitivityMaker", "BkgData", "Irf"]


class BkgData(object):
    """
    Class storing background data in a NDDataArray object.

    It's a bit hacky, but gammapy sensitivity estimator does not take individual IRF,
    it takes a CTAPerf object. So i'm emulating that... We need a bkg format!!!
    """

    def __init__(self, data):
        self.data = data

    @property
    def energy(self):
        """Get energy."""
        return self.data.axes[0]


class Irf(object):
    """
    Class storing IRF for sensitivity computation (emulating CTAPerf)
    """

    def __init__(self, bkg, aeff, rmf):
        self.bkg = bkg
        self.aeff = aeff
        self.rmf = rmf


class SensitivityMaker(object):
    """
    Class which estimate sensitivity with IRF

    Parameters
    ----------
    config: `dict`
        Configuration file
    outdir: `str`
        Output directory where analysis results is saved
    """

    def __init__(self, config, outdir):
        self.config = config
        self.outdir = outdir
        self.irf = None
        self.arf = None
        self.rmf = None
        self.bkg = None

    def load_irf(self):
        filename = os.path.join(self.outdir, "irf.fits.gz")
        with fits.open(filename, memmap=False) as hdulist:
            aeff = EffectiveAreaTable.from_hdulist(hdulist=hdulist)
            edisp = EnergyDispersion2D.read(filename, hdu="ENERGY DISPERSION")

            bkg_fits_table = hdulist["BACKGROUND"]
            bkg_table = Table.read(bkg_fits_table)
            energy_lo = bkg_table["ENERG_LO"].quantity
            energy_hi = bkg_table["ENERG_HI"].quantity
            bkg = bkg_table["BGD"].quantity

            # axes = [
            #     BinnedDataAxis(
            #         energy_lo, energy_hi, interpolation_mode="log", name="energy"
            #     )
            # ]
            axes = [
                MapAxis.from_edges(edges_from_lo_hi(energy_lo, energy_hi), interp='log', name='energy')
            ]
            bkg = BkgData(data=NDDataArray(axes=axes, data=bkg))

        # Create rmf with appropriate dimensions (e_reco->bkg, e_true->area)
        # e_reco_min = bkg.energy.lo[0]
        # e_reco_max = bkg.energy.hi[-1]
        # e_reco_bin = bkg.energy.nbins
        e_reco_min = bkg.energy.edges[0]
        e_reco_max = bkg.energy.edges[-1]
        e_reco_bin = bkg.energy.nbin

        # e_reco_axis = EnergyBounds.equal_log_spacing(
        #     e_reco_min, e_reco_max, e_reco_bin, "TeV"
        # )
        e_reco_axis = MapAxis.from_energy_bounds(
            e_reco_min, e_reco_max, e_reco_bin, "TeV"
        ).edges

        # e_true_min = aeff.energy.lo[0]
        # e_true_max = aeff.energy.hi[-1]
        # e_true_bin = aeff.energy.nbins
        e_true_min = aeff.energy.edges[0]
        e_true_max = aeff.energy.edges[-1]
        e_true_bin = aeff.energy.nbin

        # e_true_axis = EnergyBounds.equal_log_spacing(
        #     e_true_min, e_true_max, e_true_bin, "TeV"
        # )
        e_true_axis = MapAxis.from_energy_bounds(
            e_true_min, e_true_max, e_true_bin, "TeV"
        ).edges

        # Fake offset...
        rmf = edisp.to_energy_dispersion(
            offset=0.5 * u.deg, e_reco=e_reco_axis, e_true=e_true_axis
        )
        self.arf = aeff
        self.bkg = bkg
        self.rmf = rmf
        self.irf = Irf(bkg=bkg, aeff=aeff, rmf=rmf)

    def estimate_sensitivity(self):
        obs_time = self.config["analysis"]["obs_time"]["value"] * u.Unit(
            self.config["analysis"]["obs_time"]["unit"]
        )
        # sensitivity_estimator = SensitivityEstimator(irf=self.irf, livetime=obs_time)
        # sensitivity_estimator = SensitivityEstimator(arf=self.arf,
        #                                             rmf=self.rmf,
        #                                             bkg=self.bkg,
        #                                             livetime=obs_time,
        #                                             )
        # from gammapy.datasets.spectrum import SpectrumDatasetOnOff
        # dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
        #     dataset=spectrum_dataset, acceptance=1, acceptance_off=1,
        # )
        #
        # sensitivity_estimator.run()
        # self.sens = sensitivity_estimator.results_table
        #
        # self.add_sensitivity_to_irf()
        pass

    def add_sensitivity_to_irf(self):
        cfg_binning = self.config["analysis"]["ereco_binning"]
        ereco = np.logspace(
            np.log10(cfg_binning["emin"]),
            np.log10(cfg_binning["emax"]),
            cfg_binning["nbin"] + 1,
        )

        t = Table()
        t["ENERG_LO"] = Column(
            self.irf.bkg.energy.lo.value,
            unit="TeV",
            description="energy min",
            format="E",
        )
        t["ENERG_HI"] = Column(
            self.irf.bkg.energy.hi.value,
            unit="TeV",
            description="energy max",
            format="E",
        )
        t["SENSITIVITY"] = Column(
            self.sens["e2dnde"],
            unit="erg/(cm2 s)",
            description="sensitivity",
            format="E",
        )
        t["EXCESS"] = Column(
            self.sens["excess"], unit="", description="excess", format="E"
        )
        t["BKG"] = Column(
            self.sens["background"], unit="", description="bkg", format="E"
        )

        filename = os.path.join(self.outdir, "irf.fits.gz")
        hdulist = fits.open(filename, memmap=False, mode="update")
        col_list = [
            fits.Column(col.name, col.format, unit=str(col.unit), array=col.data)
            for col in t.columns.values()
        ]
        sens_hdu = fits.BinTableHDU.from_columns(col_list)
        sens_hdu.header.set("EXTNAME", "SENSITIVITY")
        hdulist.append(sens_hdu)
        hdulist.flush()


class IrfMaker(object):
    """
    Class building IRF for point-like analysis.

    Parameters
    ----------
    config: `dict`
        Configuration file
    outdir: `str`
        Output directory where analysis results is saved
    """

    def __init__(self, config, evt_dict, outdir):
        self.config = config
        self.outdir = outdir

        # Read data saved on disk
        self.evt_dict = {}
        for particle in ["gamma", "electron", "proton"]:
            self.evt_dict[particle] = pd.read_hdf(
                os.path.join(outdir, "{}_processed.h5".format(particle))
            )

        # Read table with cuts
        self.table = Table.read(
            os.path.join(
                outdir, "{}.fits".format(config["general"]["output_table_name"])
            ),
            format="fits",
        )
        self.table = self.table[np.where(self.table["keep"].data)[0]]

        # Binning
        self.ereco = np.logspace(
            np.log10(self.table["emin"][0]),
            np.log10(self.table["emax"][-1]),
            len(self.table),
        )

        cfg_binning = config["analysis"]["etrue_binning"]
        self.etrue = np.logspace(
            np.log10(cfg_binning["emin"]),
            np.log10(cfg_binning["emax"]),
            cfg_binning["nbin"] + 1,
        )

    def build_irf(self):
        bkg_rate = self.make_bkg_rate()
        psf = self.make_point_spread_function()
        area = self.make_effective_area(
            apply_score_cut=True, apply_angular_cut=True, hdu_name="SPECRESP"
        )  # Effective area with cuts applied
        edisp = self.make_energy_dispersion()

        # Add usefull effective areas for debugging
        area_no_cuts = self.make_effective_area(
            apply_score_cut=False,
            apply_angular_cut=False,
            hdu_name="SPECRESP (NO CUTS)",
        )  # Effective area with cuts applied
        area_no_score_cut = self.make_effective_area(
            apply_score_cut=False,
            apply_angular_cut=True,
            hdu_name="SPECRESP (WITH ANGULAR CUT)",
        )  # Effective area with cuts applied
        area_no_angular_cut = self.make_effective_area(
            apply_score_cut=True,
            apply_angular_cut=False,
            hdu_name="SPECRESP (WITH SCORE CUT)",
        )  # Effective area with cuts applied

        # Primary header
        n = np.arange(100.0)
        primary_hdu = fits.PrimaryHDU(n)

        # Fill HDU list
        hdulist = fits.HDUList(
            [
                primary_hdu,
                area,
                psf,
                edisp,
                bkg_rate,
                area_no_cuts,
                area_no_score_cut,
                area_no_angular_cut,
            ]
        )

        hdulist.writeto(os.path.join(self.outdir, "irf.fits.gz"), overwrite=True)

    def compute_acceptance(self):
        """
        Compute acceptance gamma/background
        """


    def make_bkg_rate(self):
        """Build background rate"""
        nbin = len(self.table)
        energ_lo = np.zeros(nbin)
        energ_hi = np.zeros(nbin)
        bgd = np.zeros(nbin)

        obs_time = self.config["analysis"]["obs_time"]["value"] * u.Unit(
            self.config["analysis"]["obs_time"]["unit"]
        )

        for ibin, info in enumerate(self.table):
            energ_lo[ibin] = info["emin"]
            energ_hi[ibin] = info["emax"]

            # References
            data_p = self.evt_dict["proton"]
            data_e = self.evt_dict["electron"]

            # Compute number of events passing cuts selection
            n_p = sum(
                data_p[
                    (data_p["reco_energy"] >= info["emin"])
                    & (data_p["reco_energy"] < info["emax"])
                    & (data_p["pass_best_cutoff"])
                    ]["weight"]
            )

            n_e = sum(
                data_e[
                    (data_e["reco_energy"] >= info["emin"])
                    & (data_e["reco_energy"] < info["emax"])
                    & (data_e["pass_best_cutoff"])
                    ]["weight"]
            )

            # Correct number of background due to acceptance
            acceptance_g = (
                    2 * np.pi * (1 - np.cos(info["angular_cut"] * u.deg.to("rad")))
            )
            acceptance_p = (
                    2
                    * np.pi
                    * (
                            1
                            - np.cos(
                        self.config["particle_information"]["proton"]["offset_cut"]
                        * u.deg.to("rad")
                    )
                    )
            )
            acceptance_e = (
                    2
                    * np.pi
                    * (
                            1
                            - np.cos(
                        self.config["particle_information"]["electron"]["offset_cut"]
                        * u.deg.to("rad")
                    )
                    )
            )

            n_p *= acceptance_g / acceptance_p
            n_e *= acceptance_g / acceptance_e
            bgd[ibin] = (n_p + n_e) / obs_time.to("s").value

        t = Table()
        t["ENERG_LO"] = Column(
            energ_lo, unit="TeV", description="energy min", format="E"
        )
        t["ENERG_HI"] = Column(
            energ_hi, unit="TeV", description="energy max", format="E"
        )
        t["BGD"] = Column(bgd, unit="1/s", description="Background", format="E")

        return IrfMaker._make_hdu("BACKGROUND", t, ["ENERG_LO", "ENERG_HI", "BGD"])

    def make_point_spread_function(self, radius=68):
        """Buil point spread function with radius containment `radius`"""
        nbin = len(self.table)
        energ_lo = np.zeros(nbin)
        energ_hi = np.zeros(nbin)
        psf = np.zeros(nbin)

        for ibin, info in enumerate(self.table):
            energ_lo[ibin] = info["emin"]
            energ_hi[ibin] = info["emax"]

            # References
            data_g = self.evt_dict["gamma"]

            # Select data passing cuts selection
            sel = data_g.loc[
                (data_g["reco_energy"] >= info["emin"])
                & (data_g["reco_energy"] < info["emax"])
                & (data_g["pass_best_cutoff"]),
                [self.config['column_definition']['angular_distance_to_the_src']],
            ]

            # Compute PSF
            psf[ibin] = np.percentile(sel[self.config['column_definition']['angular_distance_to_the_src']], radius)

        t = Table()
        t["ENERG_LO"] = Column(
            energ_lo, unit="TeV", description="energy min", format="E"
        )
        t["ENERG_HI"] = Column(
            energ_hi, unit="TeV", description="energy max", format="E"
        )
        t["PSF68"] = Column(psf, unit="TeV", description="PSF", format="E")

        return IrfMaker._make_hdu(
            "POINT SPREAD FUNCTION", t, ["ENERG_LO", "ENERG_HI", "PSF68"]
        )

    def make_effective_area(
            self, apply_score_cut=True, apply_angular_cut=True, hdu_name="SPECRESP"
    ):
        nbin = len(self.etrue) - 1
        energ_lo = np.zeros(nbin)
        energ_hi = np.zeros(nbin)
        area = np.zeros(nbin)

        # Get simulation infos
        cfg_particule = self.config["particle_information"]["gamma"]
        simu_index = cfg_particule["gen_gamma"]
        index = 1.0 - simu_index  # for futur integration
        nsimu_tot = float(cfg_particule["n_files"]) * float(
            cfg_particule["n_events_per_file"]
        )
        emin_simu = cfg_particule["e_min"]
        emax_simu = cfg_particule["e_max"]
        area_simu = (np.pi * cfg_particule["gen_radius"] ** 2) * u.Unit("m2")

        for ibin in range(nbin):

            emin = self.etrue[ibin] * u.TeV
            emax = self.etrue[ibin + 1] * u.TeV

            # References
            data_g = self.evt_dict["gamma"]

            # Conditions to select gamma-rays
            condition = (data_g["mc_energy"] >= emin) & (data_g["mc_energy"] < emax)
            if apply_score_cut is True:
                condition &= data_g["pass_best_cutoff"]
            if apply_angular_cut is True:
                condition &= data_g["pass_angular_cut"]

            # Compute number of events passing cuts selection
            sel = len(data_g.loc[condition, ["weight"]])

            # Compute number of number of events in simulation
            simu_evts = (
                    float(nsimu_tot)
                    * (emax.value ** index - emin.value ** index)
                    / (emax_simu ** index - emin_simu ** index)
            )

            area[ibin] = (sel / simu_evts) * area_simu.value
            energ_lo[ibin] = emin.value
            energ_hi[ibin] = emax.value

        t = Table()
        t["ENERG_LO"] = Column(
            energ_lo, unit="TeV", description="energy min", format="E"
        )
        t["ENERG_HI"] = Column(
            energ_hi, unit="TeV", description="energy max", format="E"
        )
        t[hdu_name] = Column(area, unit="m2", description="Effective area", format="E")

        return IrfMaker._make_hdu(hdu_name, t, ["ENERG_LO", "ENERG_HI", hdu_name])

    def make_energy_dispersion(self):
        migra = np.linspace(0.0, 3.0, 300 + 1)
        etrue = np.logspace(np.log10(0.01), np.log10(10000), 60 + 1)
        counts = np.zeros([len(migra) - 1, len(etrue) - 1])

        # Select events
        data_g = self.evt_dict["gamma"]
        data_g = data_g[
            (data_g["pass_best_cutoff"]) & (data_g["pass_angular_cut"])
            ].copy()

        for imigra in range(len(migra) - 1):
            migra_min = migra[imigra]
            migra_max = migra[imigra + 1]

            for ietrue in range(len(etrue) - 1):
                emin = etrue[ietrue]
                emax = etrue[ietrue + 1]

                sel = len(
                    data_g[
                        (data_g["mc_energy"] >= emin)
                        & (data_g["mc_energy"] < emax)
                        & ((data_g["reco_energy"] / data_g["mc_energy"]) >= migra_min)
                        & ((data_g["reco_energy"] / data_g["mc_energy"]) < migra_max)
                        ]
                )
                counts[imigra][ietrue] = sel

        table_energy = Table()
        table_energy["ETRUE_LO"] = Column(
            etrue[:-1],
            unit="TeV",
            description="energy min",
            format=str(len(etrue) - 1) + "E",
        )
        table_energy["ETRUE_HI"] = Column(
            etrue[1:],
            unit="TeV",
            description="energy max",
            format=str(len(etrue) - 1) + "E",
        )

        table_migra = Table()
        table_migra["MIGRA_LO"] = Column(
            migra[:-1],
            unit="",
            description="migra min",
            format=str(len(migra) - 1) + "E",
        )
        table_migra["MIGRA_HI"] = Column(
            migra[1:],
            unit="",
            description="migra max",
            format=str(len(migra) - 1) + "E",
        )

        # Needed for format, a bit hacky...
        theta_lo = [0.0, 1.0]
        theta_hi = [1.0, 2.0]
        table_theta = Table()
        table_theta["THETA_LO"] = Column(
            theta_lo,
            unit="deg",
            description="theta min",
            format=str(len(theta_lo)) + "E",
        )
        table_theta["THETA_HI"] = Column(
            theta_hi,
            unit="deg",
            description="theta max",
            format=str(len(theta_hi)) + "E",
        )

        extended_mig_matrix = np.resize(
            counts, (len(theta_lo), counts.shape[0], counts.shape[1])
        )
        dim_matrix = (
                len(table_energy["ETRUE_LO"])
                * len(table_migra["MIGRA_LO"])
                * len(table_theta["THETA_LO"])
        )
        matrix = Table([extended_mig_matrix.ravel()], names=["MATRIX"])
        matrix["MATRIX"].unit = u.Unit("")
        matrix["MATRIX"].format = str(dim_matrix) + "E"
        hdu = IrfMaker._make_edisp_hdu(table_energy, table_migra, table_theta, matrix)

        return hdu

    @classmethod
    def _make_hdu(cls, hdu_name, t, cols):
        """List of columns"""
        col_list = [
            fits.Column(col.name, col.format, unit=str(col.unit), array=col.data)
            for col in t.columns.values()
        ]
        hdu = fits.BinTableHDU.from_columns(col_list)
        hdu.header.set("EXTNAME", hdu_name)
        return hdu

    @classmethod
    def _make_edisp_hdu(cls, table_energy, table_migra, table_theta, matrix):
        """List of columns"""
        hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="ETRUE_LO",
                    format=table_energy["ETRUE_LO"].format,
                    unit=table_energy["ETRUE_LO"].unit.to_string(),
                    array=np.atleast_2d(table_energy["ETRUE_LO"]),
                ),
                fits.Column(
                    "ETRUE_HI",
                    table_energy["ETRUE_HI"].format,
                    unit=table_energy["ETRUE_HI"].unit.to_string(),
                    array=np.atleast_2d(table_energy["ETRUE_HI"]),
                ),
                fits.Column(
                    "MIGRA_LO",
                    table_migra["MIGRA_LO"].format,
                    unit=table_migra["MIGRA_LO"].unit.to_string(),
                    array=np.atleast_2d(table_migra["MIGRA_LO"]),
                ),
                fits.Column(
                    "MIGRA_HI",
                    table_migra["MIGRA_HI"].format,
                    unit=table_migra["MIGRA_HI"].unit.to_string(),
                    array=np.atleast_2d(table_migra["MIGRA_HI"]),
                ),
                fits.Column(
                    "THETA_LO",
                    table_theta["THETA_LO"].format,
                    unit=table_theta["THETA_LO"].unit.to_string(),
                    array=np.atleast_2d(table_theta["THETA_LO"]),
                ),
                fits.Column(
                    "THETA_HI",
                    table_theta["THETA_HI"].format,
                    unit=table_theta["THETA_HI"].unit.to_string(),
                    array=np.atleast_2d(table_theta["THETA_HI"]),
                ),
                fits.Column(
                    "MATRIX",
                    matrix["MATRIX"].format,
                    unit=matrix["MATRIX"].unit.to_string(),
                    array=np.expand_dims(matrix["MATRIX"], 0),
                ),
            ]
        )

        hdu.header.set(
            "TDIM7",
            "("
            + str(len(table_energy["ETRUE_LO"]))
            + ","
            + str(len(table_migra["MIGRA_LO"]))
            + ","
            + str(len(table_theta["THETA_LO"]))
            + ")",
        )
        hdu.header.set(
            "EXTNAME", "ENERGY DISPERSION", "name of this binary table extension "
        )
        return hdu
