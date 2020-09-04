import os
import numpy as np
import astropy.units as u
from astropy.table import Table, Column
from astropy.io import fits
import pandas as pd

from gammapy.utils.nddata import NDDataArray, BinnedDataAxis
from gammapy.utils.energy import EnergyBounds
from gammapy.irf import EffectiveAreaTable, EnergyDispersion2D
from gammapy.spectrum import SensitivityEstimator

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

    def load_irf(self):
        filename = os.path.join(self.outdir, "irf.fits.gz")
        with fits.open(filename, memmap=False) as hdulist:
            aeff = EffectiveAreaTable.from_hdulist(hdulist=hdulist)
            edisp = EnergyDispersion2D.read(filename, hdu="ENERGY DISPERSION")

            bkg_fits_table = hdulist["BACKGROUND"]
            bkg_table = Table.read(bkg_fits_table)
            energy_lo = bkg_table["ENERG_LO"].quantity
            energy_hi = bkg_table["ENERG_HI"].quantity
            bkg = bkg_table["BKG"].quantity

            axes = [
                BinnedDataAxis(
                    energy_lo, energy_hi, interpolation_mode="log", name="energy"
                )
            ]
            bkg = BkgData(data=NDDataArray(axes=axes, data=bkg))

        # Create rmf with appropriate dimensions (e_reco->bkg, e_true->area)
        e_reco_min = bkg.energy.lo[0]
        e_reco_max = bkg.energy.hi[-1]
        e_reco_bin = bkg.energy.nbins
        e_reco_axis = EnergyBounds.equal_log_spacing(
            e_reco_min, e_reco_max, e_reco_bin, "TeV"
        )

        e_true_min = aeff.energy.lo[0]
        e_true_max = aeff.energy.hi[-1]
        e_true_bin = aeff.energy.nbins
        e_true_axis = EnergyBounds.equal_log_spacing(
            e_true_min, e_true_max, e_true_bin, "TeV"
        )

        # Fake offset...
        rmf = edisp.to_energy_dispersion(
            offset=0.5 * u.deg, e_reco=e_reco_axis, e_true=e_true_axis
        )

        self.irf = Irf(bkg=bkg, aeff=aeff, rmf=rmf)

    def estimate_sensitivity(self):
        obs_time = self.config["analysis"]["obs_time"]["value"] * u.Unit(
            self.config["analysis"]["obs_time"]["unit"]
        )
        sensitivity_estimator = SensitivityEstimator(irf=self.irf, livetime=obs_time)
        sensitivity_estimator.run()
        self.sens = sensitivity_estimator.results_table

        self.add_sensitivity_to_irf()

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
    evt_dict : `dict`
        Dict for each particle type, containing a table with the required column for IRF computing.
        TODO: define explicitely the name it expects.
    outdir: `str`
        Output directory where analysis results is saved
    """

    def __init__(self, config, evt_dict, outdir):
        self.config = config
        self.outdir = outdir

        # Read data saved on disk
        self.evt_dict = evt_dict
        # Loop on the particle type
        for particle in evt_dict.keys():
            self.evt_dict[particle] = evt_dict[particle]

        #Read table with cuts
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



        #cfg_binning_ereco = config["analysis"]["ereco_binning"]
        #cfg_binning_etrue = config["analysis"]["etrue_binning"]
        #self.nbin_ereco = cfg_binning_ereco["nbin"]
        #self.nbin_etrue = cfg_binning_etrue["nbin"]

        # Binning
        #self.ereco = np.logspace(
        #    np.log10(cfg_binning_ereco["emin"]),
        #    np.log10(cfg_binning_ereco["emax"]),
        #    self.nbin_ereco + 1,
        #)
        #self.etrue = np.logspace(
        #    np.log10(cfg_binning_etrue["emin"]),
        #    np.log10(cfg_binning_etrue["emax"]),
        #    self.nbin_etrue + 1,
        #)

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

    def make_bkg_rate(self):
        """Build background rate

         Parameters
        ----------
        angular_cut: `astropy.units.Quantity`, dimension N reco energy bin
            Array of angular cut to apply in each reconstructed energy bin
            to estimate the acceptance ratio for the background estimate
        """
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
                    (data_p["reco_energy"] >= emin)
                    & (data_p["reco_energy"] < emax)
                    & (data_p["pass_best_cutoff"])
                    ]["weight"]
            )

            n_e = sum(
                data_e[
                    (data_e["reco_energy"] >= emin)
                    & (data_e["reco_energy"] < emax)
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
        theta_lo = [0.0, 1.0]
        theta_hi = [1.0, 2.0]
        table_theta = Table()
        t["THETA_LO"] = Column(
            theta_lo,
            unit="deg",
            description="theta min",
            format=str(len(theta_lo)) + "E",
        )
        t["THETA_HI"] = Column(
            theta_hi,
            unit="deg",
            description="theta max",
            format=str(len(theta_hi)) + "E",
        )

        t["BKG"] = Column(bgd, unit="TeV", description="Background", format="E")

        return IrfMaker._make_hdu("BKG_2D", t, ["ENERG_LO", "ENERG_HI", "THETA_LO","THETA_HI", "BKG"])

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
            self, apply_score_cut=True, apply_angular_cut=True
    ):
        nbin = len(self.etrue) - 1
        energy_true_lo = np.zeros(nbin)
        energy_true_hi = np.zeros(nbin)
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
            energy_true_lo[ibin] = emin.value
            energy_true_hi[ibin] = emax.value

        table_energy = Table()
        table_energy["ETRUE_LO"] = Column(
            energy_true_lo,
            unit="TeV",
            description="energy min",
            format=str(len(energy_true_lo)) + "E",
        )
        table_energy["ETRUE_HI"] = Column(
            energy_true_hi,
            unit="TeV",
            description="energy max",
            format=str(len(energy_true_hi)) + "E",
        )

        # Needed for format, a bit hacky...
        # Those value are artificial. In the DL3 format, the IRFs are offset FOV dependant, here name theta.
        # For point-like MC simulation produced at only one offset theta0 this trick is needed because those IRF can
        # only be use for sources located at theta0 from the camera center. We give the same value for the IRFs for two
        # artificial offsets, like this the interpolation at theta0 in the high level analysis tools will
        # be correct. This will be remove when we use diffuse MC simulation that will allow to define IRF properly at
        # different offset in the FOV.
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

        extended_area = np.resize(
            area, (len(theta_lo), area.shape[0])
        )
        dim_extended_area = (
                len(table_energy["ETRUE_LO"])
                * len(table_theta["THETA_LO"])
        )

        aeff_2D = Table([extended_area], names=["AEFF"])
        aeff_2D["AEFF"].unit = u.Unit("m2")
        aeff_2D["AEFF"].format = str(dim_extended_area) + "E"

        hdu = IrfMaker._make_aeff_hdu(table_energy, table_theta, aeff_2D)

        return hdu

    def make_energy_dispersion(self):
        etrue = self.etrue#np.logspace(np.log10(0.01)), np.log10(10000), 60 + 1)
        migra_bin = self.config["analysis"]["emigra_binning"]['nbin']
        migra = np.linspace(0.0, 5.0, migra_bin)

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
        # Those value are artificial. In the DL3 format, the IRFs are offset FOV dependant, here name theta.
        # For point-like MC simulation produced at only one offset theta0 this trick is needed because those IRF can
        # only be use for sources located at theta0 from the camera center. We give the same value for the IRFs for two
        # artificial offsets, like this the interpolation at theta0 in the high level analysis tools will
        # be correct. This will be remove when we use diffuse MC simulation that will allow to define IRF properly at
        # different offset in the FOV.
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
        matrix = Table([extended_mig_matrix], names=["MATRIX"])
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
    def _make_aeff_hdu(cls, table_energy, table_theta, aeff):
        """Create the Bintable HDU for the effective area describe here
        https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html#effective-area-vs-true-energy
        """
        table = Table({
            'ENERG_LO': table_energy["ETRUE_LO"][np.newaxis, :].data * table_energy["ETRUE_LO"].unit,
            'ENERG_HI': table_energy["ETRUE_HI"][np.newaxis, :].data * table_energy["ETRUE_HI"].unit,
            'THETA_LO': table_theta["THETA_LO"][np.newaxis, :].data * table_theta["THETA_LO"].unit,
            'THETA_HI': table_theta["THETA_HI"][np.newaxis, :].data * table_theta["THETA_HI"].unit,
            'EFFAREA': aeff["AEFF"].data[np.newaxis, :, :] * aeff["AEFF"].unit,
        })

        header = fits.Header()
        header['HDUDOC'] = 'https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/index.html', ''
        header['HDUCLASS'] = 'aeff_2d', ''
        header['HDUCLAS1'] = 'aeff_2d', ''
        header['HDUCLAS2'] = 'EFF_AREA', ''
        header['HDUCLAS3'] = 'POINT-LIKE', ''
        header['HDUCLAS4'] = 'AEFF_2D', ''
        header['TELESCOP'] = 'CTA', ''
        header['INSTRUME'] = 'LST-1', ''

        aeff_hdu = fits.BinTableHDU(table, header, name='EFFECTIVE AREA')

        primary_hdu = fits.PrimaryHDU()
        hdulist = fits.HDUList([primary_hdu, aeff_hdu])

        return hdulist

    @classmethod
    def _make_edisp_hdu(cls, table_energy, table_migra, table_theta, matrix):
        """Create the Bintable HDU for the energy dispersion describe here
        https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/edisp/index.html
        """

        table = Table({
            'ENERG_LO': table_energy["ETRUE_LO"][np.newaxis, :].data * table_energy["ETRUE_LO"].unit,
            'ENERG_HI': table_energy["ETRUE_HI"][np.newaxis, :].data * table_energy["ETRUE_HI"].unit,
            'MIGRA_LO': table_migra["MIGRA_LO"][np.newaxis, :].data * table_migra["MIGRA_LO"].unit,
            'MIGRA_HI': table_migra["MIGRA_HI"][np.newaxis, :].data * table_migra["MIGRA_HI"].unit,
            'THETA_LO': table_theta["THETA_LO"][np.newaxis, :].data * table_theta["THETA_LO"].unit,
            'THETA_HI': table_theta["THETA_HI"][np.newaxis, :].data * table_theta["THETA_HI"].unit,
            'MATRIX': matrix["MATRIX"][np.newaxis, :, :] * matrix["MATRIX"].unit,
        })

        header = fits.Header()
        header['HDUDOC'] = 'https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/index.html', ''
        header['HDUCLASS'] = 'edisp_2d', ''
        header['HDUCLAS1'] = 'edisp_2d', ''
        header['HDUCLAS2'] = 'EDISP', ''
        header['HDUCLAS3'] = 'POINT-LIKE', ''
        header['HDUCLAS4'] = 'EDISP_2D', ''
        header['TELESCOP'] = 'CTA', ''
        header['INSTRUME'] = 'LST-1', ''

        edisp_hdu = fits.BinTableHDU(table, header, name='ENERGY DISPERSION')

        primary_hdu = fits.PrimaryHDU()
        hdulist = fits.HDUList([primary_hdu, edisp_hdu])

        return hdulist
