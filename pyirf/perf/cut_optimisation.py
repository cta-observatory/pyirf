import os
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.table import Table, Column
# from gammapy.spectrum.models import PowerLaw
from gammapy.modeling.models import PowerLawSpectralModel
# from gammapy.stats import significance_on_off
from gammapy.stats import WStatCountsStatistic

from .utils import save_obj, load_obj, plot_hist

__all__ = ['CutsOptimisation', 'CutsDiagnostic', 'CutsApplicator']


class CutsApplicator(object):
    """
    Apply best cut and angular cut to events.

    Apply cuts to gamma, proton and electrons that will be further used for
    performance estimation (irf, sensitivity, etc.).

    Parameters
    ----------
    config: `dict`
        Configuration file
    outdir: `str`
        Output directory where analysis results is saved
    evt_dict: `dict`
        Dictionary of `pandas.DataFrame`
    """
    def __init__(self, config, evt_dict, outdir):
        self.config = config
        self.evt_dict = evt_dict
        self.outdir = outdir

        # Read table with cuts
        self.table = Table.read(
            os.path.join(outdir, '{}.fits'.format(config['general']['output_table_name'])),
            format='fits'
        )

    def apply_cuts(self):
        """
        Flag particles (gamma; hadron, electron) passing angular cut or the best cutoff
        and the save the data
        """
        for particle in self.evt_dict.keys():
            data = self.apply_cuts_on_data(self.evt_dict[particle].copy())
            data.to_hdf(
                os.path.join(self.outdir, '{}_processed.h5'.format(particle)),
                key='dl2',
                mode='w'
            )

    def apply_cuts_on_data(self, data):
        """
        Flag particle passing angular cut and the best cutoff

        Parameters
        ----------
        data: `pandas.DataFrame`
            Data set corresponding to one type of particle
        """
        # Add columns with False initialisation
        data['pass_best_cutoff'] = np.zeros(len(data), dtype=bool)
        data['pass_angular_cut'] = np.zeros(len(data), dtype=bool)

        colname_reco_energy = self.config['column_definition']['reco_energy']
        colname_clf_output = self.config['column_definition']['classification_output']['name']
        colname_angular_dist = self.config['column_definition']['angular_distance_to_the_src']

        # Loop over energy bins and apply cutoff for each slice
        table = self.table[np.where(self.table['keep'].data)[0]]
        for info in table:

            # print('Processing bin [{:.3f},{:.3f}]... (cut={:.3f}, theta={:.3f})'.format(
            #     info['emin'], info['emax'], info['best_cutoff'], info['angular_cut']
            # ))

            # Best cutoff
            data.loc[(data[colname_reco_energy] >= info['emin']) &
                     (data[colname_reco_energy] < info['emax']) &
                     (data[colname_clf_output] >= info['best_cutoff']), ['pass_best_cutoff']] = True
            # Angular cut
            data.loc[(data[colname_reco_energy] >= info['emin']) &
                     (data[colname_reco_energy] < info['emax']) &
                     (data[colname_angular_dist] <= info['angular_cut']), ['pass_angular_cut']] = True

        # Handle events which are not in energy range
        # Best cutoff
        data.loc[(data[colname_reco_energy] < table['emin'][0]) &
                 (data[colname_clf_output] >= table['best_cutoff'][0]), ['pass_best_cutoff']] = True
        data.loc[(data[colname_reco_energy] >= table['emin'][-1]) &
                 (data[colname_clf_output] >= table['best_cutoff'][-1]), ['pass_best_cutoff']] = True
        # Angular cut
        data.loc[(data[colname_reco_energy] < table['emin'][0]) &
                 (data[colname_angular_dist] <= table['angular_cut'][0]), ['pass_angular_cut']] = True
        data.loc[(data[colname_reco_energy] >= table['emin'][-1]) &
                 (data[colname_angular_dist] <= table['angular_cut'][-1]), ['pass_angular_cut']] = True

        return data


class CutsDiagnostic(object):
    """
    Class used to get some diagnostic related to the optimal working point.

    Parameters
    ----------
    config: `dict`
        Configuration file
    indir: `str`
        Output directory where analysis results is located
    """
    def __init__(self, config, indir):
        self.config = config
        self.indir = indir
        self.outdir = os.path.join(indir, 'diagnostic')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.table = Table.read(
            os.path.join(indir, '{}.fits'.format(config['general']['output_table_name'])),
            format='fits'
        )

        self.clf_output_bounds = self.config['column_definition']['classification_output']['range']

    def plot_optimisation_summary(self):
        """Plot efficiencies and angular cut as a function of energy bins"""
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        t = self.table[np.where(self.table['keep'].data)[0]]

        ax.plot(np.sqrt(t['emin'] * t['emax']), t['eff_sig'], color='blue', marker='o',
                label='Signal')
        ax.plot(np.sqrt(t['emin'] * t['emax']), t['eff_bkg'], color='red', marker='o',
                label='Background (p+e)')
        ax.grid(which='both')
        ax.set_xlabel('Reco energy [TeV]')
        ax.set_ylabel('Efficiencies')
        ax.set_xscale('log')
        ax.set_ylim([0., 1.1])

        ax_th = ax.twinx()
        ax_th.plot(np.sqrt(t['emin'] * t['emax']), t['angular_cut'], color='darkgreen',
                   marker='s')
        ax_th.set_ylabel('Angular cut [deg]', color='darkgreen')
        ax_th.tick_params('y', colors='darkgreen', )
        ax_th.set_ylim([0., 0.5])

        ax.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, 'efficiencies.pdf'))

        return ax

    def plot_diagnostics(self):
        """Plot efficiencies and rates as a function of score"""

        for info in self.table[np.where(self.table['keep'].data)[0]]:
            obj_name = 'diagnostic_data_emin{:.3f}_emax{:.3f}.pkl.gz'.format(
                info['emin'], info['emax']
            )
            data = load_obj(os.path.join(self.outdir, obj_name))

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax_eff = axes[0]
            ax_rate = axes[1]

            ax_eff = self.plot_efficiencies_vs_score(ax_eff, data, info)
            ax_rate = self.plot_rates_vs_score(ax_rate, data, info,
                                               self.config['analysis']['obs_time']['unit'])

            ax_eff.set_xlim(self.clf_output_bounds)
            ax_rate.set_xlim(self.clf_output_bounds)
            #print('JLK HAAAAACCCCKKKKKKK!!!!')
            #ax_eff.set_xlim(-0.5, 0.5)
            #ax_rate.set_xlim(-0.5, 0.5)

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.outdir, 'diagnostic_{:.2f}_{:.2f}TeV.pdf'.format(
                    info['emin'], info['emax']
                ))
            )

    @classmethod
    def plot_efficiencies_vs_score(cls, ax, data, info):
        """Plot efficiencies as a function of score"""
        ax.plot(data['score'], data['hist_eff_sig'], color='blue',
                label='Signal', lw=2)

        ax.plot(data['score'], data['hist_eff_bkg'], color='red',
                label='Background (p+e)', lw=2)

        ax.plot([info['best_cutoff'], info['best_cutoff']], [0, 1.1], ls='--', lw=2,
                color='darkgreen', label='Best cutoff')

        ax.set_xlabel('Score')
        ax.set_ylabel('Efficiencies')
        ax.set_ylim([0., 1.1])
        ax.grid(which='both')
        ax.legend(loc='lower left', framealpha=1)
        return ax

    @classmethod
    def plot_rates_vs_score(cls, ax, data, info, time_unit):
        """Plot rates as a function of score"""
        scale = info['min_flux']

        opt = {'edgecolor': 'blue', 'color': 'blue', 'label': 'Excess in ON region',
               'alpha': 0.2, 'fill': True, 'ls': '-', 'lw': 1}
        error_kw = dict(ecolor='blue', lw=1, capsize=1, capthick=1, alpha=1)
        ax = plot_hist(ax=ax, data=(data['cumul_excess'] * scale) /
                                   (info['obs_time'] * u.Unit(time_unit).to('s')),
                       edges=data['score_edges'], norm=False,
                       yerr=False, error_kw=error_kw, hist_kwargs=opt)

        opt = {'edgecolor': 'red', 'color': 'red', 'label': 'Bkg in ON region',
               'alpha': 0.2, 'fill': True, 'ls': '-', 'lw': 1}
        error_kw = dict(ecolor='red', lw=1, capsize=1, capthick=1, alpha=1)
        ax = plot_hist(ax=ax, data=data['cumul_noff'] * info['alpha'] / (info['obs_time'] * u.Unit(time_unit).to('s')),
                       edges=data['score_edges'], norm=False,
                       yerr=False, error_kw=error_kw, hist_kwargs=opt)

        ax.plot([info['best_cutoff'], info['best_cutoff']], [0, 1.1], ls='--', lw=2,
                color='darkgreen', label='Best cutoff')

        max_rate_p = (data['cumul_noff'] * info['alpha'] / (info['obs_time'] * u.Unit(time_unit).to('s'))).max()
        max_rate_g = (data['cumul_excess'] / (info['obs_time'] * u.Unit(time_unit).to('s'))).max()

        scaled_rate = max_rate_g * scale
        max_rate = scaled_rate if scaled_rate >= max_rate_p else max_rate_p

        ax.set_ylim([0., max_rate * 1.15])
        ax.set_ylabel('Rates [HZ]')
        ax.set_xlabel('Score')
        ax.grid(which='both')
        ax.legend(loc='upper right', framealpha=1)

        ax.text(
            0.52, 0.35, CutsDiagnostic.get_text(info),
            horizontalalignment='left',
            verticalalignment='bottom',
            multialignment='left',
            bbox=dict(facecolor='white', alpha=0.5),
            transform=ax.transAxes
        )
        return ax

    @classmethod
    def get_text(cls, info):
        """Returns a text summarising the optimisation result"""
        text = 'E in [{:.2f},{:.2f}] TeV\n'.format(info['emin'], info['emax'])
        text += 'Theta={:.2f} deg\n'.format(info['angular_cut'])
        text += 'Best cutoff:\n'
        text += '-min_flux={:.2f} Crab\n'.format(info['min_flux'])
        text += '-score={:.2f}\n'.format(info['best_cutoff'])
        text += '-non={:.2f}\n'.format(info['non'])
        text += '-noff={:.2f}\n'.format(info['noff'])
        text += '-alpha={:.2f}\n'.format(info['alpha'])
        text += '-excess={:.2f}'.format(info['excess'])
        if info['systematic'] is True:
            text += '(syst.!)\n'
        else:
            text += '\n'
        text += '-nbkg={:.2f}\n'.format(info['background'])
        text += '-sigma={:.2f} (Li & Ma)'.format(info['sigma'])

        return text


class CutsOptimisation(object):
    """
    Class used to find best cutoff to obtain minimal
    sensitivity in a given amount of time.

    Parameters
    ----------
    config: `dict`
        Configuration file
    evt_dict: `dict`
        Dictionary of `pandas` files
    """
    def __init__(self, config, evt_dict, verbose_level=0):
        self.config = config
        self.evt_dict = evt_dict
        self.verbose_level = verbose_level

    def weight_events(self, model_dict, colname_mc_energy):
        """
        Add a weight column to the files, in order to scale simulated data to reality.

        Parameters
        ----------
        model_dict: dict
            Dictionary of models
        colname_mc_energy: str
            Column name for the true energy
        """
        for particle in self.evt_dict.keys():
            self.evt_dict[particle]['weight'] = self.compute_weight(
                energy=self.evt_dict[particle][colname_mc_energy].values * u.TeV,
                particle=particle,
                model=model_dict[particle]
            )

    def compute_weight(self, energy, particle, model):
        """
        Weight particles, according to: [phi_exp(E) / phi_simu(E)] * (t_obs / t_simu)
        where E is the true energy of the particles
        """
        conf_part = self.config['particle_information'][particle]

        area_simu = (np.pi * conf_part['gen_radius'] ** 2) * u.Unit('m2')

        omega_simu = 2 * np.pi * (
                    1 - np.cos(conf_part['diff_cone'] * np.pi / 180.)) * u.sr
        if particle in 'gamma':  # Gamma are point-like
            omega_simu = 1.

        nsimu = conf_part['n_simulated']
        index_simu = conf_part['gen_gamma']
        emin = conf_part['e_min'] * u.TeV
        emax = conf_part['e_max'] * u.TeV
        amplitude = 1. * u.Unit('1 / (cm2 s TeV)')
        # pwl_integral = PowerLaw(
        #     index=index_simu, amplitude=amplitude).integral(emin=emin, emax=emax)
        pwl_integral = PowerLawSpectralModel(
            index=index_simu, amplitude=amplitude).integral(emin=emin, emax=emax)

        tsimu = nsimu / (area_simu * pwl_integral)
        tobs = self.config['analysis']['obs_time']['value'] * u.Unit(self.config['analysis']['obs_time']['unit'])

        phi_simu = amplitude * (energy / (1 * u.TeV)) ** (-index_simu) #/ omega_simu

        # if particle in 'proton':
        #     phi_exp = model(energy, 'proton')
        # elif particle in 'electron':
        #     phi_exp = model(energy, 'electron')
        # elif particle in 'gamma':
        #     phi_exp = model(energy)
        # else:
        #     print('oups...')
        phi_exp = model(energy)

        return ((tobs / tsimu) * (phi_exp / phi_simu)).decompose()

    def find_best_cutoff(self, energy_values, angular_values):
        """
        Find best cutoff to reach the best sensitivity. Optimisation is done as a function
        of energy and theta square cut. Correct the number of events
        according to the ON region which correspond to the angular cut applied to
        the gamma-ray events.

        Parameters
        ----------
        energy_values: `astropy.Quantity`
            Energy bins
        angular_values: `astropy.Quantity`
            Angular cuts
        """
        self.results_dict = dict()
        colname_reco_energy = self.config['column_definition']['reco_energy']
        clf_output_bounds = self.config['column_definition']['classification_output']['range']
        colname_angular_dist = self.config['column_definition']['angular_distance_to_the_src']
        thsq_opt_type = self.config['analysis']['thsq_opt']['type']

        # Loop on energy
        for ibin in range(len(energy_values) - 1):
            emin = energy_values[ibin]
            emax = energy_values[ibin + 1]
            print(' ==> {}) Working in E=[{:.3f},{:.3f}]'.format(ibin, emin, emax))

            # Apply cuts (energy and additional if there is)
            query_emin = '{} > {}'.format(colname_reco_energy, emin.value)
            query_emax = '{} <= {}'.format(colname_reco_energy, emax.value)
            energy_query = '{} and {}'.format(query_emin, query_emax)

            g = self.evt_dict['gamma'].query(energy_query).copy()
            p = self.evt_dict['proton'].query(energy_query).copy()
            e = self.evt_dict['electron'].query(energy_query).copy()

            if self.verbose_level > 0:
                print('Total evts for optimisation: Ng={}, Np={}, Ne={}'.format(
                    len(g), len(p), len(e))
                )

            min_stat = 100
            if len(g) <= min_stat or len(g) <= min_stat or len(g) <= min_stat:
                print('Not enough statistics')
                print('  g={}, p={} e={}'.format(len(g), len(p), len(e)))
                key = CutsOptimisation._get_energy_key(emin, emax)
                self.results_dict[key] = {'emin': emin.value, 'emax': emax.value,
                                          'keep': False}
                continue

            # To store intermediate results
            results_th_cut_dict = dict()

            theta_to_loop_on = angular_values
            if thsq_opt_type in 'r68':
                theta_to_loop_on = [angular_values[ibin]]

            # Loop on angular cut
            for th_cut in theta_to_loop_on:
                if self.verbose_level > 0:
                    print('- Theta={:.2f}'.format(th_cut))

                # Select gamma-rays in ON region
                th_query = '{} <= {}'.format(colname_angular_dist, th_cut.value)
                sel_g = g.query(th_query).copy()

                # Correct number of background due to acceptance
                acceptance_g = 2 * np.pi * (1 - np.cos(th_cut.to('rad').value))
                acceptance_p = 2 * np.pi * (
                        1 - np.cos(self.config['particle_information']['proton']['offset_cut'] * u.deg.to('rad'))
                )
                acceptance_e = 2 * np.pi * (
                        1 - np.cos(self.config['particle_information']['electron']['offset_cut'] * u.deg.to('rad'))
                )

                # Add corrected weight taking into angular cuts applied to gamma-rays
                sel_g['weight_corrected'] = sel_g['weight']
                p['weight_corrected'] = p['weight'] * acceptance_g / acceptance_p
                e['weight_corrected'] = e['weight'] * acceptance_g / acceptance_e

                # Get binned data as a function of score
                binned_data = self.get_binned_data(
                    sel_g, p, e, nbins=2000, score_range=clf_output_bounds
                )

                # Get re-binned data as a function of score for diagnostic plots
                re_binned_data = self.get_binned_data(
                    sel_g, p, e, nbins=200, score_range=clf_output_bounds
                )

                # Get optimisation results
                results_th_cut_dict[CutsOptimisation._get_angular_key(th_cut.value)] = {
                    'th_cut': th_cut,
                    'result': self.find_best_cutoff_for_one_bin(binned_data=binned_data),
                    'diagnostic_data': re_binned_data
                }

            # Select best theta cut (lowest flux). In case of equality, select the
            # one with the highest sig efficiency (flux are sorted as a function of
            # decreasing signal efficiencies)
            flux_list = []
            eff_sig = []
            th = []
            key_list = []
            for key in results_th_cut_dict:
                key_list.append(key)
                flux_list.append((results_th_cut_dict[key]['result']['min_flux']))
                eff_sig.append((results_th_cut_dict[key]['result']['eff_sig']))
                th.append(results_th_cut_dict[key]['th_cut'])

            # In case of equal min fluxes, take the one with bigger sig efficiency
            lower_flux_idx = np.where(
                np.array(flux_list) == np.array(flux_list).min()
            )[0][0]

            if self.verbose_level > 0:
                print(
                    'Select th={:.3f}, cutoff={:.3f} (eff_sig={:.3f}, eff_bkg={:.3f}, flux={:.3f}, syst={})'.format(
                        results_th_cut_dict[key_list[lower_flux_idx]]['th_cut'],
                        results_th_cut_dict[key_list[lower_flux_idx]]['result']['best_cutoff'],
                        results_th_cut_dict[key_list[lower_flux_idx]]['result']['eff_sig'],
                        results_th_cut_dict[key_list[lower_flux_idx]]['result']['eff_bkg'],
                        results_th_cut_dict[key_list[lower_flux_idx]]['result']['min_flux'],
                        results_th_cut_dict[key_list[lower_flux_idx]]['result']['systematic'])
                )

            key = CutsOptimisation._get_energy_key(emin.value, emax.value)
            self.results_dict[key] = {
                'emin': emin.value,
                'emax': emax.value,
                'obs_time': self.config['analysis']['obs_time']['value'],
                'th_cut': results_th_cut_dict[key_list[lower_flux_idx]]['th_cut'].value,
                'keep': True,
                'results': results_th_cut_dict[key_list[lower_flux_idx]]['result'],
                'diagnostic_data': results_th_cut_dict[key_list[lower_flux_idx]]['diagnostic_data']}

            print('     Ang. cut: {:.2f}, score cut: {}'.format(
                self.results_dict[key]['th_cut'],
                self.results_dict[key]['results']['best_cutoff']
            ))

    def find_best_cutoff_for_one_bin(self, binned_data):
        """
        Find the best cut off for one bin os the phase space
        """
        alpha = self.config['analysis']['alpha']

        # Scan eff_bkg efficiency (going from 0.05 to 0.5, 10 bins as in MARS analysis)
        fixed_bkg_eff = np.linspace(0.05, 0.5, 15)

        # Find corresponding indexes
        fixed_bkg_eff_indexes = np.zeros(len(fixed_bkg_eff), dtype=int)
        for idx in range(len(fixed_bkg_eff)):
            the_idx = (
                np.abs(binned_data['hist_eff_bkg'] - fixed_bkg_eff[idx])
            ).argmin()
            fixed_bkg_eff_indexes[idx] = the_idx

        # Will contain
        minimal_fluxes = np.zeros(len(fixed_bkg_eff))
        minimal_sigma = np.zeros(len(fixed_bkg_eff))
        minimal_syst = np.zeros(len(fixed_bkg_eff), dtype=bool)
        minimal_excess = np.zeros(len(fixed_bkg_eff))

        for iflux in range(len(minimal_fluxes)):

            excess = binned_data['cumul_excess'][fixed_bkg_eff_indexes][iflux]
            n_bkg = binned_data['cumul_noff'][fixed_bkg_eff_indexes][iflux] * alpha
            effsig = binned_data['hist_eff_sig'][fixed_bkg_eff_indexes][iflux]
            effbkg = binned_data['hist_eff_bkg'][fixed_bkg_eff_indexes][iflux]
            score = binned_data['score'][fixed_bkg_eff_indexes][iflux]
            minimal_syst[iflux] = False

            if n_bkg == 0:
                if self.verbose_level > 0:
                    print(
                        'Warning> To be dealt with')
                pass

            minimal_fluxes[iflux], minimal_sigma[iflux] = self._get_sigma_flux(
                excess, n_bkg, alpha, self.config['analysis']['min_sigma']
            )
            minimal_excess[iflux] = minimal_fluxes[iflux] * excess

            if self.verbose_level > 1:
                print('eff_bkg={:.2f}, eff_sig={:.2f}, score={:.2f}, excess={:.2f}, bkg={:.2f}, min_flux={:.3f}, sigma={:.3f}'.format(
                    effbkg, effsig, score, minimal_excess[iflux], n_bkg,
                    minimal_fluxes[iflux], minimal_sigma[iflux])
                )

            if minimal_excess[iflux] < self.config['analysis']['min_excess']:
                minimal_syst[iflux] = True
                # Rescale flux accodring to minimal acceptable excess
                minimal_fluxes[iflux] = self.config['analysis']['min_excess'] / excess
                minimal_excess[iflux] = self.config['analysis']['min_excess']
                if self.verbose_level > 1:
                    print(' WARNING> Not enough signal!')

            if minimal_excess[iflux] < self.config['analysis']['bkg_syst'] * n_bkg:
                minimal_syst[iflux] = True
                minimal_fluxes[iflux] = self.config['analysis']['bkg_syst'] * n_bkg / excess
                if self.verbose_level > 1:
                    print(' WARNING> Bkg systematics!')

        # In case of equal min fluxes, take the one with bigger sig efficiency
        # (last value)
        opti_cut_index = np.where(minimal_fluxes == minimal_fluxes.min())[0][-1]
        min_flux = minimal_fluxes[opti_cut_index]
        min_sigma = minimal_sigma[opti_cut_index]
        min_excess = minimal_excess[opti_cut_index]
        min_syst = minimal_syst[opti_cut_index]

        best_cut_index = fixed_bkg_eff_indexes[opti_cut_index]  # for fine binning

        return {'best_cutoff': binned_data['score'][best_cut_index],
                'noff': binned_data['cumul_noff'][best_cut_index],
                'background': binned_data['cumul_noff'][best_cut_index] * alpha,
                'non': binned_data['cumul_excess'][best_cut_index] * min_flux + binned_data['cumul_noff'][best_cut_index] * alpha,
                'alpha': alpha,
                'eff_sig': binned_data['hist_eff_sig'][best_cut_index],
                'eff_bkg': binned_data['hist_eff_bkg'][best_cut_index],
                'min_flux': min_flux,
                'excess': min_excess,
                'sigma': min_sigma,
                'systematic': min_syst,
                }

    @classmethod
    def _get_sigma_flux(cls, excess, bkg, alpha, min_sigma):
        """Compute flux to get `min_sigma` sigma detection. Returns fraction
        of minimal flux and the resulting signifiance"""


        # Gross binning
        flux_level = np.arange(0., 10, 0.01)[1:]
        # sigma = significance_on_off(n_on=excess * flux_level + bkg,
        #                             n_off=bkg / alpha,
        #                             alpha=alpha,
        #                             method='lima')
        sigma = WStatCountsStatistic(n_on=excess * flux_level + bkg,
                                    n_off=bkg / alpha,
                                    alpha=alpha).significance


        the_idx = (np.abs(sigma - min_sigma)).argmin()
        min_flux = flux_level[the_idx]

        # Fine binning
        flux_level = np.arange(min_flux - 0.05, min_flux + 0.05, 0.001)
        # sigma = significance_on_off(n_on=excess * flux_level + bkg,
        #                             n_off=bkg / alpha,
        #                             alpha=alpha,
        #                             method='lima')
        sigma = WStatCountsStatistic(n_on=excess * flux_level + bkg,
                                    n_off=bkg / alpha,
                                    alpha=alpha).significance

        the_idx = (np.abs(sigma - min_sigma)).argmin()

        return flux_level[the_idx], sigma[the_idx]


    @classmethod
    def _get_energy_key(cls, emin, emax):
        return '{:.3f}-{:.3f}TeV'.format(emin, emax)

    @classmethod
    def _get_angular_key(cls, ang):
        return '{:.3f}deg'.format(ang)

    def get_binned_data(self, g, p, e, nbins=100, score_range=[-1,1]):
        """Returns binned data as a dictionnary"""
        colname_clf_output = self.config['column_definition']['classification_output']['name']

        res = dict()
        # Histogram of events
        res['hist_sig'], edges = np.histogram(
            a=g[colname_clf_output].values, bins=nbins, range=score_range, weights=g['weight_corrected'].values
        )
        res['hist_p'], edges = np.histogram(
            a=p[colname_clf_output].values, bins=nbins, range=score_range, weights=p['weight_corrected'].values
        )
        res['hist_e'], edges = np.histogram(
            a=e[colname_clf_output].values, bins=nbins, range=score_range, weights=e['weight_corrected'].values
        )
        res['hist_bkg'] = res['hist_p'] + res['hist_e']
        res['score'] = (edges[:-1] + edges[1:]) / 2.
        res['score_edges'] = edges

        # Efficiencies
        res['hist_eff_sig'] = 1. - np.cumsum(res['hist_sig']) / np.sum(res['hist_sig'])
        res['hist_eff_bkg'] = 1. - np.cumsum(res['hist_bkg']) / np.sum(res['hist_bkg'])

        # Cumulative statistics
        alpha = self.config['analysis']['alpha']
        res['cumul_noff'] = res['hist_eff_bkg'] * sum(res['hist_bkg']) / alpha
        res['cumul_excess'] = sum(res['hist_sig']) - np.cumsum(res['hist_sig'])
        res['cumul_non'] = res['cumul_excess'] + res['cumul_noff'] * alpha
        # res['cumul_sigma'] = significance_on_off(
        #     n_on=res['cumul_non'], n_off=res['cumul_noff'], alpha=alpha, method='lima'
        # )
        res['cumul_sigma'] = WStatCountsStatistic(
            n_on=res['cumul_non'], n_off=res['cumul_noff'], alpha=alpha
        ).significance

        return res

    def write_results(self, outdir, outfile, format, overwrite=True):
        """Write results with astropy utilities"""
        # Declare and initialise vectors to save
        n = len(self.results_dict)
        feature_to_save = [('best_cutoff', float), ('non', float), ('noff', float),
                           ('alpha', float), ('background', float), ('excess', float),
                           ('eff_sig', float), ('eff_bkg', float), ('systematic', bool),
                           ('min_flux', float), ('sigma', float)]
        emin = np.zeros(n)
        emax = np.zeros(n)
        angular_cut = np.zeros(n)
        obs_time = np.zeros(n)
        keep = np.zeros(n, dtype=bool)

        res_to_save = dict()
        for feature in feature_to_save:
            res_to_save[feature[0]] = np.zeros(n, dtype=feature[1])

        # Fill data and save diagnostic result
        for idx, key in enumerate(self.results_dict.keys()):
            bin_info = self.results_dict[key]
            if bin_info['keep'] == False:
                keep[idx] = bin_info['keep']
                continue
            bin_results = self.results_dict[key]['results']
            bin_data = self.results_dict[key]['diagnostic_data']

            keep[idx] = bin_info['keep']
            emin[idx] = bin_info['emin']
            emax[idx] = bin_info['emax']
            angular_cut[idx] = bin_info['th_cut']
            obs_time[idx] = bin_info['obs_time']
            for feature in feature_to_save:
                res_to_save[feature[0]][idx] = bin_results[feature[0]]

            obj_name = 'diagnostic_data_emin{:.3f}_emax{:.3f}.pkl.gz'.format(
                bin_info['emin'], bin_info['emax']
            )

            diagnostic_dir = os.path.join(outdir, 'diagnostic')
            if not os.path.exists(diagnostic_dir):
                os.makedirs(diagnostic_dir)
            save_obj(bin_data, os.path.join(outdir, 'diagnostic', obj_name))

        # Save data
        t = Table()
        t['keep'] = Column(keep, dtype=bool)
        t['emin'] = Column(emin, unit='TeV')
        t['emax'] = Column(emax, unit='TeV')
        t['obs_time'] = Column(obs_time, unit=self.config['analysis']['obs_time']['unit'])
        t['angular_cut'] = Column(angular_cut, unit='TeV')
        for feature in feature_to_save:
            t[feature[0]] = Column(res_to_save[feature[0]])
        t.write(os.path.join(outdir, outfile), format=format, overwrite=overwrite)


