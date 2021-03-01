"""
Example macro for reading in IRFs and performing interpolation
The interpolation is done over a number of parameters that are specified
in json config file, and the actual value to interpolate to is read from
a data file

For this example you should download the prod3b CTA IRFs:
https://www.cta-observatory.org/wp-content/uploads/2019/04/CTA-Performance-prod3b-v2-FITS.tar.gz
"""
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import json
import pyirf.interpolation as interp
from pyirf.io import create_aeff2d_hdu, create_energy_dispersion_hdu
import time
import sys

plt.ion()

# settings
aeff_name = 'EFFECTIVE AREA'
edisp_name = 'ENERGY DISPERSION'

# we should check if the files were produced with consistent theta2 and g/h separation cuts
# however the test files over which this macro is run do not have them stored
check_cut_consistency = False

data_file = 'lstdata/dl2_LST-1.Run03642.0110.h5'
config_file = './interpol_irf.json'
min_aeff = 1.  # to avoid zeros in log
interp_method = 'linear'

draw_results = True
output_file = 'irf_interp.fits'

# load in a configuration file
with open(config_file) as pars_file:
    config = json.load(pars_file)
pars = config['interpol_irf']['pars']
files = config['interpol_irf']['files']

print('Interpolating over the following variables:')
for par in pars:
    print(par[0], '=', par[1])

print("opening: ", data_file)
interp_pars = interp.read_mean_pars_data(data_file, pars)

interp_names = np.array(pars)[:, 0].tolist()
interp_dim = len(interp_names)

n_files = len(files)

if check_cut_consistency:
    if interp.compare_irf_cuts(np.array(files)[:, 0], 'THETA_CUTS') is False:
        sys.exit("not compatible theta2 cuts")
    if interp.compare_irf_cuts(np.array(files)[:, 0], 'GH_CUTS') is False:
        sys.exit("not compatible theta2 cuts")

# effective area interpolation
aeff_all, pars_all, energy_bins, theta_bins = interp.read_irf_grid(files, aeff_name, 'EFFAREA')

en = 0.5 * (energy_bins[1:] + energy_bins[:-1])
n_theta = len(theta_bins) - 1

# pars_all=(np.array(to_process)[:,1:]).tolist()  #alternative way of doing it

aeff_interp = interp.interpolate_effective_area(aeff_all, pars_all, interp_pars, method=interp_method)


if draw_results:
    legentries = np.empty(n_files, dtype=object)
    for i_file in range(n_files):
        legentries[i_file] = ""
        for i in range(interp_dim):
            legentries[i_file] += f'{interp_names[i]}={pars_all[i_file,i]:.2f} '

    leg_interp = "Interpol. "
    for i in range(interp_dim):
        leg_interp += f'{interp_names[i]}={interp_pars[i]:.2f} '

    fig0 = plt.figure(figsize=(12, 9))
    axs = []

    for i_th in range(n_theta):
        axs.append(plt.subplot(2, 3, 1 + i_th))
        title = r'$A_{eff}$, $\theta$=' + f'{theta_bins[i_th]}-{theta_bins[i_th+1]}' + r'$^{\circ}$'
        axs[i_th].set_title(title)

        for i_file in range(n_files):
            aeff = aeff_all[i_file][i_th]
            axs[i_th].loglog(en, aeff, label=legentries[i_file])

        axs[i_th].loglog(en, aeff_interp[:, i_th], label=leg_interp, linewidth=3)
        axs[i_th].legend()

    plt.tight_layout()


# migration matrix
mig_all, _, energy_bins_mig, theta_bins_mig = interp.read_irf_grid(files, edisp_name, 'MATRIX')

if draw_results:
    fig2 = plt.figure(figsize=(18, 9))
    axs = []

    for i_file in range(n_files):
        axs.append(plt.subplot(2, 3, 1 + i_file))
        axs[i_file].set_title(legentries[i_file])
        plt.imshow(mig_all[i_file, 0, :, :], origin='lower')
    plt.tight_layout()

mig_bins = interp.read_fits_bins_lo_hi(files[0][0], edisp_name, 'MIGRA')
start = time.time()
mig_interp = interp.interpolate_dispersion_matrix(mig_all, pars_all, interp_pars, method=interp_method)
end = time.time()
print("migration matrix interpolation elapsed time=", end - start)

if draw_results:
    plt.figure(3)
    plt.imshow(mig_interp[:, :, 0].T, origin='lower')

# now write an output fits file
hdus = [fits.PrimaryHDU()]
hdus.append(create_aeff2d_hdu(aeff_interp, energy_bins, theta_bins, extname=aeff_name))
hdus.append(create_energy_dispersion_hdu(mig_interp, energy_bins_mig, mig_bins, theta_bins_mig, extname=edisp_name))
fits.HDUList(hdus).writeto(output_file, overwrite=True)
