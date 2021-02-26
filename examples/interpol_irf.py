"""
Example macro for reading in IRFs and performing interpolation
The interpolation is done over a number of parameters that are specified
in json config file, and the actual value to interpolate to is read from
a data file
"""
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import json
import pyirf.interpolation as interp
import astropy.units as u
from pyirf.io import create_aeff2d_hdu
plt.ion()

aeff_name = 'EFFECTIVE AREA'

# settings
# data_file = 'lstdata/dl2_LST-1.Run03635.0001.h5'
data_file = 'lstdata/dl2_LST-1.Run03642.0110.h5'
config_file = './interpol_irf.json'
min_aeff = 1.  # to avoid zeros in log
interp_method = 'linear'

output_file = 'irf_interp.fits'

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

print(interp_names)
interp_dim = len(interp_names)

print(files)
n_files = len(files)

aeff_all, pars_all, energy_bins, theta_bins = interp.read_irf_grid(files, aeff_name, 'EFFAREA')

en = 0.5 * (energy_bins[1:] + energy_bins[:-1])
n_theta = len(theta_bins) - 1

fig0 = plt.figure(figsize=(12, 9))
axs = []


for i_th in range(n_theta):
    axs.append(plt.subplot(2, 3, 1 + i_th))
    title = r'$A_{eff}$, $\theta$=' + f'{theta_bins[i_th]}-{theta_bins[i_th+1]}' + r'$^{\circ}$'
    axs[i_th].set_title(title)

    for i_file in range(n_files):
        legentry = ""
        for i in range(interp_dim):
            legentry += f'{interp_names[i]}={pars_all[i_file,i]:.2f} '
        aeff = aeff_all[i_file][i_th]
        axs[i_th].loglog(en, aeff, label=legentry)

# pars_all=(np.array(to_process)[:,1:]).tolist()  #alternative way of doing it

leg_interp = "Interpol. "
for i in range(interp_dim):
    leg_interp += f'{interp_names[i]}={interp_pars[i]:.2f} '

aeff_interp = interp.interpolate_effective_area(aeff_all, pars_all, interp_pars, method=interp_method)

for i_th in range(n_theta):
    axs[i_th].loglog(en, aeff_interp[:, i_th], label=leg_interp, linewidth=3)

for iplot in range(n_theta):
    axs[iplot].legend()
plt.tight_layout()


# now write an output fits file

hdus = [
    fits.PrimaryHDU(),
    create_aeff2d_hdu(
        # effective_area[..., np.newaxis],
        aeff_interp,
        energy_bins,
        theta_bins,
        extname="EFFECTIVE AREA")
]
fits.HDUList(hdus).writeto(output_file, overwrite=True)
