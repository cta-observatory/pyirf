"""
Example macro for reading in IRFs and performing interpolation
The interpolation is done over a number of parameters that are specified
in json config file, and the actual value to interpolate to is read from
a data file
"""
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.interpolate import griddata
plt.ion()

# definition from lstchain.io.io copied here to avoid dependency
dl2_params_lstcam_key = 'dl2/event/telescope/parameters/LST_LSTCam'
aeff_name = 'EFFECTIVE AREA'

# settings
data_file = 'lstdata/dl2_LST-1.Run03635.0001.h5'
config_file = './interpol_irf.json'
min_aeff = 1.  # to avoid zeros in log
interp_method = 'linear'


with open(config_file) as pars_file:
    config = json.load(pars_file)
pars = config['interpol_irf']['pars']
files = config['interpol_irf']['files']

print('Interpolating over the following variables:')
for par in pars:
    print(par[0], '=', par[1])

print("opening: ", data_file)
# read in the data
data = pd.read_hdf(data_file, key=dl2_params_lstcam_key)
interp_pos = []  # position for which to interpolate
for par in pars:
    val = np.mean(data.eval(par[1]))
    print(par[0], val)
    interp_pos.append(val)
interp_pos = tuple(interp_pos)

interp_names = np.array(pars)[:, 0].tolist()

print(interp_names)
interp_dim = len(interp_names)

to_process = files
n_process = len(to_process)

# open the first file and check some settings
hdul = fits.open(to_process[0][0])
hdul.info()
aeff_tab = hdul[aeff_name].data[0]
en = np.sqrt(aeff_tab['ENERG_LO'] * aeff_tab['ENERG_HI'])
en[en == 0] = 0.5 * aeff_tab['ENERG_HI'][en == 0]  # fix for the first bin that starts at E=0
th_lows = aeff_tab['THETA_LO']
th_his = aeff_tab['THETA_HI']
n_theta = len(th_lows)
n_en = len(en)

fig0 = plt.figure(figsize=(12, 9))
axs = []
for iplot in range(n_theta):
    axs.append(plt.subplot(2, 3, 1 + iplot))
    title = r'$A_{eff}$, $\theta$=' + f'{th_lows[iplot]}-{th_his[iplot]}' + r'$^{\circ}$'
    axs[iplot].set_title(title)

log_aeff_all = np.empty((n_process, n_theta, n_en))
pars_all = np.empty((n_process, interp_dim))
# pars_all=(np.array(to_process)[:,1:]).tolist()  #alternative way of doing it

for iprocess, process_it in enumerate(to_process):
    infile = process_it[0]
    pars = process_it[1:]
    pars_all[iprocess, :] = pars

    legentry = ""
    for i in range(interp_dim):
        legentry += f'{interp_names[i]}={pars[i]:.2f} '
    hdul = fits.open(infile)

    for i_th in range(n_theta):
        aeff_tab = hdul[aeff_name].data[0]
        aeff = aeff_tab['EFFAREA'][i_th]
        log_aeff_all[iprocess, i_th, :] = aeff
        axs[i_th].loglog(en, aeff, label=legentry)

# remove zeros and log it
log_aeff_all[log_aeff_all < min_aeff] = min_aeff
log_aeff_all = np.log(log_aeff_all)

# interpolation
leg_interp = "Interpol. "
for i in range(interp_dim):
    leg_interp += f'{interp_names[i]}={interp_pos[i]:.2f} '

aeff_interp = np.empty((n_theta, n_en))
for i_th in range(n_theta):
    for i_en in range(n_en):
        aeff_interp[i_th, i_en] = griddata(pars_all, log_aeff_all[:, i_th, i_en], interp_pos, method='linear')

# exp it and set to zero too low values
aeff_interp = np.exp(aeff_interp)
aeff_interp[aeff_interp < min_aeff * 1.1] = 0  # 1.1 to correct for numerical uncertainty and interpolation

for i_th in range(n_theta):
    axs[i_th].loglog(en, aeff_interp[i_th, :], label=leg_interp, linewidth=3)

for iplot in range(n_theta):
    axs[iplot].legend()
plt.tight_layout()
