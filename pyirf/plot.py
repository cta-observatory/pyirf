import matplotlib.pyplot as plt
from astropy.io import fits


def plot_angular_resolution(irf_filename, ax=None, **kwargs):
    """
    Plot angular resolution from an IRF file
    Parameters
    ----------
    irf_filename
    ax
    kwargs
    Returns
    -------
    """

    ax = plt.gca() if ax is None else ax

    with fits.open(irf_filename) as irf:
        psf_hdu = irf['ANGULAR_RESOLUTION']
        energy = psf_hdu.data['true_energy_center']
        energy_lo = psf_hdu.data['true_energy_low']
        energy_hi = psf_hdu.data['true_energy_high']
        psf = psf_hdu.data['angular_resolution']

    kwargs.setdefault('fmt', 'o')
    kwargs.setdefault('ms', 3)

    ax.errorbar(energy, psf,
                xerr=((energy - energy_lo), (energy_hi - energy)),
                **kwargs,
                )
    ax.set_xscale('log')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.grid(which='both')
    return ax