try:
    import gammapy
except ImportError:
    raise ImportError('You need gammapy installed to use this module of pyirf') from None

from gammapy.irf import EffectiveAreaTable2D, PSF3D
from gammapy.maps import MapAxis
import astropy.units as u



def create_offset_axis(fov_offset_bins):
    return MapAxis.from_edges(fov_offset_bins, name="offset")

def create_energy_axis_true(true_energy_bins):
    return MapAxis.from_edges(true_energy_bins, name="energy_true")


@u.quantity_input(
    effective_area=u.m ** 2, true_energy_bins=u.TeV, fov_offset_bins=u.deg
)
def create_effective_area_table_2d(
    effective_area,
    true_energy_bins,
    fov_offset_bins,
):
    '''
    Create a ``gammapy.irf.EffectiveAreaTable2D`` from pyirf outputs.

    Parameters
    ----------
    effective_area: astropy.units.Quantity[area]
        Effective area array, must have shape (n_energy_bins, n_fov_offset_bins)
    true_energy_bins: astropy.units.Quantity[energy]
        Bin edges in true energy
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
        For Point-Like IRFs, only giving a single bin is appropriate.

    Returns
    -------
    gammapy.irf.EffectiveAreaTable2D

    '''

    offset_axis = create_offset_axis(fov_offset_bins)
    energy_axis_true = create_energy_axis_true(true_energy_bins)

    return EffectiveAreaTable2D(
        energy_axis_true=energy_axis_true,
        offset_axis=offset_axis,
        data=effective_area,
    )



@u.quantity_input(
    psf=u.sr ** -1,
    true_energy_bins=u.TeV,
    source_offset_bins=u.deg,
    fov_offset_bins=u.deg,
)
def create_psf_3d(
    psf,
    true_energy_bins,
    source_offset_bins,
    fov_offset_bins,
):
    """
    Create a ``gammapy.irf.PSF3D`` from pyirf outputs.

    Parameters
    ----------
    psf: astropy.units.Quantity[(solid angle)^-1]
        Point spread function array, must have shape
        (n_energy_bins, n_fov_offset_bins, n_source_offset_bins)
    true_energy_bins: astropy.units.Quantity[energy]
        Bin edges in true energy
    source_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the source offset.
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
        For Point-Like IRFs, only giving a single bin is appropriate.
    """
    offset_axis = create_offset_axis(fov_offset_bins)
    energy_axis_true = create_energy_axis_true(true_energy_bins)
    rad_axis = MapAxis.from_edges(source_offset_bins, name='rad')

    return PSF3D(
        energy_axis_true=energy_axis_true,
        offset_axis=offset_axis,
        rad_axis=rad_axis,
        psf_value=psf,
    )
