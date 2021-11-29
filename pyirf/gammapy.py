try:
    import gammapy
except ImportError:
    raise ImportError('You need gammapy installed to use this module of pyirf') from None

from gammapy.irf import EffectiveAreaTable2D, PSF3D, EnergyDispersion2D
from gammapy.maps import MapAxis
import astropy.units as u



def _create_offset_axis(fov_offset_bins):
    return MapAxis.from_edges(fov_offset_bins, name="offset")

def _create_energy_axis_true(true_energy_bins):
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
    Create a :py:class:`gammapy.irf.EffectiveAreaTable2D` from pyirf outputs.

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
    aeff2d: gammapy.irf.EffectiveAreaTable2D
    '''
    offset_axis = _create_offset_axis(fov_offset_bins)
    energy_axis_true = _create_energy_axis_true(true_energy_bins)

    return EffectiveAreaTable2D(
        axes = [energy_axis_true,
                offset_axis],
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
    Create a :py:class:`gammapy.irf.PSF3D` from pyirf outputs.

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

    Returns
    -------
    psf: gammapy.irf.PSF3D
    """
    offset_axis = _create_offset_axis(fov_offset_bins)
    energy_axis_true = _create_energy_axis_true(true_energy_bins)
    rad_axis = MapAxis.from_edges(source_offset_bins, name='rad')

    return PSF3D(
        axes = [energy_axis_true,
                offset_axis,
                rad_axis],
        data = psf
    )


@u.quantity_input(
    true_energy_bins=u.TeV, fov_offset_bins=u.deg,
)
def create_energy_dispersion_2d(
    energy_dispersion,
    true_energy_bins,
    migration_bins,
    fov_offset_bins,
):
    """
    Create a :py:class:`gammapy.irf.EnergyDispersion2D` from pyirf outputs.

    Parameters
    ----------
    energy_dispersion: numpy.ndarray
        Energy dispersion array, must have shape
        (n_energy_bins, n_migra_bins, n_source_offset_bins)
    true_energy_bins: astropy.units.Quantity[energy]
        Bin edges in true energy
    migration_bins: numpy.ndarray
        Bin edges for the relative energy migration (``reco_energy / true_energy``)
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
        For Point-Like IRFs, only giving a single bin is appropriate.

    Returns
    -------
    edisp: gammapy.irf.EnergyDispersion2D
    """
    offset_axis = _create_offset_axis(fov_offset_bins)
    energy_axis_true = _create_energy_axis_true(true_energy_bins)
    migra_axis = MapAxis.from_edges(migration_bins, name="migra")

    return EnergyDispersion2D(
        axes = [energy_axis_true,
                migra_axis,
                offset_axis],
        data = energy_dispersion,
    )
