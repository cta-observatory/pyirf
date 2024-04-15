import astropy.units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame, angular_separation, position_angle

__all__ = [
    'fov_coords_lon_lat',
    'fov_coords_theta_phi'
]


def gadf_fov_coords_lon_lat(lon, lat, pointing_lon, pointing_lat):
    """Transform sky coordinates to field-of-view longitude-latitude coordinates in accordance with
    the definitions laid out by the Gamma Astro Data Format.

    GADF documentation here:
    https://gamma-astro-data-formats.readthedocs.io/en/latest/general/coordinates.html

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity`
        Sky coordinate to be transformed.
    pointing_lon, pointing_lat : `~astropy.units.Quantity`
        Coordinate specifying the pointing position.
        (i.e. the center of the field of view.)

    Returns
    -------
    lon_t, lat_t : `~astropy.units.Quantity`
        Transformed field-of-view coordinate.
    """
    # Create a frame that is centered on the pointing position
    center = SkyCoord(pointing_lon, pointing_lat)
    fov_frame = SkyOffsetFrame(origin=center)

    # Define coordinate to be transformed.
    target_sky = SkyCoord(lon, lat)

    # Transform into FoV-system
    target_fov = target_sky.transform_to(fov_frame)

    # Switch sign of longitude angle since this axis is
    # reversed in our definition of the FoV-system
    return -target_fov.lon, target_fov.lat


def gadf_fov_coords_theta_phi(az, alt, pointing_az, pointing_alt):
    """Transform sky coordinates to field-of-view theta-phi coordinates in accordance with
    the definitions laid out by the Gamma Astro Data Format.

    GADF documentation here:
    https://gamma-astro-data-formats.readthedocs.io/en/latest/general/coordinates.html

    Parameters
    ----------
    az, alt : `~astropy.units.Quantity`
        Sky coordinate to be transformed.
    pointing_az, pointing_alt : `~astropy.units.Quantity`
        Coordinate specifying the pointing position.
        (i.e. the center of the field of view.)

    Returns
    -------
    theta, phi : `~astropy.units.Quantity`
        Transformed field-of-view coordinate.
    """

    theta = angular_separation(pointing_az, pointing_alt, az, alt)

    # astropy defines the position angle as increasing towards east of north
    phi = position_angle(pointing_az, pointing_alt, az, alt)

    # GADF defines FOV PHI opposite to the position angle way so the sign is switched
    return theta.to(u.deg), (-phi).wrap_at(360 * u.deg).to(u.deg)