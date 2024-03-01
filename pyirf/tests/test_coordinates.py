from numpy.testing import assert_allclose
import astropy.units as u

def test_fov_coords_lon_lat():
    from pyirf.coordinates import fov_coords_lon_lat
    # test some simple cases
    lon, lat = fov_coords_lon_lat(1 * u.deg, 1 * u.deg, 0 * u.deg, 0 * u.deg)
    assert_allclose(lon.value, -1)
    assert_allclose(lat.value, 1)

    lon, lat = fov_coords_lon_lat(269 * u.deg, 0 * u.deg, 270 * u.deg, 0 * u.deg)
    assert_allclose(lon.value, 1)
    assert_allclose(lat.value, 0, atol=1e-7)

    lon, lat = fov_coords_lon_lat(1 * u.deg, 60 * u.deg, 0 * u.deg, 60 * u.deg)
    assert_allclose(lon.value, -0.5, rtol=1e-3)
    assert_allclose(lat.value, 0.003779, rtol=1e-3)

    # these are cross-checked with the
    # transformation as implemented in H.E.S.S.
    az = [51.320575, 50.899125, 52.154053, 48.233023]
    alt = [49.505451, 50.030165, 51.811739, 54.700102]
    az_pointing = [52.42056255, 52.24706061, 52.06655505, 51.86795724]
    alt_pointing = [51.11908203, 51.23454751, 51.35376141, 51.48385814]
    lon, lat = fov_coords_lon_lat(
        az * u.deg, alt * u.deg, az_pointing * u.deg, alt_pointing * u.deg
    )
    assert_allclose(
        lon.value, [0.7145614, 0.86603433, -0.05409698, 2.10295248], rtol=1e-5
    )
    assert_allclose(
        lat.value, [-1.60829115, -1.19643974, 0.45800984, 3.26844192], rtol=1e-5
    )

def test_fov_coords_theta_phi():
    from pyirf.coordinates import fov_coords_theta_phi

    theta, phi = fov_coords_theta_phi(
        alt=1 * u.deg, az=0 * u.deg, pointing_alt=0 * u.deg, pointing_az=0 * u.deg
    )
    assert u.isclose(theta, 1 * u.deg)
    assert u.isclose(phi, 0 * u.deg)
    
    theta, phi = fov_coords_theta_phi(
        alt=-1 * u.deg, az=0 * u.deg, pointing_alt=0 * u.deg, pointing_az=0 * u.deg
    )
    assert u.isclose(theta, 1 * u.deg)
    assert u.isclose(phi, 180 * u.deg)
    
    theta, phi = fov_coords_theta_phi(
        alt=0 * u.deg, az=-1 * u.deg, pointing_alt=0 * u.deg, pointing_az=0 * u.deg
    )
    assert u.isclose(theta, 1 * u.deg)
    assert u.isclose(phi, 90 * u.deg)
    
    theta, phi = fov_coords_theta_phi(
        alt=0 * u.deg, az=1 * u.deg, pointing_alt=0 * u.deg, pointing_az=0 * u.deg
    )
    assert u.isclose(theta, 1 * u.deg)
    assert u.isclose(phi, 270 * u.deg)