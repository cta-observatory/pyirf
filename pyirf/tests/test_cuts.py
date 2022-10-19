import operator
import numpy as np
from astropy.table import QTable
import astropy.units as u
import pytest
from scipy.stats import norm


@pytest.fixture
def events():
    return QTable(
        {
            "bin_reco_energy": [0, 0, 1, 1, 2, 2],
            "theta": [0.1, 0.02, 0.3, 0.15, 0.01, 0.1] * u.deg,
            "gh_score": [1.0, -0.2, 0.5, 0.05, 1.0, 0.3],
        }
    )


def test_calculate_percentile_cuts():
    from pyirf.cuts import calculate_percentile_cut

    np.random.seed(0)

    dist1 = norm(0, 1)
    dist2 = norm(10, 1)
    N = int(1e4)

    values = np.append(dist1.rvs(size=N), dist2.rvs(size=N)) * u.deg
    bin_values = np.append(np.zeros(N), np.ones(N)) * u.m
    # add some values outside of binning to test that under/overflow are ignored
    bin_values[10] = 5 * u.m
    bin_values[30] = -1 * u.m

    bins = [-0.5, 0.5, 1.5] * u.m

    cuts = calculate_percentile_cut(values, bin_values, bins, fill_value=np.nan * u.deg)
    assert np.all(cuts["low"] == bins[:-1])
    assert np.all(cuts["high"] == bins[1:])

    assert np.allclose(
        cuts["cut"].to_value(u.deg),
        [dist1.ppf(0.68), dist2.ppf(0.68)],
        rtol=0.1,
    )

    # test with min/max value
    cuts = calculate_percentile_cut(
        values,
        bin_values,
        bins,
        fill_value=np.nan * u.deg,
        min_value=1 * u.deg,
        max_value=5 * u.deg,
    )
    assert np.all(cuts["cut"] == [1.0, 5.0] * u.deg)


def test_calculate_percentile_cuts_no_units():
    from pyirf.cuts import calculate_percentile_cut

    np.random.seed(0)

    dist1 = norm(0, 1)
    dist2 = norm(10, 1)
    N = int(1e4)

    values = np.append(dist1.rvs(size=N), dist2.rvs(size=N))
    bin_values = np.append(np.zeros(N), np.ones(N)) * u.m
    bins = [-0.5, 0.5, 1.5] * u.m

    cuts = calculate_percentile_cut(values, bin_values, bins, fill_value=np.nan)
    assert np.all(cuts["low"] == bins[:-1])
    assert np.all(cuts["high"] == bins[1:])

    assert np.allclose(
        cuts["cut"],
        [dist1.ppf(0.68), dist2.ppf(0.68)],
        rtol=0.1,
    )


def test_calculate_percentile_cuts_smoothing():
    from pyirf.cuts import calculate_percentile_cut

    np.random.seed(0)

    dist1 = norm(0, 1)
    dist2 = norm(10, 1)
    N = int(1e4)

    values = np.append(dist1.rvs(size=N), dist2.rvs(size=N))
    bin_values = np.append(np.zeros(N), np.ones(N)) * u.m
    bins = [-0.5, 0.5, 1.5] * u.m

    cuts = calculate_percentile_cut(values, bin_values, bins, fill_value=np.nan, smoothing=1)
    assert np.all(cuts["low"] == bins[:-1])
    assert np.all(cuts["high"] == bins[1:])

    assert np.allclose(
        cuts["cut"],
        [3.5, 7.5],
        rtol=0.2,
    )


def test_evaluate_binned_cut():
    from pyirf.cuts import evaluate_binned_cut

    cuts = QTable({"low": [0, 1], "high": [1, 2], "cut": [100, 1000],})

    survived = evaluate_binned_cut(
        np.array([500, 1500, 50, 2000, 25, 800]),
        np.array([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        cut_table=cuts,
        op=operator.ge,
    )
    assert np.all(survived == [True, True, False, True, False, False])

    # test with quantity
    cuts = QTable(
        {"low": [0, 1] * u.TeV, "high": [1, 2] * u.TeV, "cut": [100, 1000] * u.m,}
    )

    survived = evaluate_binned_cut(
        [500, 1500, 50, 2000, 25, 800] * u.m,
        [0.5, 1.5, 0.5, 1.5, 0.5, 1.5] * u.TeV,
        cut_table=cuts,
        op=operator.ge,
    )
    assert np.all(survived == [True, True, False, True, False, False])


def test_compare_irf_cuts():
    """Tests compare_irf_cuts."""

    from pyirf.cuts import compare_irf_cuts
    # first create some dummy cuts
    enbins = np.logspace(-2, 3) * u.TeV
    thcuts1 = np.linspace(0.5, 0.1) * u.deg
    thcuts2 = np.linspace(0.6, 0.2) * u.deg
    names = ("Energy", "Theta2")
    t1 = QTable([enbins, thcuts1], names=names)
    t1b = QTable([enbins, thcuts1], names=names)
    t2 = QTable([enbins, thcuts2], names=names)
    # comparing identical cuts ==> should return True
    assert compare_irf_cuts([t1, t1b])

    # different cuts ==> should return False
    assert compare_irf_cuts([t1, t2]) is False


def test_calculate_percentile_cuts_table():
    '''Test that calculate percentile cuts does not modify input table'''
    from pyirf.cuts import calculate_percentile_cut

    np.random.seed(0)

    dist1 = norm(0, 1)
    dist2 = norm(10, 1)
    N = int(1e4)

    table = QTable({
        "foo": np.append(dist1.rvs(size=N), dist2.rvs(size=N)) * u.deg,
        "bar": np.append(np.zeros(N), np.ones(N)) * u.m,
    })

    bins = [-0.5, 0.5, 1.5] * u.m
    cuts = calculate_percentile_cut(
        table["foo"],
        table["bar"],
        bins,
        fill_value=np.nan * u.rad,
    )
    assert table.colnames == ["foo", "bar"]
    assert np.allclose(
        cuts["cut"].to_value(u.deg),
        [dist1.ppf(0.68), dist2.ppf(0.68)],
        rtol=0.1,
    )
