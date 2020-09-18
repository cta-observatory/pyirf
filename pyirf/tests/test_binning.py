import astropy.units as u
import numpy as np


def test_add_overflow_bins():
    from pyirf.binning import add_overflow_bins

    bins = np.array([1, 2])
    bins_uo = add_overflow_bins(bins)

    assert len(bins_uo) == 4
    assert bins_uo[0] == 0
    assert np.isinf(bins_uo[-1])
    assert np.all(bins_uo[1:-1] == bins)

    bins = np.array([1, 2])
    bins_uo = add_overflow_bins(bins, positive=False)

    assert bins_uo[0] < 0
    assert np.isinf(bins_uo[0])

    # make sure we don't any bins if over / under is already present
    bins = np.array([0, 1, 2, np.inf])
    assert len(add_overflow_bins(bins)) == len(bins)


def test_add_overflow_bins_units():
    from pyirf.binning import add_overflow_bins

    bins = np.array([1, 2]) * u.TeV
    bins_uo = add_overflow_bins(bins)

    assert bins_uo.unit == bins.unit
    assert len(bins_uo) == 4
    assert bins_uo[0] == 0 * u.TeV
    assert np.isinf(bins_uo[-1])
    assert np.all(bins_uo[1:-1] == bins)


def test_bins_per_decade():
    from pyirf.binning import create_bins_per_decade

    bins = create_bins_per_decade(100 * u.GeV, 100 * u.TeV)

    assert bins.unit == u.GeV
    assert len(bins) == 15  # end non-inclusive

    assert bins[0] == 100 * u.GeV
    assert np.allclose(np.diff(np.log10(bins.to_value(u.GeV))), 0.2)

    bins = create_bins_per_decade(100 * u.GeV, 100 * u.TeV, 10)
    assert bins.unit == u.GeV
    assert len(bins) == 30  # end non-inclusive

    assert bins[0] == 100 * u.GeV
    assert np.allclose(np.diff(np.log10(bins.to_value(u.GeV))), 0.1)


def test_calculate_bin_indices():
    from pyirf.binning import calculate_bin_indices

    bins = np.array([0, 1, 2])
    values = [0.5, 0.5, 1, 1.1, 1.9, 2, -1, 2.5]

    true_idx = np.array([0, 0, 1, 1, 1, 2, -1, 2])

    assert np.all(calculate_bin_indices(values, bins) == true_idx)

    # test with units
    bins *= u.TeV
    values *= 1000 * u.GeV
    assert np.all(calculate_bin_indices(values, bins) == true_idx)
