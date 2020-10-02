import pytest
import numpy as np


def test_lima():
    from pyirf.statistics import li_ma_significance

    assert li_ma_significance(10, 2, 0.2) > 5
    assert li_ma_significance(10, 0, 0.2) > 5
    assert li_ma_significance(1, 6, 0.2) == 0


def test_lima_gammapy():
    pytest.importorskip('gammapy')
    from gammapy.stats import WStatCountsStatistic
    from pyirf.statistics import li_ma_significance

    n_ons = [100, 50, 10]
    n_offs = [10, 20, 30]
    alphas = [2, 1, 0.2]
    for n_on, n_off, alpha in zip(n_ons, n_offs, alphas):
        sig_gammapy = WStatCountsStatistic(n_on, n_off, alpha).significance
        assert np.isclose(li_ma_significance(n_on, n_off, alpha), sig_gammapy)
