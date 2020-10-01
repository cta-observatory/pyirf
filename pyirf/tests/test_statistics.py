def test_lima():
    from pyirf.statistics import li_ma_significance

    assert li_ma_significance(10, 2, 0.2) > 5
    assert li_ma_significance(10, 0, 0.2) > 5
    assert li_ma_significance(1, 6, 0.2) == 0
