def test_prod5_irfs(prod5_irfs):
    assert len(prod5_irfs) == 3
    assert "edisp" in prod5_irfs[0]
