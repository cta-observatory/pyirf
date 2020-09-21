import numpy as np
from astropy.table import QTable
import astropy.units as u
import pytest


@pytest.fixture
def events():
    return QTable({
        'bin_reco_energy': [0, 0, 1, 1, 2, 2],
        'theta': [0.1, 0.02, 0.3, 0.15, 0.01, 0.1] * u.deg,
        'gh_score': [1.0, -0.2, 0.5, 0.05, 1.0, 0.3],
    })


def test_is_selected(events):
    from pyirf.cuts import is_selected

    cut_definition = {
        'theta': {
            'operator': 'le',
            'cut_values': [0.05, 0.15, 0.25] * u.deg,
        },
        'gh_score': {
            'operator': 'ge',
            'cut_values': np.array([0.0, 0.1, 0.5]),
        }
    }

    # if you make no cuts, all events are selected
    assert np.all(is_selected(events, {}, bin_index=events['bin_reco_energy']) == True)  # noqa

    selected = is_selected(
        events, cut_definition, bin_index=events['bin_reco_energy']
    )

    assert selected.dtype == np.bool
    assert np.all(selected == [False, False, False, False, True, False])


def test_is_selected_single_numbers(events):
    from pyirf.cuts import is_selected

    cut_definition = {
        'theta': {
            'operator': 'le',
            'cut_values': 0.05 * u.deg,
        },
        'gh_score': {
            'operator': 'ge',
            'cut_values': 0.5,
        }
    }

    selected = is_selected(
        events, cut_definition, bin_index=events['bin_reco_energy']
    )

    assert selected.dtype == np.bool
    assert np.all(selected == [False, False, False, False, True, False])


def test_is_scalar():
    from pyirf.cuts import is_scalar

    assert is_scalar(1.0)
    assert is_scalar(5 * u.m)
    assert is_scalar(np.array(5))

    assert not is_scalar([1, 2, 3])
    assert not is_scalar([1, 2, 3] * u.m)
    assert not is_scalar(np.ones(5))
    assert not is_scalar(np.ones((3, 4)))
