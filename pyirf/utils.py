import numpy as np


def is_scalar(val):
    '''Workaround that also supports astropy quantities'''
    return np.array(val, copy=False).shape == tuple()
