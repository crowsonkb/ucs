"""Constants required by CAM02-UCS or which are otherwise useful."""

import numpy as np
import theano

EPS = np.finfo(theano.config.floatX).eps

floatX = getattr(np, theano.config.floatX)


class Surrounds:
    """CIECAM02 surround conditions."""
    AVERAGE = dict(F=1, c=0.69, N_c=1)
    DIM = dict(F=0.9, c=0.59, N_c=0.95)
    DARK = dict(F=0.8, c=0.525, N_c=0.8)

M_CAT02 = floatX(
    [[0.7328, 0.4296, -0.1624],
     [-0.7036, 1.6975, 0.0061],
     [0.003, 0.0136, 0.9834]]).T

M_CAT02_inv = np.linalg.inv(M_CAT02)

M_HPE = floatX(
    [[0.38971, 0.68898, -0.07868],
     [-0.22981, 1.1834, 0.04641],
     [0, 0, 1]]).T

M_SRGB_to_XYZ = floatX(
    [[0.4124, 0.3576, 0.1805],
     [0.2126, 0.7152, 0.0722],
     [0.0193, 0.1192, 0.9505]]).T
