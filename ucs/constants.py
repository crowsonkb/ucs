from collections import namedtuple

import numpy as np

Surround = namedtuple('Surround', 'F c N_c')


class Surrounds:
    AVERAGE = Surround(1, 0.69, 1)
    DIM = Surround(0.9, 0.59, 0.95)
    DARK = Surround(0.8, 0.525, 0.8)

M_CAT02 = np.float64(
    [[0.7328, 0.4296, -0.1624],
     [-0.7036, 1.6975, 0.0061],
     [0.003, 0.0136, 0.9834]]).T

M_CAT02_inv = np.linalg.inv(M_CAT02)

M_HPE = np.float64(
    [[0.38971, 0.68898, -0.07868],
     [-0.22981, 1.1834, 0.04641],
     [0, 0, 1]]).T

sRGB_to_XYZ = np.float64(
    [[0.4124, 0.3576, 0.1805],
     [0.2126, 0.7152, 0.0722],
     [0.0193, 0.1192, 0.9505]]).T
