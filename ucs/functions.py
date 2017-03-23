"""Compiled Theano functions, as well as NumPy equivalents of other symbolic functions."""

import numpy as np
import theano
import theano.tensor as T

from . import symbolic

_srgb_to_ucs = None


def srgb_to_ucs(RGB, L_A, Y_b, F, c, N_c):
    """Converts an sRGB (gamma=2.2) color to CAM02-UCS (Luo et al. (2006)) Jab."""
    global _srgb_to_ucs

    if _srgb_to_ucs is None:
        rgb = T.matrix('rgb')
        _L_A, _Y_b, _F, _c, _N_c = T.scalars('L_A', 'Y_b', 'F', 'c', 'N_c')
        ucs = symbolic.srgb_to_ucs(rgb, _L_A, _Y_b, _F, _c, _N_c)
        _srgb_to_ucs = theano.function([rgb, _L_A, _Y_b, _F, _c, _N_c], ucs,
                                       allow_input_downcast=True, on_unused_input='ignore')
    return _srgb_to_ucs(np.asarray(RGB), L_A, Y_b, F, c, N_c)


def delta_e(Jab1, Jab2):
    """Returns the Euclidean distance between two CAM02-UCS Jab colors."""
    return np.sqrt(np.sum(np.square(Jab1 - Jab2)))


def jab_to_jmh(Jab):
    """Converts a rectangular (Jab) CAM02-UCS color to cylindrical (JMh) format."""
    J, a, b = Jab[:, 0], Jab[:, 1], Jab[:, 2]
    M = np.sqrt(a**2 + b**2)
    h = np.rad2deg(np.arctan2(b, a))
    return np.stack([J, M, h], axis=1)


def jmh_to_jab(JMh):
    """Converts a cylindrical (JMh) CAM02-UCS color to rectangular (Jab) format."""
    J, M, h = JMh[:, 0], JMh[:, 1], JMh[:, 2]
    a = M * np.cos(np.deg2rad(h))
    b = M * np.sin(np.deg2rad(h))
    return np.stack([J, a, b], axis=1)
