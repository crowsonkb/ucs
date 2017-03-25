"""Compiled Theano functions, as well as NumPy equivalents of other symbolic functions."""

import sys

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import theano
import theano.tensor as T

from ucs.constants import EPS, M_SRGB_to_XYZ
from ucs import symbolic

_srgb_to_ucs = None
_ucs_to_srgb_helper = None


def srgb_to_xyz(RGB):
    """Converts sRGB (gamma=2.2) colors to XYZ."""
    RGB_linear = np.maximum(EPS, RGB)**2.2
    return np.dot(RGB_linear, M_SRGB_to_XYZ)


def srgb_to_ucs(RGB, Y_w=100, L_A=20, Y_b=20, F=1, c=0.69, N_c=1):
    """Converts sRGB (gamma=2.2) colors to CAM02-UCS (Luo et al. (2006)) Jab."""
    global _srgb_to_ucs

    if _srgb_to_ucs is None:
        print('Building srgb_to_ucs()...', file=sys.stderr)
        rgb = T.matrix('rgb')
        conditions = T.scalars('Y_w', 'L_A', 'Y_b', 'F', 'c', 'N_c')
        ucs = symbolic.srgb_to_ucs(rgb, *conditions)
        _srgb_to_ucs = theano.function([rgb] + conditions, ucs,
                                       allow_input_downcast=True, on_unused_input='ignore')
    return _srgb_to_ucs(np.atleast_2d(RGB), Y_w, L_A, Y_b, F, c, N_c)


def ucs_to_srgb_helper(X, Jab, Y_w, L_A, Y_b, F, c, N_c):
    """Loss and gradient at point X (sRGB space) of the distance between the corresponding
    Jab color and a target Jab color. Descending this gradient will approximately invert
    srgb_to_ucs()."""
    global _ucs_to_srgb_helper

    if _ucs_to_srgb_helper is None:
        print('Building ucs_to_srgb_helper()...', file=sys.stderr)
        conditions = T.scalars('Y_w', 'L_A', 'Y_b', 'F', 'c', 'N_c')
        x, jab = T.vectors('x', 'jab')
        jab_x = symbolic.srgb_to_ucs(x, *conditions)
        loss = symbolic.delta_e(jab_x, jab)**2
        grad = T.grad(loss, x)
        _ucs_to_srgb_helper = theano.function([x, jab] + conditions, [loss, grad],
                                              allow_input_downcast=True, on_unused_input='ignore')
    return _ucs_to_srgb_helper(np.atleast_1d(X), np.atleast_1d(Jab), Y_w, L_A, Y_b, F, c, N_c)


def ucs_to_srgb(Jab, Y_w=100, L_A=20, Y_b=20, F=1, c=0.69, N_c=1):
    """Approximately inverts srgb_to_ucs() for a single color."""
    x, _, _ = fmin_l_bfgs_b(ucs_to_srgb_helper, np.float64([0.5, 0.5, 0.5]),
                            args=(Jab, Y_w, L_A, Y_b, F, c, N_c))
    return x


def ucs_to_srgb_b(Jab, Y_w=100, L_A=20, Y_b=20, F=1, c=0.69, N_c=1):
    """Approximately inverts srgb_to_ucs() for a single color subject to sRGB gamut limits."""
    x, _, _ = fmin_l_bfgs_b(ucs_to_srgb_helper, np.float64([0.5, 0.5, 0.5]),
                            args=(Jab, Y_w, L_A, Y_b, F, c, N_c), bounds=[(0, 1)]*3)
    return x


def delta_e(Jab1, Jab2):
    """Returns the Euclidean distance between two CAM02-UCS Jab colors."""
    return np.sqrt(np.sum(np.square(Jab1 - Jab2)))


def jab_to_jmh(Jab):
    """Converts rectangular (Jab) CAM02-UCS colors to cylindrical (JMh) format."""
    Jab = np.atleast_1d(Jab)
    J, a, b = Jab[..., 0], Jab[..., 1], Jab[..., 2]
    M = np.sqrt(a**2 + b**2)
    h = np.rad2deg(np.arctan2(b, a))
    return np.stack([J, M, h], axis=-1)


def jmh_to_jab(JMh):
    """Converts cylindrical (JMh) CAM02-UCS colors to rectangular (Jab) format."""
    JMh = np.atleast_1d(JMh)
    J, M, h = JMh[..., 0], JMh[..., 1], JMh[..., 2]
    a = M * np.cos(np.deg2rad(h))
    b = M * np.sin(np.deg2rad(h))
    return np.stack([J, a, b], axis=-1)
