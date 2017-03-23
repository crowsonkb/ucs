"""Implements the CAM02-UCS (Luo et al. (2006)) forward transform symbolically, using Theano."""

# pylint: disable=assignment-from-no-return, too-many-locals

import numpy as np
import theano
import theano.tensor as T

from ucs.constants import EPS, floatX, M_CAT02, M_CAT02_inv, M_HPE, M_SRGB_to_XYZ, Surrounds


def srgb_to_ucs(RGB, Y_w, L_A, Y_b, F, c, N_c):
    """Converts sRGB (gamma=2.2) colors to CAM02-UCS (Luo et al. (2006)) Jab."""
    XYZ_w = T.dot(floatX([[1, 1, 1]]), M_SRGB_to_XYZ) * Y_w
    RGB_w = T.dot(XYZ_w, M_CAT02)
    # D = T.clip(F * (1 - (1/3.6) * T.exp((-L_A - 42) / 92)), 0, 1)
    D = floatX([1, 1, 1])  # Discount the illuminant fully
    k = 1 / (5 * L_A + 1)
    D_rgb = D * Y_w / RGB_w + 1 - D
    F_L = 0.2 * k**4 * (5 * L_A) + 0.1 * (1 - k**4)**2 * (5 * L_A)**(1/3)
    n = Y_b / Y_w
    z = 1.48 + T.sqrt(n)
    N_bb = 0.725 * (1/n)**0.2
    N_cb = N_bb
    RGB_wc = D_rgb * RGB_w
    RGB_wp = T.dot(T.dot(RGB_wc, M_CAT02_inv), M_HPE)
    RGB_aw_i = (F_L * RGB_wp / 100)**0.42
    RGB_aw = 400 * RGB_aw_i / (RGB_aw_i + 27.13) + 0.1
    A_w = (T.sum(RGB_aw * floatX([2, 1, 1/20]), axis=-1) - 0.305) * N_bb

    RGB_linear = T.maximum(EPS, RGB)**2.2
    XYZ = T.dot(RGB_linear, M_SRGB_to_XYZ) * Y_w
    RGB_ = T.dot(XYZ, M_CAT02)
    RGB_c = D_rgb * RGB_
    RGB_p = T.dot(T.dot(RGB_c, M_CAT02_inv), M_HPE)
    RGB_ap_p_i = (F_L * RGB_p / 100)**0.42
    RGB_ap_n_i = (-F_L * RGB_p / 100)**0.42
    RGB_ap_p = 400 * RGB_ap_p_i / (RGB_ap_p_i + 27.13) + 0.1
    RGB_ap_n = -400 * RGB_ap_n_i / (RGB_ap_n_i + 27.13) + 0.1
    RGB_ap = T.switch(RGB_ap_p > 0, RGB_ap_p, RGB_ap_n)

    a = T.sum(RGB_ap * floatX([1, -12/11, 1/11]), axis=-1)
    b = T.sum(RGB_ap * floatX([1/9, 1/9, -2/9]), axis=-1)
    h = T.rad2deg(T.arctan2(b, a))
    h_p = T.switch(h < 0, h + 360, h)
    e_t = (T.cos(h_p * np.pi / 180 + 2) + 3.8) / 4

    A = (T.sum(RGB_ap * floatX([2, 1, 1/20]), axis=-1) - 0.305) * N_bb
    J = 100 * T.maximum(0, A / A_w)**(c * z)
    t_num = 50000/13 * N_c * N_cb * e_t * T.sqrt(a**2 + b**2)
    t = t_num / T.sum(RGB_ap * floatX([1, 1, 21/20]), axis=-1)
    C = t**0.9 * T.sqrt(J / 100) * (1.64 - 0.29**n)**0.73
    M = C * F_L**0.25

    K_L, c_1, c_2 = 1, 0.007, 0.0228
    J_p = (1 + 100 * c_1) * J / (1 + c_1 * J)
    M_p = (1 / c_2) * T.log(1 + c_2 * M)
    a_Mp = M_p * T.cos(T.deg2rad(h_p))
    b_Mp = M_p * T.sin(T.deg2rad(h_p))
    return T.stack([J_p, a_Mp, b_Mp], axis=-1)


def delta_e(Jab1, Jab2):
    """Returns the Euclidean distance between two CAM02-UCS Jab colors."""
    return T.sqrt(T.sum(T.sqr(Jab1 - Jab2)))


def jab_to_jmh(Jab):
    """Converts a rectangular (Jab) CAM02-UCS color to cylindrical (JMh) format."""
    J, a, b = Jab[:, 0], Jab[:, 1], Jab[:, 2]
    M = T.sqrt(a**2 + b**2)
    h = T.rad2deg(T.arctan2(b, a))
    return T.stack([J, M, h], axis=-1)


def jmh_to_jab(JMh):
    """Converts a cylindrical (JMh) CAM02-UCS color to rectangular (Jab) format."""
    J, M, h = JMh[:, 0], JMh[:, 1], JMh[:, 2]
    a = M * T.cos(T.deg2rad(h))
    b = M * T.sin(T.deg2rad(h))
    return T.stack([J, a, b], axis=-1)


def main():
    """A simple test case."""
    rgb1, rgb2 = T.matrices('rgb1', 'rgb2')
    jab1 = srgb_to_ucs(rgb1, 100, 20, 20, **Surrounds.AVERAGE)
    jab2 = srgb_to_ucs(rgb2, 100, 20, 20, **Surrounds.AVERAGE)
    loss = delta_e(jab1, jab2)**2
    grad_ = T.grad(loss, rgb2)
    grad = theano.function([rgb1, rgb2], grad_)

    # Inversion of CAM02-UCS via gradient descent
    target = floatX([[0.25, 0.25, 1]])
    x = np.zeros_like(target) + 0.5
    print(x)
    for i in range(1500):
        g = grad(target, x)
        x -= 1e-5 * g
        if i % 100 == 99:
            print(x)

if __name__ == '__main__':
    main()
