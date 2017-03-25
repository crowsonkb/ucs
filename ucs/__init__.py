"""Implements the CAM02-UCS (Luo et al. (2006)) forward transform."""

import theano, warnings
if theano.config.floatX != 'float64':
    warnings.warn('theano.config.floatX is not \'float64\'. It is recommended that the ucs'
                  'package be used with Theano in float64 mode.')
del theano, warnings

from ucs.constants import Hues, Surrounds
from ucs.functions import (delta_e, jab_to_jmh, jmh_to_jab, srgb_to_ucs, srgb_to_xyz,
                           ucs_to_srgb, ucs_to_srgb_b)
