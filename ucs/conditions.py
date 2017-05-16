"""Represents CIECAM02/CAM02-UCS viewing conditions."""

from ucs.constants import Surrounds


class Conditions:
    """The following viewing conditions can be specified:

    Y_w: relative luminance of reference white in adapting field. For a display, the brightness in
        cd/m2 of the display's white point.
    L_A: luminance of adapting field. Defaults to Y_w / 5.
    Y_b: luminance of background. For a display, this is the average luminance in cd/m2 of the
        display given typical content. Defaults to Y_w / 5.
    surround: the CIECAM02 surround ('average', 'dim', or 'dark'). Defaults to 'average'.

    Instantiating this class with no parameters will choose reasonable defaults for an sRGB display.
    """
    def __init__(self, Y_w=80, L_A=None, Y_b=None, surround='average'):
        self.Y_w = float(Y_w)
        self.L_A = L_A if L_A else Y_w / 5
        self.Y_b = Y_b if Y_b else Y_w / 5
        if isinstance(surround, dict):
            surr_dict = surround
        else:
            surr_dict = getattr(Surrounds, surround.upper(), Surrounds.AVERAGE)
        self.F, self.c, self.N_c = surr_dict['F'], surr_dict['c'], surr_dict['N_c']

    def __repr__(self):
        kv = ', '.join('{:s}={:g}'.format(k, v) for k, v in vars(self).items())
        return 'Conditions(' + kv + ')'

    def __iter__(self):
        return iter(vars(self).values())
