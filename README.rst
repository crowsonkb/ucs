ucs
===

Implements the CAM02-UCS (Luo et al. (2006), "Uniform Colour Spaces Based on CIECAM02 Colour Appearance Model") forward transform symbolically, using Theano.

See: `CIECAM02 and Its Recent Developments <http://www.springer.com/cda/content/document/cda_downloaddocument/9781441961891-c1.pdf>`_.

The forward transform is symbolically differentiable in Theano and it may be approximately inverted, subject to gamut boundaries, by constrained function minimization (e.g. projected gradient descent or L-BFGS-B).

Package contents
----------------

- ``constants.py`` contains constants needed by CAM02-UCS and others which are merely useful.

- ``functions.py`` contains compiled Theano functions, as well as NumPy equivalents of other symbolic functions.

- ``symbolic.py`` implements the forward transform symbolically in Theano. The functions therein can be used to construct custom auto-differentiable loss functions to be subject to optimization.
