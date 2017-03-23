ucs
===

Implements the CAM02-UCS (Luo et al. (2006), "Uniform Colour Spaces Based on CIECAM02 Colour Appearance Model") forward transform symbolically, using Theano.

See: `CIECAM02 and Its Recent Developments <http://www.springer.com/cda/content/document/cda_downloaddocument/9781441961891-c1.pdf>`_.

The forward transform is symbolically differentiable in Theano and it may be approximately inverted, subject to gamut boundaries, by constrained function minimization (e.g. projected gradient descent or L-BFGS-B).
