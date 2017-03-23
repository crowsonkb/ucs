ucs
===

Implements the CAM02-UCS (Luo et al. (2006)) forward transform symbolically, using Theano.

See: `CIECAM02 and Its Recent Developments <http://www.springer.com/cda/content/document/cda_downloaddocument/9781441961891-c1.pdf>`_.

The forward transform is symbolically differentiable and it may be inverted, subject to gamut boundaries, by constrained function minimization (e.g. projected gradient descent or L-BFGS-B).
