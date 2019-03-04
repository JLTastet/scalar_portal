# Scalar Portal

This package implements the main production and decay channels of a light (GeV-scale) Higgs-like scalar in a beam-dump setup.

### Known issues
* Which mass to use to compute the velocity factor β³ in `S -> q qbar` is ambiguous. We use the quark mass (pole mass near the threshold, MSbar mass otherwise), but [1310.8042](http://arxiv.org/abs/1310.8042) instead uses the mass of the lightest meson containing the quark.
