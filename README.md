# Scalar Portal

[![Build Status](https://travis-ci.org/JLTastet/scalar_portal.svg?branch=master)](https://travis-ci.org/JLTastet/scalar_portal)
[![codecov.io](http://codecov.io/github/JLTastet/scalar_portal/coverage.svg?branch=master)](http://codecov.io/github/JLTastet/scalar_portal?branch=master)

This package implements the main production and decay channels of a light (GeV-scale) Higgs-like scalar in a beam-dump setup.

### Known issues
* We currently ignore charm threshold effects such as S – χ mixing, which can lead to a large enhancement of Γ(S → g g) in the vicinity of c–cbar bound states, as discussed in \[1\].
* In order to obtain the correct physical threshold for S → D Dbar, we use the S-wave phase-space factor β(m\_D) instead of the P-wave one, β³(mbar\_c), for Γ(S → c cbar). This needs to be double-checked.

1. Drees, M., Hikasa, K., 1990.  
   Heavy-quark thresholds in Higgs-boson physics.  
   Phys. Rev. D 41, 1547–1561.  
   https://doi.org/10.1103/PhysRevD.41.1547
