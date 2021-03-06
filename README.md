# Scalar Portal

[![Build Status](https://travis-ci.org/JLTastet/scalar_portal.svg?branch=master)](https://travis-ci.org/JLTastet/scalar_portal)
[![codecov.io](http://codecov.io/github/JLTastet/scalar_portal/coverage.svg?branch=master)](http://codecov.io/github/JLTastet/scalar_portal?branch=master)

This package implements the main production and decay channels of a light (GeV-scale) Higgs-like scalar in a beam-dump setup.
It closely follows formulas from [1904.10447](http://arxiv.org/abs/1904.10447).
Please cite this paper if you use this package in your research.

### Usage

```python
from __future__ import print_function
from scalar_portal import Model

# Instantiate the model.
m = Model()

# List the available production channels and groups.
m.production.list_available()
m.production.list_available_groups()

# Enable all B -> S K? production channels, where K? = K, K*, K_1, K*_2, ...
m.production.enable('B -> S K?')

# Check which production channels have been enabled.
m.production.list_enabled()

# List the available decay channels and groups.
m.decay.list_available()
m.decay.list_available_groups()

# Enable the relevant decay channels for a light Scalar below 2 GeV.
m.decay.enable('LightScalar')

# Check which decay channels have been enabled.
m.decay.list_enabled()

# Compute the branching ratios for a Scalar mass of 1.2 GeV and θ = 1e-4.
res = m.compute_branching_ratios(1.2, theta=1e-4)

# Query the lifetime of the Scalar (in seconds) and its total decay width in GeV.
res.lifetime_si
res.total_width

# Query the production branching ratios for the Scalar.
# This returns a dictionary of (channel string, branching ratio) pairs.
res.production.branching_ratios

# Query its decay branching ratios.
res.decay.branching_ratios

# Print the PYTHIA string that would be used to implement this model.
print(res.pythia_full_string())

# Note: properties return precomputed results, while method calls do extra computations.

# Now use the vectorized interface to compute the branching ratios for a range of masses and couplings.
# Note: not all calculations are internally vectorized (e.g. the running of alpha_s is not).
import numpy as np
mS = np.linspace(0.1, 1.8, 50)
theta = np.linspace(1e-3, 1e-2, 20)
res = m.compute_branching_ratios(mS[:,np.newaxis], theta=theta[np.newaxis,:])
assert(res.total_width.shape == (50, 20))

# All results except PYTHIA strings are vectorized.
res.lifetime_si
res.production.branching_ratios
res.decay.widths

# If the calculation is invalid for a given mass and θ, NaN's are returned, e.g.:
res = m.compute_branching_ratios(3.0, theta=1e-5)
assert(np.isnan(res.total_width))

# For a Scalar above 2 GeV, we must select the decay channels for the heavy Scalar.
m.decay.disable_all()
m.decay.enable('HeavyScalar')
res = m.compute_branching_ratios(3.0, theta=1e-5)
assert(np.isfinite(res.total_width))
```

### Known issues
* We currently ignore charm threshold effects such as S – χ mixing, which can lead to a large enhancement of Γ(S → g g) in the vicinity of c–cbar bound states, as discussed in \[1\].
* We only consider weak eigenstates for B_s(bar)0 -> S S.
* H -> H' S S is currently implemented as a pure phase-space decay in PYTHIA (meMode=0).

1. Drees, M., Hikasa, K., 1990.  
   Heavy-quark thresholds in Higgs-boson physics.  
   Phys. Rev. D 41, 1547–1561.  
   https://doi.org/10.1103/PhysRevD.41.1547
