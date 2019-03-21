# Scalar Portal

This package implements the main production and decay channels of a light (GeV-scale) Higgs-like scalar in a beam-dump setup.

### Known issues
* We currently ignore charm threshold effects such as S – χ mixing, which can lead to a large enhancement of Γ(S → g g) in the vicinity of c–cbar bound states, as discussed in \[1\].
* In order to obtain the correct physical threshold for S → D Dbar, we compute the phase-space factor β³ using m\_D instead of mbar\_c in Γ(S → c cbar). It is not clear whether this is correct, since β³(mbar\_c) was computed for a P-wave c–cbar final state, while \[1\] argues that the first mesonic open-flavor decay modes, such as S → D Dbar, should be S-wave. So we might need to use β(m\_D) instead.

1. Drees, M., Hikasa, K., 1990.  
   Heavy-quark thresholds in Higgs-boson physics.  
   Phys. Rev. D 41, 1547–1561.  
   https://doi.org/10.1103/PhysRevD.41.1547
