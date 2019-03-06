# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

# Bibliography:
# [RPP18]
#   M. Tanabashi et al. (Particle Data Group), Phys. Rev. D 98, 030001 (2018)
#   DOI: 10.1103/PhysRevD.98.030001

# Shortcuts
pi = np.pi
sqrt2 = np.sqrt(2)
degree = pi/180

# Physical constants
#   [RPP18] Extracted from chapter 10: Electroweak Model and Constraints on New Physics

# Vacuum expection value of the Higgs field
v = 246.0 # GeV
vev = v / np.sqrt(2) # GeV

# Fermi constant
GF = 1.1663787e-5 # GeV⁻²

# Lepton masses (in GeV)
#   [RPP18] Extracted from Lepton Summary Table
m_e   =    0.51099894e-3
m_mu  =  105.65837e-3
m_tau = 1776.9e-3

# Electroweak boson masses (in GeV)
#   [RPP18] Extracted from Gauge & Higgs Boson summary table
M_W = 80.379
M_Z = 91.1876

# Strong coupling constant, in the MSbar scheme at the Z mass
#   [RPP18] Extracted from Review 1. Physical Constants, p.127
alpha_s_MZ = 0.1181

# Quark masses (in GeV)
#   [RPP18] Extracted from chapter 66: Quark Masses.

# Current quark masses (in GeV) in MSbar scheme at μ = 2 GeV
m_u_msbar_2GeV =  2.15e-3 # Eq. (66.6)
m_d_msbar_2GeV =  4.70e-3 # Eq. (66.6)
m_s_msbar_2GeV = 93.8e-3  # Eq. (66.3)

# Scale-invariant masses (in GeV) (MSbar masses at μ = m_Q(μ))
#   p.10, average from continuum determinations
m_c_si = 1.28
m_b_si = 4.18

# Top pole mass (in GeV).
#   [RPP18] Extracted from Quark Summary Table
m_t_os = 173.0 # p.40, Mass (direct measurements)

# Scale-invariant top mass, computed by finding the fixed point of the MS-bar
# mass using the formula from [1], accurate to order O(αs³) + O(α) + O(α αs).
# [1] Jegerlehner, Kalmykov and Kniehl (2013), 10.1016/j.physletb.2013.04.012
m_t_si = 174.0

# CKM matrix
#   [RPP18] Data extracted from chapter 12: CKM Quark-Mixing Matrix
#   The values quoted below are the absolute values of the (non-squared) CKM matrix elements

Vud     =  0.97420
Vud_err =  0.00021

Vus     =  0.2243
Vus_err =  0.0005

Vcd     =  0.218
Vcd_err =  0.004

Vcs     =  0.997
Vcs_err =  0.017

Vcb     = 42.2e-3
Vcb_err =  0.8e-3

Vub     =  3.94e-3
Vub_err =  0.36e-3

Vtd     =  8.1e-3
Vtd_err =  0.5e-3

Vts     = 39.4e-3
Vts_err =  2.3e-3

Vtb     =  1.019
Vtb_err =  0.025

_ckm_abs = np.array([
    [Vud, Vus, Vub],
    [Vcd, Vcs, Vcb],
    [Vtd, Vts, Vtb]
])

_ckm_abs_err = np.array([
    [Vud_err, Vus_err, Vub_err],
    [Vcd_err, Vcs_err, Vcb_err],
    [Vtd_err, Vts_err, Vtb_err]
])

def ckm(i,j):
    """
    Absolute value of the CKM entry corresponding to U generation index i=1,2,3
    and D generation index j=1,2,3.
    """
    if i >= 1 and i <= 3 and j >= 1 and j <= 3:
        return _ckm_abs[i-1, j-1]
    else:
        raise(ValueError('Wrong generation index in CKM matrix: ({},{}).'.format(i,j)))

# PYTHIA-related constants
#    Source: http://home.thep.lu.se/~torbjorn/pythia82html/Welcome.html

# "91 : decay to q qbar or g g, which should shower and hadronize"
pythia_me_mode_hadronize = 91

# Set some default PDG ID for the scalar particle.
# Here, we choose to prepend 99000 to the SM Higgs ID.
default_scalar_id = 9900025
