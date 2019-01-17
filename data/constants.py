# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

# Shortcuts

pi = np.pi
sqrt2 = np.sqrt(2)
degree = pi/180

# Physical constants
#   Source: M. Tanabashi et al. (Particle Data Group), Phys. Rev. D 98, 030001 (2018)
#   DOI: 10.1103/PhysRevD.98.030001
#   Extracted from chapter 10: Electroweak Model and Constraints on New Physics

# Vacuum expection value of the Higgs field
v = 246.0 # GeV
vev = v / np.sqrt(2) # GeV

# Fermi constant
GF = 1.1663787e-5 # GeV⁻²

# Quark masses (in GeV)
#   Source: M. Tanabashi et al. (Particle Data Group), Phys. Rev. D 98, 030001 (2018)
#   DOI: 10.1103/PhysRevD.98.030001
#   Extracted from Quark Summary Table

# Current quark masses (MSbar)
m_u =  2.2e-3
m_d =  4.7e-3
m_s = 95e-3

# Running masses
m_c_msbar = 1.275
m_b_msbar = 4.18

# Pole masses
m_b_pole =   4.8
m_t_pole = 173.0

# Masses used in the calculations
m_c = m_c_msbar
m_b = m_b_msbar
m_t = m_t_pole

_m_Q = {
    'U': [m_u, m_c, m_t],
    'D': [m_d, m_s, m_b]
}

def m_Q(UD, i):
    """
    Mass of the quark of type UD and generation i.
    """
    if i < 1 or i > 3:
        raise(ValueError('Wrong generation index {}.'.format(i)))
    try:
        return _m_Q[UD][i-1]
    except:
        raise(ValueError('No mass for quark {} and generation {}.'.format(UD, i)))

# CKM matrix
#   Source: M. Tanabashi et al. (Particle Data Group), Phys. Rev. D 98, 030001 (2018)
#   DOI: 10.1103/PhysRevD.98.030001
#   Data extracted from chapter 12: CKM Quark-Mixing Matrix
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

# ξⁱʲ_Q constants from the effective flavor-changing Lagrangian (2.11).
#   The quoted values are taken from Table 1.

xi_d_ds = 3.3e-6
xi_d_db = 7.9e-5
xi_d_sb = 3.6e-4

xi_u_uc = 1.4e-9

_xi = {
    ('D', 1, 2): xi_d_ds,
    ('D', 1, 3): xi_d_db,
    ('D', 2, 3): xi_d_sb,
    ('U', 1, 2): xi_u_uc
}

def xi(UD, i, j):
    r"""
    Absolute value of the constant $\xi_Q^{ij}$ from the effective flavor-changing
    Lagrangian.
    """
    if not j > i:
        raise(ValueError('Initial generation i={} must be higher than j={}.'.format(i, j)))
    prefactor = 3*sqrt2*GF / (16*pi**2)
    if UD == 'D':
        return prefactor * sum(ckm(k,i) * m_Q('U', k)**2 * ckm(k,j) for k in range(1, 4))
    elif UD == 'U':
        return prefactor * sum(ckm(i,k) * m_Q('D', k)**2 * ckm(j,k) for k in range(1, 4))
    else:
        raise(ValueError('Wrong quark type {} (must be U or D).'.format(UD)))
