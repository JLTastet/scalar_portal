# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

from ..data.constants import *
from ..data.particles import *
from ..data.form_factors import *

import numpy as np


_quark_masses = {
    'u': 0,
    'd': 0,
    's': 0,
    'c': 0,
    'b': scale_invariant_mass('b'),
    't': scale_invariant_mass('t'),
}

def _get_quark_mass(Q):
    assert(Q in _quark_masses)
    return _quark_masses[Q]

_up_quarks   = ['u', 'c', 't']
_down_quarks = ['d', 's', 'b']

def xi(UD, Qi, Qj):
    r"""
    Absolute value of the constant $\xi_Q^{ij}$ from the effective
    flavor-changing Lagrangian.

    We assume all light quarks (u, d, s, c) to be massless, and we use the
    scale-invariant mass in the MS-bar scheme for the heavy quarks.
    """
    prefactor = 3*sqrt2*GF / (16*pi**2)
    if UD == 'D':
        return prefactor * sum(VUD(k,Qj) * _get_quark_mass(k)**2 * VUD(k,Qi)
                               for k in _up_quarks)
    elif UD == 'U':
        return prefactor * sum(VUD(Qj,k) * _get_quark_mass(k)**2 * VUD(Qi,k)
                               for k in _down_quarks)
    else:
        raise(ValueError('Wrong quark type {} (must be U or D).'.format(UD)))

_quark_transitions = {
    ( 0,  0, -1): ('D', 'b', 'd'),
    (+1,  0, -1): ('D', 'b', 's'),
    ( 0, -1,  0): ('U', 'c', 'u'),
    (-1,  0,  0): ('D', 's', 'd'),
}

def _get_quark_transition(Y, Y1):
    S , C , B  = get_abs_strangeness(Y ), get_abs_charm(Y ), get_abs_beauty(Y )
    S1, C1, B1 = get_abs_strangeness(Y1), get_abs_charm(Y1), get_abs_beauty(Y1)
    dS, dC, dB = S1-S, C1-C, B1-B
    try:
        return _quark_transitions[(dS, dC, dB)]
    except KeyError:
        raise(ValueError('Unhandled quark transition {} -> {}.'.format(Y, Y1)))

def _get_xi(Y, Y1):
    return xi(*_get_quark_transition(Y, Y1))

def _momentum(m0, m1, m2):
    with np.errstate(invalid='ignore'):
        return np.sqrt((m0**2 - (m1+m2)**2) * (m0**2 - (m1-m2)**2)) / (2*m0)

# Matrix elements
# ---------------

# Compared to the paper, the matrix elements include an additional factor mQi.

# Keys: (2S+1, P)
_matrix_elements = {}

# Pseudoscalar

def MXP(q2, mX, mP, mQi, mQj, F):
    return (mX**2 - mP**2) / (mQj/mQi - 1) * F(q2)

_matrix_elements[(1, -1)] = MXP

# Scalar

def MXS(q2, mX, mS, mQi, mQj, F):
    return 1j * (mX**2 - mS**2) / (mQj/mQi + 1) * F(q2)

_matrix_elements[(1, +1)] = MXS

# Vector

def MXV(q2, mX, mV, mQi, mQj, F):
    pV = _momentum(mX, mV, np.sqrt(q2))
    return (2 * mX * pV) / (mQj/mQi + 1) * F(q2)

_matrix_elements[(3, -1)] = MXV

# Pseudo-vector

def MXA(q2, mX, mA, mQi, mQj, F):
    pA = _momentum(mX, mA, np.sqrt(q2))
    return (2 * mX * pA) / (mQj/mQi - 1) * F(q2)

_matrix_elements[(3, +1)] = MXA

# Tensor

def MXT(q2, mX, mT, mQi, mQj, F):
    pT = _momentum(mX, mT, np.sqrt(q2))
    return 1/(mQj/mQi + 1) * np.sqrt(2/3) * (mX * pT**2 / mT) * 2 * F(q2)

_matrix_elements[(5, +1)] = MXT

def get_matrix_element(X, X1):
    mX  = get_mass(X )
    mX1 = get_mass(X1)
    F = get_form_factor(X, X1)
    assert(get_parity(X) == -1)
    P = get_parity(X1)
    spin_code = get_spin_code(X1) # 2S+1
    assert((spin_code, P) in _matrix_elements)
    M = _matrix_elements[(spin_code, P)]
    _, Qi, Qj = _get_quark_transition(X, X1)
    mQi = _get_quark_mass(Qi)
    mQj = _get_quark_mass(Qj)
    # Regularize the sum if both quarks are massless
    # This gives the right answer in the limit mQi >> mQj.
    if mQi == 0 and mQj == 0:
        mQi = 1
    return lambda q2: M(q2, mX, mX1, mQi, mQj, F)
