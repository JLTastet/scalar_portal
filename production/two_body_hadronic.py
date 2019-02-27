# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

from ..data.constants import *
from ..data.particles import *
from ..data.form_factors import *

import numpy as np

# ξⁱʲ_Q constants from the effective flavor-changing Lagrangian (2.11).
#   Values from Kyrylo's thesis (Table 1).
_xi_d_ds = 3.3e-6
_xi_d_db = 7.9e-5
_xi_d_sb = 3.6e-4
_xi_u_uc = 1.4e-9
_xi_ref = {
    ('D', 2, 1): _xi_d_ds,
    ('D', 3, 1): _xi_d_db,
    ('D', 3, 2): _xi_d_sb,
    ('U', 2, 1): _xi_u_uc
}

def xi(UD, i, j):
    r"""
    Absolute value of the constant $\xi_Q^{ij}$ from the effective
    flavor-changing Lagrangian.

    We assume all light quarks (u, d, s, c) to be massless.

    NOTE: The ambiguity regarding which renormalization scheme to use leads to
    large uncertainties on this value, which is the symptom of potentially
    large QCD corrections. Here we use the on-shell scheme, which results in a
    conservative estimate compared to MSbar.
    """
    if not i > j:
        raise(ValueError('Initial generation i={} must be higher than j={}.'.format(i, j)))
    prefactor = 3*sqrt2*GF / (16*pi**2)
    if UD == 'D':
        mt = on_shell_mass('t') #FIXME
        return prefactor * ckm(3,j) * mt**2 * ckm(3,i)
    elif UD == 'U':
        mb = on_shell_mass('b') #FIXME
        return prefactor * ckm(j,3) * mb**2 * ckm(i,3)
    else:
        raise(ValueError('Wrong quark type {} (must be U or D).'.format(UD)))

def _scalar_momentum(mY, mY1, mS):
    with np.errstate(invalid='ignore'):
        return np.sqrt( (mY**2 - (mS+mY1)**2) * (mY**2 - (mS-mY1)**2) ) / (2*mY)

def _chi0(mY, mY1, mS):
    return (mY**2 - mY1**2) / 2

def _chi1(mY, mY1, mS):
    pS = _scalar_momentum(mY, mY1, mS)
    return mY * pS

def _chi2(mY, mY1, mS):
    pS = _scalar_momentum(mY, mY1, mS)
    return np.sqrt(2/3) * (mY/mY1) * pS**2

def _chi(Y, Y1, mS):
    mY  = get_mass(Y )
    mY1 = get_mass(Y1)
    J1  = get_spin_code(Y1)
    if J1 == 1: # Spin 0
        return _chi0(mY, mY1, mS)
    elif J1 == 3: # Spin 1
        return _chi1(mY, mY1, mS)
    elif J1 == 5: # Spin 2
        return _chi2(mY, mY1, mS)
    else:
        raise(ValueError('Invalid spin code {} for meson {}.'.format(J1, Y1)))

_quark_transitions = {
    ( 0,  0, -1): ('D', 3, 1),
    (+1,  0, -1): ('D', 3, 2),
    ( 0, -1,  0): ('U', 2, 1),
    (-1,  0,  0): ('D', 2, 1),
}

def _get_quark_transition(Y, Y1):
    S , C , B  = get_abs_strangeness(Y ), get_abs_charm(Y ), get_abs_beauty(Y )
    S1, C1, B1 = get_abs_strangeness(Y1), get_abs_charm(Y1), get_abs_beauty(Y1)
    dS, dC, dB = S1-S, C1-C, B1-B
    try:
        return _quark_transitions[(dS, dC, dB)]
    except KeyError:
        raise(ValueError('Unhandled quark transition for {} -> {}.'.format(Y, Y1)))

def _get_xi(Y, Y1):
    return xi(*_get_quark_transition(Y, Y1))

def _available_mass(Y, Y1):
    return get_mass(Y) - get_mass(Y1)

def _normalized_amplitude(Y, Y1, mS):
    """
    Computes the transition amplitude for the process Y_q -> S Y'_q', divided by
    the mixing angle θ.

    We approximate $m_{Q_i} / (m_{Q_i} \pm m_{Q_j}) ≈ 1$.

    Clebsch-Gordan coefficients are irrelevant for the considered modes.
    """
    xi_Q = _get_xi(Y, Y1)
    chi = _chi(Y, Y1, mS)
    F = get_form_factor(Y, Y1)
    A = xi_Q * (chi / v) * F(mS**2)
    # Set the amplitude to zero if the channel is kinematically closed.
    return np.where(mS < _available_mass(Y, Y1), A, 0.)

def normalized_decay_width(Y, Y1, mS):
    """
    Computes the decay width for the process Y_q -> S Y'_q', divided by the
    mixing angle θ.
    """
    A = _normalized_amplitude(Y, Y1, mS)
    mY  = get_mass(Y )
    mY1 = get_mass(Y1)
    pS = np.where(mS < _available_mass(Y, Y1), _scalar_momentum(mY, mY1, mS), 0.)
    return ( A**2 * pS ) / ( 8*pi * mY**2 )
