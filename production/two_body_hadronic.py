# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

from ..data.constants import *
from ..data.particles import *
from ..data.form_factors import *
from ..api.channel import ProductionChannel

import numpy as np


def xi(UD, i, j):
    r"""
    Absolute value of the constant $\xi_Q^{ij}$ from the effective
    flavor-changing Lagrangian.

    We assume all light quarks (u, d, s, c) to be massless, and we use the
    scale-invariant mass in the MS-bar scheme for the heavy quarks.
    """
    if not i > j:
        raise(ValueError('Initial generation i={} must be higher than j={}.'.format(i, j)))
    prefactor = 3*sqrt2*GF / (16*pi**2)
    if UD == 'D':
        mt = scale_invariant_mass('t')
        return prefactor * ckm(3,j) * mt**2 * ckm(3,i)
    elif UD == 'U':
        mb = scale_invariant_mass('b')
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
    mS = np.asarray(mS, dtype='float')
    A = _normalized_amplitude(Y, Y1, mS)
    mY  = get_mass(Y )
    mY1 = get_mass(Y1)
    pS = np.where(mS < _available_mass(Y, Y1), _scalar_momentum(mY, mY1, mS), 0.)
    return ( A**2 * pS ) / ( 8*pi * mY**2 )


class TwoBodyHadronic(ProductionChannel):
    '''
    Scalar production through exclusive 2-body hadronic decays: H -> S H'.
    '''
    def __init__(self, H, H1):
        super(TwoBodyHadronic, self).__init__(H, [H1])
        if not (is_meson(H) and is_meson(H1)):
            raise(ValueError('{} and {} must be mesons.'.format(H, H1)))
        try:
            self._Y  = get_qcd_state(H )
            self._Y1 = get_qcd_state(H1)
        except:
            raise(ValueError('The charges of {} and {} must be specified.'.format(H, H1)))

    def normalized_width(self, mS):
        return normalized_decay_width(self._Y, self._Y1, mS)
