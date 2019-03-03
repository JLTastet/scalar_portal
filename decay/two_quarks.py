# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import numpy as np
from scipy.special import spence

from ..data.constants import *
from ..data.particles import *

def _beta(mq, mS):
    """
    Perturbative "velocity" of the two outgoing quarks.
    """
    return (1 - 4*(mq/mS)**2)**(1/2)

def _Delta_QCD(aS, Nf):
    "QCD corrections away from the threshold."
    x = aS/pi
    return 5.67*x + (35.94-1.36*Nf)*x**2 + (164.14-25.77*Nf+0.259*Nf**2)*x**3

def _Delta_t(aS, mq, mS):
    "QCD correction arising from the top triangle, away from the threshold."
    # We have to use the pole mass for the top quark.
    mt = on_shell_mass('t')
    return (aS/pi)**2 * (1.57 - (4/3)*np.log(mS/mt) + (4/9)*np.log(mq/mS)**2)

def Li2(z):
    "Dilogarithm, with the same definition as in FeynCalc and Spira's paper."
    return spence(1 - z)

def _A(b):
    return (
        (1+b**2) * (4*Li2((1-b)/(1+b)) + 2*Li2(-(1-b)/(1+b))
                    - 3*np.log((1+b)/(1-b))*np.log(2/(1+b))
                    - 2*np.log((1+b)/(1-b))*np.log(b))
        -3*b*np.log(4/(1-b**2)) - 4*b*np.log(b)
    )

def _Delta_H(b):
    "QCD correction factor near the threshold."
    return (_A(b)/b
            + (3 + 34*b**2 - 13*b**4) / (16*b**3) * np.log((1+b)/(1-b))
            + 3 * (7*b**2 - 1) / (8*b**2))

_lower_validity_bound = 2.0 # GeV

# Number of dynamical quarks
# Nf = 4 throughout the considered mass range.
_Nf = 4

# q qbar thresholds
_thresholds = {
    's': 2 * get_mass('K'),
    'c': 2 * get_mass('D'),
    'b': 2 * get_mass('B'),
}

def _normalized_decay_width_large_mass(q, mS):
    """
    Computes the decay width of S -> q qbar away from the threshold.
    """
    mq = msbar_mass(q, mu=mS, nf=_Nf)
    aS = alpha_s(mu=mS, nf=_Nf)
    # It seems that Spira forgot the β³ in the paper, but it is needed to
    # reproduce figure 4, on page 213, so we put it back.
    beta = _beta(mq, mS)
    w = 3*mS*mq**2/(8*pi*v**2) * beta**3 * (1 + _Delta_QCD(aS, _Nf) + _Delta_t(aS, mq, mS))
    return w

def _normalized_decay_width_near_threshold(q, mS):
    """
    Computes the decay width of S -> q qbar near the threshold.
    """
    try:
        Mq = on_shell_mass(q)
        aS = alpha_s(mu=mS, nf=_Nf)
        # FIXME: use the mass of the lightest stable hadron containing Q
        # Ref: arXiv:1310.8042, p. 2
        beta = _beta(Mq, mS)
        w = 3*mS*Mq**2/(8*pi*v**2) * beta**3 * (1 + (4/3)*(aS/pi)*_Delta_H(beta))
        return w
    except ValueError:
        return np.full_like(mS, np.nan)

def normalized_decay_width(q, mS):
    """
    Computes the decay width into two quarks: S -> q qbar, for q ∈ {s, c}.

    This computation is only valid above 2 GeV and below the b threshold. This
    function will return NaNs outside this range.
    """
    mS = np.asarray(mS, dtype='float')
    if q not in ['s', 'c']:
        raise(ValueError('S -> {} {}bar not implemented.'.format(q, q)))
    w = np.zeros_like(mS, dtype='float')
    valid = (mS >= _lower_validity_bound) & (mS < _thresholds['b'])
    w[~valid] = np.nan
    open_channels = valid & (mS >= _thresholds[q])
    # Only do the calculation for open channels
    mS_open = mS[open_channels]
    w_large_mass = _normalized_decay_width_large_mass(q, mS_open)
    w_threshold  = _normalized_decay_width_near_threshold(q, mS_open)
    # The physical decay width is closer to the minimum, so we return that.
    # If either calculation returns NaN's, we return the result of the other.
    w[open_channels] = np.fmin(w_large_mass, w_threshold)
    return w
