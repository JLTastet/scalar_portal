# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import numpy as np

from ..data.constants import *
from ..data.particles import *

def _beta(mq, mS):
    return (1 - 4*(mq/mS)**2)**(1/2)

def _Delta_QCD(aS, Nf):
    x = aS/pi
    return 5.67*x + (35.94-1.36*Nf)*x**2 + (164.14-25.77*Nf+0.259*Nf**2)*x**3

def _Delta_t(aS, mq, mS):
    # We have to use the pole mass for the top quark.
    mt = on_shell_mass('t')
    return (aS/pi)**2 * (1.57 - (4/3)*np.log(mS/mt) + (4/9)*np.log(mq/mS)**2)

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

def normalized_decay_width(q, mS):
    """
    Computes the decay width into two quarks: S -> q qbar, for q ∈ {s, c}.

    This computation is only valid above 2 GeV, away from the c threshold, and
    below the b threshold. It will return NaNs outside this range.
    """
    if q not in ['s', 'c']:
        raise(ValueError('S -> {} {}bar not implemented.'.format(q, q)))
    w = np.zeros_like(mS)
    valid = (mS >= _lower_validity_bound) & (mS < _thresholds['b'])
    w[~valid] = np.nan
    open_channels = valid & (mS >= _thresholds[q])
    # Only do the calculation for open channels
    mS_open = np.array(mS)[open_channels]
    mq = msbar_mass(q, mu=mS_open, nf=_Nf)
    aS = alpha_s(mu=mS_open, nf=_Nf)
    # We set β=1, since quarks are assumed to be massless in this calculation.
    w[open_channels] = 3*mS_open*mq**2/(8*pi*v**2) \
        * (1 + _Delta_QCD(aS, _Nf) + _Delta_t(aS, mq, mS_open))
    return w
