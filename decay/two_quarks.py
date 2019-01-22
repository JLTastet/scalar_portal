# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import numpy as np

from ..data.constants import *
from ..data.qcd_rg import *

def _beta(mq, mS):
    return (1 - 4*(mq/mS)**2)**(1/2)

def _Delta_QCD(aS, Nf):
    x = aS/pi
    return 5.67*x + (35.94-1.36*Nf)*x**2 + (164.14-25.77*Nf+0.259*Nf**2)*x**3

def _Delta_t(aS, mq, mS):
    # We have to use the pole mass for the top quark.
    return (aS/pi)**2 * (1.57 - (4/3)*np.log(mS/m_t_pole) + (4/9)*np.log(mq/mS)**2)

_lower_validity_bound = 2.0 # GeV

_Nf = 3 # Number of light quark flavors.

def normalized_decay_width(q, mS):
    """
    Computes the decay width into two quarks: S -> q qbar, for q ∈ {s, c}.

    This computation is only valid above 2 GeV, and will return NaNs below.
    """
    if q not in ['s', 'c']:
        raise(ValueError('S -> {} {}bar not implemented.'.format(q, q)))
    # Here, NaNs are meaningful: they are produced for masses for which the computation is invalid.
    # So there is no need for NumPy to print creepy warnings.
    with np.errstate(invalid='ignore'):
        mq = get_quark_mass(q, mu=mS)
        aS = alpha_s(mS)
        w = 3*mS*mq**2/(8*pi*v**2) * _beta(mq, mS)**3 * (1 + _Delta_QCD(aS, _Nf) + _Delta_t(aS, mq, mS))
        open_channel = mS > 2*mq
        res = np.zeros_like(w)
        res[open_channel] = w[open_channel]
        res[mS < _lower_validity_bound] = float('nan')
    return res
