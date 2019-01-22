# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import numpy as np

from ..data.constants import *
from ..data.qcd_rg import *

def _y_q(q, mS):
    mq = get_quark_mass(q, mu=2.0) # TODO: check whether μ = 2 GeV is correct here.
    return 4*(mq/mS)**2

def _x_q(y_q):
    x_q = np.zeros_like(y_q, dtype='complex')
    pos = y_q > 1
    x_q[ pos] = np.arctan(1 / np.sqrt(y_q[pos] - 1))
    x_q[~pos] = 1/2 * (pi + 1j * np.log( (1+np.sqrt(1-y_q[~pos])) / (1-np.sqrt(1-y_q[~pos])) ))
    return x_q

def _F_q(q, mS):
    y_q = _y_q(q, mS)
    x_q = _x_q(y_q)
    return -2*y_q*(1+(1-y_q)*x_q**2)

def _F(mS):
    return sum(_F_q(q, mS) for q in all_quarks)

_lower_validity_bound = 2.0 # GeV

def normalized_decay_width(mS):
    """
    Computes the decay width into gluons: S -> g g.

    This computation is only valid above 2 GeV, and will return NaNs below.
    """
    F = _F(mS)
    # We have to use the pole mass for the top quark.
    # TODO: check whether μ = m_S is correct here for α_s
    w = F*np.conj(F) * (alpha_s(mS)/(4*pi))**2 * (mS**3/(8*pi*v**2)) * (1 + m_t_pole**2/(8*v**2*pi**2))
    return np.where(mS >= _lower_validity_bound, w, float('nan'))
