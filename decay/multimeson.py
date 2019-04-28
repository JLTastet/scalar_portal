# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import numpy as np

from ..api.channel import DecayChannel
from ..data.particles import get_mass
from . import two_pions  as pp
from . import two_kaons  as kk
from . import two_quarks as qq
from . import two_gluons as gg


# Scale at which we match the toy model to the perturbative QCD result.
_Lambda_S_pert = 2.0 # GeV

# Threshold above which the multimeson channel opens.
# This corresponds to the S -> 4π threshold.
_m_th = 2 * get_mass('pi')

# The contribution from the multimeson channel is identically zero below m_th
# and above Λ_S^pert.

def _beta(mS):
    'Velocity factor in our toy model.'
    return (1 - 4*(_m_th/mS)**2)**(1/2)

# The phenomenological coefficient C used in the toy model.
# It is computed by matching the toy model to the perturbative QCD result at
# the scale Λ_S^pert.
_partial_width_below = (
    pp.normalized_total_width(_Lambda_S_pert) +
    kk.normalized_total_width(_Lambda_S_pert)
)
_partial_width_above = (
    gg.normalized_total_width(_Lambda_S_pert) +
    qq.normalized_total_width(_Lambda_S_pert)
)
_C = ( (_partial_width_above - _partial_width_below)
       / (_Lambda_S_pert**3 * _beta(_Lambda_S_pert)) )

def _normalized_decay_width(mS):
    return _C * mS**3 * _beta(mS)

def normalized_decay_width(mS):
    mS = np.asarray(mS, dtype='float')
    open_ch = mS > 2 * _m_th
    valid = mS <= _Lambda_S_pert
    w = np.zeros_like(mS)
    w[open_ch] = _normalized_decay_width(mS[open_ch])
    w[~valid] = float('nan')
    return w

def normalized_total_width(mS):
    return normalized_decay_width(mS)


class Multimeson(DecayChannel):
    '''
    Toy model for multi-meson channels (4π, ηη, ρρ, …)
    '''
    def __init__(self):
        super(Multimeson, self).__init__(4 * ['pi'])

    def __str__(self):
        return 'S -> mesons...'

    def normalized_width(self, mS):
        return normalized_decay_width(mS)

    def pythia_string(self, branching_ratio, scalar_id):
        return None
