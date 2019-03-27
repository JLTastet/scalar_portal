# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import numpy as np

from ..data.constants import *
from ..data.particles import *
from ..api.channel import DecayChannel, format_pythia_string


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
    Approximates the decay width of S -> q qbar above the Hq Hq threshold
    (where Hq=K for q=s, D for c, B for b).
    """
    mq = msbar_mass(q, mu=mS, nf=_Nf)
    aS = alpha_s(mu=mS, nf=_Nf)
    # It seems that Spira forgot the β³ in the paper, but it is needed to
    # reproduce figure 4, on page 213, so we put it back.
    # Moreover, to get the correct threshold in the full QCD, it makes sense to
    # replace the phase-space factor β(m_q) (obtained from pQCD) with β(m_Hq).
    # Finally, since the lightest open-flavor state is in S wave, unlike q-qbar
    # which is in P wave, we should use β¹(m_Hq) instead of β³(m_Hq).
    beta = _beta(_thresholds[q]/2, mS)
    w = 3*mS*mq**2/(8*pi*v**2) * beta * (1 + _Delta_QCD(aS, _Nf) + _Delta_t(aS, mq, mS))
    return w

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
    if np.any(open_channels):
        w[open_channels] = _normalized_decay_width_large_mass(q, mS_open)
    return w

def normalized_total_width(mS):
    return sum(normalized_decay_width(q, mS) for q in ['s', 'c'])


class TwoQuarks(DecayChannel):
    '''
    Decay channel 'S -> q qbar'.
    '''
    def __init__(self, flavor):
        if not flavor in ['s', 'c']:
            raise(ValueError('S -> {} {}bar not implemented.'.format(flavor, flavor)))
        super(TwoQuarks, self).__init__([flavor, flavor+'bar'])
        self._q = flavor

    def normalized_width(self, mS):
        return normalized_decay_width(self._q, mS)

    def pythia_string(self, branching_ratio, scalar_id):
        id_q = get_pdg_id(self._q)
        return format_pythia_string(
            scalar_id, [id_q, -id_q], branching_ratio,
            matrix_element=pythia_me_mode_hadronize)

    def is_open(self, mS):
        return mS > _thresholds[self._q]
