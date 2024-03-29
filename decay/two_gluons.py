# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import numpy as np

from ..data.constants import *
from ..data.particles import *
from ..api.channel import DecayChannel, format_pythia_string


# Number of dynamical flavors
# This calculation is only used above 2 GeV, so we are above the c threshold.
# We are also below the b threshold, since the heaviest scalar which can be
# probed at SHiP is produced in B decays.
_nf = 4
# u, d, s are assumed to be massless.
_heavy_quarks = ['c', 'b', 't']

def _y_q(q, mS):
    # We need to use the pole mass here! Cf. p.215 in Spira.
    # This only makes sense for heavy quarks (c, b, t). Light quarks (u, d, s)
    # can safely be assumed to be massless, since they give negligible
    # contributions anyway.
    mq = on_shell_mass(q)
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
    return sum(_F_q(q, mS) for q in _heavy_quarks)

def _E(mS, mu, Nf):
    '''
    NLO correction from gluon real emission and splitting.
    Cf. Eq. (23) from hep-ph/9504378 (Spira et al.)
    ΔE is neglected (it vanishes in the limit of large loop masses).
    '''
    return 95/4 - 7/6*Nf + (33-2*Nf)/6 * 2*np.log(mu/mS) # + ΔE ≈ 0

_lower_validity_bound = 2.0 # GeV
_upper_validity_bound = get_mass('B') # The b quark becomes dynamical above this threshold

def normalized_decay_width(mS):
    """
    Computes the decay width into gluons at NLO: S -> gg(g), gqq̄.

    This computation is only valid above 2 GeV, and will return NaNs below.

    Notes:
    The intrinsic uncertainty on the charm pole mass introduces a relative
    uncertainty corresponding to a factor of about 2 between the lowest and
    highest possible decay widths.

    Since this calculation is done at next-to-leading order, we have used the 1-loop
    formula to obtain the on-shell masses of c and b from the MSbar masses.

    The 1-loop contribution from the top quark is only 0.3% and is therefore neglected.
    """
    mS = np.asarray(mS, dtype='float')
    valid = (mS >= _lower_validity_bound) & (mS < _upper_validity_bound)
    F = _F(mS[valid])
    w = np.zeros_like(mS, dtype='float')
    w[~valid] = float('nan')
    if np.any(valid):
        aS = alpha_s(mu=mS[valid], nf=_nf)
        # Compute the NLO correction from the real emissions and splitting of gluons.
        E = _E(mS[valid], mS[valid], _nf)
        # Evaluate the width
        w[valid] = np.real(F*np.conj(F)) * (aS/(4*pi))**2 \
            * (mS[valid]**3/(8*pi*v**2)) * (1 + (aS/pi)*E)
    return w

def normalized_total_width(mS):
    return normalized_decay_width(mS)


class TwoGluons(DecayChannel):
    '''
    Decay channel 'S -> g g'.
    '''
    def __init__(self):
        super(TwoGluons, self).__init__(2 * ['g'])

    def normalized_width(self, mS):
        return normalized_decay_width(mS)

    def pythia_string(self, branching_ratio, scalar_id):
        return format_pythia_string(
            scalar_id, 2 * [get_pdg_id('g')], branching_ratio,
            matrix_element=pythia_me_mode_hadronize)
