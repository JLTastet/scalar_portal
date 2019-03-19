# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import scipy.interpolate as si

from ..api.channel import DecayChannel


_srcdir = os.path.dirname(__file__)
# Load the table containing the normalized S -> K K decay width.
#   Source:
#       Winkler, M.W., 2019.
#       Decay and Detection of a Light Scalar Boson Mixing with the Higgs.
#       Physical Review D 99. https://doi.org/10.1103/PhysRevD.99.015018
#   First column: scalar mass m_S in GeV.
#   Second column: Γ(S -> K K) ratio evaluated at m_S.
#   Note: The first column should not be assumed to be increasing.
_decay_width_table = np.loadtxt(os.path.join(_srcdir, 'winkler_K.txt'))
# The calculation is not valid above 2.0 GeV, so we return NaN above this value.
_upper_lim = 2.0 # GeV
# Make sure the upper limit makes sense
assert(np.max(_decay_width_table[:,0]) >= _upper_lim)

# Workaround for SciPy 0.15.1
_itp = si.interp1d(
    _decay_width_table[:,0], _decay_width_table[:,1],
    kind='linear',
    bounds_error=False,
    fill_value=0.,
    assume_sorted=False
)

def normalized_total_width(mS):
    """
    Total decay width Γ(S → K K) = Γ(S → K⁰ Kbar⁰) + Γ(S → K⁺ K⁻).
    """
    return np.where(mS <= _upper_lim, _itp(mS), float('nan'))

def normalized_decay_width(mS):
    """
    Decay width to two kaons, for a specific final state.
        Γ(S → K⁰ Kbar⁰) = 1/2×Γ(S → K K)
        Γ(S → K⁺ K⁻   ) = 1/2×Γ(S → K K)
    """
    return (1/2) * normalized_total_width(np.asarray(mS, dtype='float'))

class TwoKaons(DecayChannel):
    '''
    Decay channel 'S -> K⁰ Kbar⁰' or 'S -> K⁺ K⁻'.
    '''
    def __init__(self, final_state):
        if final_state == 'neutral':
            children = ['K0', 'Kbar0']
        elif final_state == 'charged':
            children = ['K+', 'K-'   ]
        else:
            raise(ValueError("Final state must be either 'neutral' (K0 Kbar0) or 'charged' (K+ K-)."))
        super(TwoKaons, self).__init__(children)

    def normalized_width(self, mS):
        return normalized_decay_width(mS)
