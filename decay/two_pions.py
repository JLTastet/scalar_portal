# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import scipy.interpolate as si

from ..api.channel import DecayChannel


_srcdir = os.path.dirname(__file__)
# Load the table containing the normalized S -> π π decay width.
#   Source:
#       Winkler, M.W., 2019.
#       Decay and Detection of a Light Scalar Boson Mixing with the Higgs.
#       Physical Review D 99. https://doi.org/10.1103/PhysRevD.99.015018
#   First column: scalar mass m_S in GeV.
#   Second column: Γ(S -> π π) ratio evaluated at m_S.
#   Note: The first column should not be assumed to be increasing.
_decay_width_table = np.loadtxt(os.path.join(_srcdir, 'winkler_pi.txt'))
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
    Total decay width Γ(S → π π) = Γ(S → π⁰ π⁰) + Γ(S → π⁺ π⁻).
    """
    return np.where(mS <= _upper_lim, _itp(mS), float('nan'))

def normalized_decay_width(final_state, mS):
    """
    Decay width to two pions, for a specific final state.
    Possible values for `final_state`:
        `neutral`: Γ(S → π⁰ π⁰) = 1/3×Γ(S → π π)
        `charged`: Γ(S → π⁺ π⁻) = 2/3×Γ(S → π π)
    """
    if final_state == 'neutral':
        fraction = 1/3
    elif final_state == 'charged':
        fraction = 2/3
    else:
        raise(ValueError('Unknown final state {}.'.format(final_state)))
    return fraction * normalized_total_width(np.asarray(mS, dtype='float'))


class TwoPions(DecayChannel):
    '''
    Decay channel 'S -> π⁰ π⁰' or 'S -> π⁺ π⁻'.
    '''
    def __init__(self, final_state):
        if final_state == 'neutral':
            children = ['pi0', 'pi0']
        elif final_state == 'charged':
            children = ['pi+', 'pi-']
        else:
            raise(ValueError("Final state must be either 'neutral' (2 pi0) or 'charged' (pi+ pi-)."))
        super(TwoPions, self).__init__(children)
        self._final_state = final_state

    def normalized_width(self, mS):
        return normalized_decay_width(self._final_state, mS)
