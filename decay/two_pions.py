# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import scipy.interpolate as si

from . import leptonic as lp
from ..api.channel import DecayChannel


_srcdir = os.path.dirname(__file__)
# Load the table containing the Br(S -> pi pi) / Br(S -> mu+ mu-) ratio.
#   Source: J. F. Donoghue, J. Gasser and H. Leutwyler, The Decay of a Light Higgs Boson,
#           Nucl. Phys. B343 (1990) 341
#   Extracted from Fig. 6.
#   First column: scalar mass m_S in GeV.
#   Second column: Br(S -> pi pi) / Br(S -> mu+ mu-) ratio evaluated at m_S.
#   Note: The first column should not be assumed to be increasing.
_pi_mu_ratio_table = np.loadtxt(os.path.join(_srcdir, 'pion_to_muon_ratio.dat'))
# The calculation is not valid above 1 GeV, so we return NaN above this value.
_upper_lim = 1.0 # GeV
_upper_idx = np.argmax(_pi_mu_ratio_table[:,0] >= _upper_lim)
# Make sure the upper limit makes sense
assert(_upper_idx != 0)
assert(_upper_lim <= _pi_mu_ratio_table[-1,0])

# FIXME: requires SciPy >= 0.17.0
# _pi_mu_ratio = si.interp1d(
#     _pi_mu_ratio_table[0:_upper_idx,0], _pi_mu_ratio_table[0:_upper_idx,1],
#     kind='cubic',
#     bounds_error=False,
#     fill_value=(0., float('nan')),
#     assume_sorted=False
# )

# FIXME: workaround for SciPy 0.15.1
_itp = si.interp1d(
    _pi_mu_ratio_table[0:_upper_idx+1,0], _pi_mu_ratio_table[0:_upper_idx+1,1],
    kind='linear',
    bounds_error=False,
    fill_value=0.,
    assume_sorted=False
)
def _pi_mu_ratio(mS):
    return np.where(mS <= _upper_lim, _itp(mS), float('nan'))

def _normalized_total_decay_width(mS):
    """
    Total decay width Γ(S → π π) = Γ(S → π⁰ π⁰) + Γ(S → π⁺ π⁻).
    """
    dimuon_width = lp.normalized_decay_width('mu', mS)
    return dimuon_width * _pi_mu_ratio(mS)

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
    return fraction * _normalized_total_decay_width(np.asarray(mS, dtype='float'))


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
