# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

from ..data.constants import *
from ..data.particles import *
from ..api.channel import ProductionChannel
from .two_body_hadronic import xi

import numpy as np


# Heavy meson decay constants in GeV
_decay_constants = {
    'B'  : 0.19,
    'B_s': 0.23,
    'K'  : 0.16,
}

def _get_decay_constant(X):
    try:
        return _decay_constants[X]
    except KeyError:
        raise(ValueError('Decay constant not found for {}.'.format(X)))

_quark_transitions = {
    (0, 0, 1): ('D', 3, 1),
    (1, 0, 1): ('D', 3, 2),
    (1, 0, 0): ('D', 2, 1),
}

def _get_quark_transitions(X):
    S, C, B = get_abs_strangeness(X), get_abs_charm(X), get_abs_beauty(X)
    assert(S >= 0 and C == 0 and B >= 0)
    return _quark_transitions[(S, C, B)]

def _get_xi(X):
    return xi(*_get_quark_transitions(X))

def _available_mass(X):
    return get_mass(X) / 2

def normalized_decay_width(X, mS):
    '''
    Computes the decay width for the process X -> S S, divided by the
    coefficient α.
    '''
    mS = np.asarray(mS, dtype='float')
    mX = get_mass(X)
    fX = _get_decay_constant(X)
    xi_Q = _get_xi(X)
    with np.errstate(invalid='ignore'):
        beta = np.sqrt(1 - (2*mS/mX)**2)
    w = (mX**3 / v**2) * (xi_Q**2 * fX**2) / (128*pi * M_h**4) * beta
    return np.where(mS < _available_mass(X), w, 0.)


class TwoBodyQuartic(ProductionChannel):
    '''
    Quartic scalar production from the decay of heavy neutral hadrons: H -> S S.

    If the parent particle is not a weak eigenstate (e.g. due to neutral kaon
    mixing), then a weak eigenstate (either particle or antiparticle) must be
    specified separately.
    '''
    def __init__(self, H, weak_eigenstate=None):
        if weak_eigenstate is None:
            weak_eigenstate = H
        if not is_meson(weak_eigenstate):
            raise(ValueError('{} is not a meson.'.format(weak_eigenstate)))
        super(TwoBodyQuartic, self).__init__(H, [], NS=2)
        self._X = get_qcd_state(weak_eigenstate)
        if get_charge(weak_eigenstate) != 0:
            raise(ValueError('X -> S S is only possible if X is neutral (X = {}).'
                             .format(weak_eigenstate)))

    def normalized_width(self, mS):
        return normalized_decay_width(self._X, mS)
