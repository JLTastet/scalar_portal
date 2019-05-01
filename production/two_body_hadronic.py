# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

from ..data.constants import *
from ..data.particles import *
from ..api.channel import ProductionChannel
from . import hadronic_common as h

import numpy as np


def _available_mass(Y, Y1):
    return get_mass(Y) - get_mass(Y1)

def normalized_decay_width(Y, Y1, mS):
    """
    Computes the decay width for the process Y_q -> S Y'_q', divided by the
    mixing angle θ.
    """
    kin_open = mS < _available_mass(Y, Y1)
    mS = np.asarray(mS, dtype='float')
    mu = mS
    M = h.get_matrix_element(Y, Y1)
    xi = h._get_xi(Y, Y1)
    _, Qi, _ = h._get_quark_transition(Y, Y1)
    mY  = get_mass(Y )
    mY1 = get_mass(Y1)
    pS = h._momentum(mY, mY1, mS)
    with np.errstate(invalid='ignore'):
        A = M(mS**2)
        w = ( xi**2 * np.real(A*np.conj(A)) * pS ) / ( 32*pi * v**2 * mY**2 )
    return np.where(kin_open, w, 0.)


class TwoBodyHadronic(ProductionChannel):
    '''
    Scalar production through exclusive 2-body hadronic decays: H -> S H'.

    If the parent particle is not a weak eigenstate (e.g. due to neutral kaon
    mixing), then a weak eigenstate (either particle or antiparticle) must be
    specified separately.
    '''
    def __init__(self, H, H1, weak_eigenstate=None):
        super(TwoBodyHadronic, self).__init__(H, [H1])
        if weak_eigenstate is None:
            weak_eigenstate = H
        if not (is_meson(weak_eigenstate) and is_meson(H1)):
            raise(ValueError('{} and {} must be mesons.'.format(weak_eigenstate, H1)))
        try:
            self._Y  = get_qcd_state(weak_eigenstate)
            self._Y1 = get_qcd_state(H1)
        except:
            raise(ValueError('The charges of {} and {} must be specified.'.format(weak_eigenstate, H1)))

    def normalized_width(self, mS):
        return normalized_decay_width(self._Y, self._Y1, mS)
