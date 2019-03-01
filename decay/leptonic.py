# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

from ..data.constants import *
from ..data.particles import *
from ..api.channel import DecayChannel


def normalized_decay_width(l, mS):
    """
    Computes the decay width for the leptonic decay process S -> l⁺l⁻.
    """
    if not is_lepton(l):
        raise(ValueError('{} must be a lepton.'.format(l)))
    ml = get_mass(l)
    with np.errstate(invalid='ignore', divide='ignore'):
        w = ( (ml**2 * mS) / (8*pi * v**2) ) * ( 1 - 4*ml**2/mS**2 )**(3/2)
    threshold = 2*ml
    return np.where(mS > threshold, w, 0.)


class LeptonicDecayChannel(DecayChannel):
    '''
    Leptonic decay channel 'S -> l+ l-'.
    '''
    def __init__(self, flavor):
        super(LeptonicDecayChannel, self).__init__([flavor+'+', flavor+'-'])
        self._flavor = flavor

    def normalized_width(self, mS):
        return normalized_decay_width(self._flavor, mS)
