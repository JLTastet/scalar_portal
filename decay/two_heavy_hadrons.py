# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import numpy as np

from ..data.particles import *
from ..api.channel import DecayChannel, format_pythia_string
from ..decay.two_quarks import TwoQuarks

class TwoHeavyHadrons(DecayChannel):
    '''
    Decay channel 'S -> H Hbar'.
    '''
    def __init__(self, quark_flavor, H, Hbar, fragmentation_fraction):
        super(TwoHeavyHadrons, self).__init__([H, Hbar])
        self.quarks_decay_channel = TwoQuarks(quark_flavor)
        self.fragmentation_fraction = fragmentation_fraction

    def normalized_width(self, mS):
        return self.fragmentation_fraction * self.quarks_decay_channel.normalized_width(mS)
