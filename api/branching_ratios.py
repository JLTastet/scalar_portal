# -*- coding: utf-8 -*-

from __future__ import absolute_import, division
from future.utils import viewitems, with_metaclass

import numpy as np

from ..api.channel import Channel
from ..data.constants import second, default_scalar_id


class BranchingRatios(object):
    '''
    Represents a set of computed branching ratios.
    '''
    def __init__(self, channels, mass, coupling,
                 ignore_nan=False,
                 scalar_id=default_scalar_id):
        self._channels = {str(ch): ch for ch in channels}
        self._mS = np.asarray(mass, dtype='float')
        self._coupling = np.asarray(coupling, dtype='float')
        self._scalar_id = scalar_id
        self._width = {str(ch): ch.width(mass, coupling) for ch in channels}
        if ignore_nan:
            for w in self._width.values():
                w[np.isnan(w)] = 0

    @property
    def width(self):
        return self._width


class DecayBranchingRatios(BranchingRatios):
    '''
    Represents a set of decay branching ratios for the scalar.
    '''
    def __init__(self, *args, **kwargs):
        super(DecayBranchingRatios, self).__init__(*args, **kwargs)
        self._total_width = sum(self._width.values())
        self._br = {
            ch_str: w / self._total_width for ch_str, w in viewitems(self._width)}

    @property
    def total_width(self):
        return self._total_width

    @property
    def lifetime_si(self):
        tau = 1 / self._total_width
        return tau / second

    @property
    def branching_ratios(self):
        return self._br

    def pythia_strings(self):
        if self._mS.ndim > 0 or self._coupling.ndim > 0:
            raise(ValueError('Can only generate PYTHIA strings for a single mass and coupling.'))
        return {ch_str: channel.pythia_string(self._br[ch_str], self._scalar_id)
                for ch_str, channel in viewitems(self._channels)}
