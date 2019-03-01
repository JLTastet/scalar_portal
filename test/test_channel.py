# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..api import channel as ch
from ..decay import leptonic as lp

def test_string():
    assert_equals(ch._to_channel_str('B', ['S', 'K*']), 'B -> S K*')
    assert_equals(ch._to_channel_str('S', ['c', 'cbar']), 'S -> c cbar')
    assert_equals(ch._from_channel_str('B -> S K*_0(700)'), ('B', ['S', 'K*_0(700)']))
    assert_raises(ValueError, lambda: ch._from_channel_str('e+ e- -> t tbar'))
    assert_raises(ValueError, lambda: ch._from_channel_str('B -> S K* -> S K gamma'))

def test_leptonic():
    ch = lp.LeptonicDecayChannel('mu')
    mS = np.array([0.1, 0.5, 1, 5, 10])
    assert(np.all(ch.normalized_width(mS) == lp.normalized_decay_width('mu', mS)))
    assert(np.all(ch.width(mS, 0.25) == 0.25**2 * lp.normalized_decay_width('mu', mS)))
    assert(np.all(ch.is_open(mS) == [False, True, True, True, True]))
    assert_equals(ch.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 0 -13 13')
