# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..data.particles import *
from ..decay import leptonic as lp
from ..decay import two_pions as tp

def test_leptonic_width():
    mS = np.array([0, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0])
    w = lp.normalized_decay_width('e'  , mS)
    assert(all(w[mS > 2*get_mass('e'  )] >  0))
    assert(all(w[mS < 2*get_mass('e'  )] == 0))
    w = lp.normalized_decay_width('mu' , mS)
    assert(all(w[mS > 2*get_mass('mu' )] >  0))
    assert(all(w[mS < 2*get_mass('mu' )] == 0))
    w = lp.normalized_decay_width('tau', mS)
    assert(all(w[mS > 2*get_mass('tau')] >  0))
    assert(all(w[mS < 2*get_mass('tau')] == 0))
    assert_raises(ValueError, lambda: lp.normalized_decay_width('K', mS))

def test_two_pion_width():
    mS = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 2.0, 4.0, 10.0])
    w = tp.normalized_decay_width('neutral', mS)
    assert(all(w[mS < 2*get_mass('pi')]) == 0)
    assert(all(w[np.logical_and(mS > 2*get_mass('pi'), mS < 1.0)]) > 0)
    assert(all(np.isnan(w[mS > 1.0])))
    w = tp.normalized_decay_width('charged', mS)
    assert(all(w[mS < 2*get_mass('pi')]) == 0)
    assert(all(w[np.logical_and(mS > 2*get_mass('pi'), mS < 1.0)]) > 0)
    assert(all(np.isnan(w[mS > 1.0])))
    assert_raises(ValueError, lambda: tp.normalized_decay_width('test', mS))
