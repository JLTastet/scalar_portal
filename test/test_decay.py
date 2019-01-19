# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..data.particles import *
from ..decay import leptonic as lp

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
