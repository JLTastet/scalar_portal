# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..production.two_body_hadronic import *

def test_two_body_hadronic_amplitude():
    mS = np.array([0, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0])
    A = normalized_amplitude('K', 'pi'        , mS)
    assert(all(np.isfinite(A)))
    assert(all(A[mS >= get_mass('K') - get_mass('pi')] == 0))
    A = normalized_amplitude('B', 'pi'        , mS)
    assert(all(np.isfinite(A)))
    assert(all(A[mS >= get_mass('B') - get_mass('pi')] == 0))
    A = normalized_amplitude('B', 'K'         , mS)
    A = normalized_amplitude('B', 'K*'        , mS)
    A = normalized_amplitude('B', 'K*(1410)'  , mS)
    A = normalized_amplitude('B', 'K_1(1270)' , mS)
    A = normalized_amplitude('B', 'K*_0(700)' , mS)
    A = normalized_amplitude('B', 'K*_2(1430)', mS)
    assert(all(np.isfinite(A)))
    assert(all(A[mS >= get_mass('B') - get_mass('K*_2(1430)')] == 0))
    assert_raises(ValueError, lambda: normalized_amplitude('K', 'B', mS))

def test_two_body_hadronic_width():
    mS = np.array([0, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0])
    w = normalized_decay_width('K', 'pi'        , mS)
    assert(all(np.isfinite(w)))
    assert(all(w[mS >= get_mass('K') - get_mass('pi')] == 0))
    w = normalized_decay_width('B', 'pi'        , mS)
    assert(all(np.isfinite(w)))
    assert(all(w[mS >= get_mass('B') - get_mass('pi')] == 0))
    w = normalized_decay_width('B', 'K'         , mS)
    w = normalized_decay_width('B', 'K*'        , mS)
    w = normalized_decay_width('B', 'K*(1410)'  , mS)
    w = normalized_decay_width('B', 'K_1(1270)' , mS)
    w = normalized_decay_width('B', 'K*_0(700)' , mS)
    w = normalized_decay_width('B', 'K*_2(1430)', mS)
    assert(all(np.isfinite(w)))
    assert(all(w[mS >= get_mass('B') - get_mass('K*_2(1430)')] == 0))
    assert_raises(ValueError, lambda: normalized_decay_width('K', 'B', mS))
