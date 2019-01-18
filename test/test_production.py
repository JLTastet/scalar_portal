# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..production.two_body_hadronic import *

def test_two_body_hadronic_amplitude():
    # mS = np.array([0, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0]) # FIXME: handle mS > threshold
    mS = np.array([0, 0.1, 0.5])
    A = normalized_amplitude('K', 'pi'        , mS)
    mS = np.array([0, 0.1, 0.5, 1.0, 2.0])
    A = normalized_amplitude('B', 'pi'        , mS)
    A = normalized_amplitude('B', 'K'         , mS)
    A = normalized_amplitude('B', 'K*'        , mS)
    A = normalized_amplitude('B', 'K*(1410)'  , mS)
    A = normalized_amplitude('B', 'K_1(1270)' , mS)
    A = normalized_amplitude('B', 'K*_0(700)' , mS)
    A = normalized_amplitude('B', 'K*_2(1430)', mS)
    assert_raises(ValueError, lambda: normalized_amplitude('K', 'B', mS))
