# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..api import channel as ch
from ..production import two_body_hadronic as hh
from ..decay import leptonic as lp
from ..decay import two_pions as pi
from ..decay import two_gluons as gg

def test_string():
    assert_equals(ch._to_channel_str('B', ['S', 'K*']), 'B -> S K*')
    assert_equals(ch._to_channel_str('S', ['c', 'cbar']), 'S -> c cbar')
    assert_equals(ch._from_channel_str('B -> S K*_0(700)'), ('B', ['S', 'K*_0(700)']))
    assert_raises(ValueError, lambda: ch._from_channel_str('e+ e- -> t tbar'))
    assert_raises(ValueError, lambda: ch._from_channel_str('B -> S K* -> S K gamma'))

def test_leptonic():
    ch = lp.Leptonic('mu')
    mS = np.array([0.1, 0.5, 1, 5, 10])
    assert(np.all(ch.normalized_width(mS) == lp.normalized_decay_width('mu', mS)))
    assert(np.all(ch.width(mS, 0.25) == 0.25**2 * lp.normalized_decay_width('mu', mS)))
    assert(np.all(ch.is_open(mS) == [False, True, True, True, True]))
    assert(np.all(ch.is_valid(mS)))
    assert_equals(ch.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 0 -13 13')
    assert_raises(ValueError, lambda: lp.Leptonic("tau'"))
    assert_raises(ValueError, lambda: lp.Leptonic('pi0' ))

def test_two_pions():
    ch0 = pi.TwoPions('neutral')
    ch1 = pi.TwoPions('charged')
    mS = np.array([0.1, 0.25, 0.3, 1])
    assert(np.all(ch0.normalized_width(mS) == pi.normalized_decay_width('neutral', mS)))
    assert(np.all(ch1.normalized_width(mS) == pi.normalized_decay_width('charged', mS)))
    assert(np.all(ch0.is_open(mS) == [False, False, True, True]))
    assert(np.all(ch0.is_valid(mS)))
    assert(np.all(~ch0.is_valid([1.5, 2, 5])))
    assert_equals(ch0.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 0 111 111' )
    assert_equals(ch1.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 0 211 -211')
    assert_raises(ValueError, lambda: pi.TwoPions('pi0 pi0'))

def test_two_gluons():
    ch = gg.TwoGluons()
    mS = np.array([2, 3, 5, 10])
    assert(np.all(ch.normalized_width(mS) == gg.normalized_decay_width(mS)))
    assert(np.all(ch.is_open(mS)))
    assert(np.all(ch.is_valid(mS)))
    assert(np.all(~ch.is_valid([0.5, 1, 1.5])))
    assert_equals(ch.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 91 21 21')

def test_hadronic_production():
    ch = hh.TwoBodyHadronic('B+', 'K*+')
    mS = np.array([0, 0.1, 0.5, 1, 2, 3, 5])
    assert(np.all(ch.normalized_width(mS) == hh.normalized_decay_width('B', 'K*', mS)))
    assert(np.all(ch.width(mS, 0.25) == 0.25**2 * hh.normalized_decay_width('B', 'K*', mS)))
    assert(np.all(ch.is_open(mS) == [True, True, True, True, True, True, False]))
    assert(np.all(ch.is_valid(mS)))
    assert_equals(ch.pythia_string(0.42, 9900025), '521:addChannel = 1 0.42 0 9900025 323')
    assert_raises(ValueError, lambda: hh.TwoBodyHadronic('B+', 'e+' ))
    assert_raises(ValueError, lambda: hh.TwoBodyHadronic('B+', 'K*' ))
    assert_raises(ValueError, lambda: hh.TwoBodyHadronic('B' , 'K*0'))

def test_vectorization():
    # Check that lists are handled as well as NumPy arrays
    def check_vectorization(channel, mS):
        assert(np.all(channel.normalized_width(mS) == \
                      channel.normalized_width(np.asarray(mS, dtype='float'))))
        for m in mS:
            assert(channel.normalized_width(m) == \
                   channel.normalized_width(np.asarray(m, dtype='float')))
    mS = [0.1, 0.5, 1, 2, 5, 10]
    check_vectorization(hh.TwoBodyHadronic('B+', 'pi+'        ), mS)
    check_vectorization(hh.TwoBodyHadronic('B+', 'K+'         ), mS)
    check_vectorization(hh.TwoBodyHadronic('B+', 'K*_0(700)+' ), mS)
    check_vectorization(hh.TwoBodyHadronic('B+', 'K*+'        ), mS)
    check_vectorization(hh.TwoBodyHadronic('B+', 'K_1(1270)+' ), mS)
    check_vectorization(hh.TwoBodyHadronic('B+', 'K*_2(1430)+'), mS)
    check_vectorization(lp.Leptonic('e')                       , mS)
    mS = [0.1, 0.5, 1]
    check_vectorization(pi.TwoPions('neutral'), mS)
    mS = [2, 3, 5]
    check_vectorization(gg.TwoGluons()   , mS)
    # check_vectorization(qq.TwoQuarks('c'), mS) # TODO
