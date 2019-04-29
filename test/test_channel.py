# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..api import channel as ch
from ..production import two_body_hadronic as hh
from ..production import two_body_quartic as q2
from ..production import three_body_quartic as q3
from ..decay import leptonic as lp
from ..decay import two_pions as pi
from ..decay import two_kaons as kk
from ..decay import multimeson as mm
from ..decay import two_gluons as gg
from ..decay import two_quarks as qq

def test_string():
    assert_equals(ch._to_channel_str('B', ['S', 'K*']), 'B -> S K*')
    assert_equals(ch._to_channel_str('S', ['c', 'cbar']), 'S -> c cbar')
    assert_equals(ch._from_channel_str('B -> S K*_0(700)'), ('B', ['S', 'K*_0(700)']))
    assert_raises(ValueError, lambda: ch._from_channel_str('e+ e- -> t tbar'))
    assert_raises(ValueError, lambda: ch._from_channel_str('B -> S K* -> S K gamma'))

def test_lt():
    assert(lp.Leptonic('mu') < lp.Leptonic('e'))
    assert(not (lp.Leptonic('e') < lp.Leptonic('e')))
    assert(hh.TwoBodyHadronic('B0', 'K0') < hh.TwoBodyHadronic('B+', 'K+'))
    assert(not (hh.TwoBodyHadronic('B+', 'K+') < hh.TwoBodyHadronic('B0', 'K0')))
    assert(hh.TwoBodyHadronic('B0', 'K0') < lp.Leptonic('e'))

def test_leptonic():
    ch = lp.Leptonic('mu')
    mS = np.array([0.1, 0.5, 1, 5, 10])
    assert(np.all(ch.normalized_width(mS) == lp.normalized_decay_width('mu', mS)))
    assert(np.all(ch.width(mS, {'theta': 0.25}) == 0.25**2 * lp.normalized_decay_width('mu', mS)))
    assert(np.all(ch.is_open(mS) == [False, True, True, True, True]))
    assert(np.all(ch.is_valid(mS)))
    assert_equals(str(ch), 'S -> mu+ mu-')
    assert_equals(ch.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 0 -13 13')
    assert_raises(ValueError, lambda: lp.Leptonic("tau'"))
    assert_raises(ValueError, lambda: lp.Leptonic('pi0' ))

def test_two_pions():
    ch0 = pi.TwoPions('neutral')
    ch1 = pi.TwoPions('charged')
    mS = np.array([0.1, 0.25, 0.3, 1, 2])
    assert(np.all(ch0.normalized_width(mS) == pi.normalized_decay_width('neutral', mS)))
    assert(np.all(ch1.normalized_width(mS) == pi.normalized_decay_width('charged', mS)))
    assert(np.all(ch0.is_open(mS) == [False, False, True, True, True]))
    assert(np.all(ch0.is_valid(mS)))
    assert(np.all(~ch0.is_valid([2.5, 5])))
    assert_equals(str(ch0), 'S -> pi0 pi0')
    assert_equals(str(ch1), 'S -> pi+ pi-')
    assert_equals(ch0.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 0 111 111' )
    assert_equals(ch1.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 0 211 -211')
    assert_raises(ValueError, lambda: pi.TwoPions('pi0 pi0'))

def test_two_kaons():
    ch0 = kk.TwoKaons('neutral')
    ch1 = kk.TwoKaons('charged')
    mS = np.array([0.9, 1, 1.4, 2])
    assert(np.all(ch0.normalized_width(mS) == kk.normalized_decay_width(mS)))
    # The widths should be equal since we use the same mass for K0 and K+.
    assert(np.all(ch0.normalized_width(mS) == ch1.normalized_width(mS)))
    assert(np.all(ch0.is_open(mS) == [False, True, True, True]))
    assert(np.all(ch0.is_valid(mS)))
    assert(np.all(~ch0.is_valid([2.5, 5])))
    assert_equals(str(ch0), 'S -> K0 Kbar0')
    assert_equals(str(ch1), 'S -> K+ K-'   )
    assert_equals(ch0.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 0 311 -311')
    assert_equals(ch1.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 0 321 -321')
    assert_raises(ValueError, lambda: kk.TwoKaons('K+ K-'))

def test_multimeson():
    ch = mm.Multimeson()
    mS = np.array([0, 0.5, 0.6, 1, 1.4, 1.7, 2])
    assert(np.all(ch.normalized_width(mS) == mm.normalized_decay_width(mS)))
    assert(np.all(ch.is_open(mS) == [False, False, True, True, True, True, True]))
    assert(np.all(ch.is_valid(mS)))
    assert(np.all(~ch.is_valid([2.5, 4])))
    assert_equals(str(ch), 'S -> mesons...')
    assert_equals(ch.pythia_string(0.42, 9900025), None)

def test_two_gluons():
    ch = gg.TwoGluons()
    mS = np.array([2, 3, 5, 10])
    assert(np.all(ch.normalized_width(mS) == gg.normalized_decay_width(mS)))
    assert(np.all(ch.is_open(mS)))
    assert(np.all(ch.is_valid(mS)))
    assert(np.all(~ch.is_valid([0.5, 1, 1.5])))
    assert_equals(str(ch), 'S -> g g')
    assert_equals(ch.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 91 21 21')

def test_two_quarks():
    ch_s = qq.TwoQuarks('s')
    mS = np.array([2, 3, 4, 10])
    assert(np.all(ch_s.normalized_width(mS) == qq.normalized_decay_width('s', mS)))
    assert(np.all(ch_s.is_open(mS)))
    assert(np.all(ch_s.is_valid(mS)))
    assert(np.all(~ch_s.is_valid([0.5, 1, 1.5, 11])))
    assert_equals(str(ch_s), 'S -> s sbar')
    assert_equals(ch_s.pythia_string(0.42, 9900025), '9900025:addChannel = 1 0.42 91 3 -3')
    ch_c = qq.TwoQuarks('c')
    assert(np.all(ch_c.is_open(mS) == [False, False, True, True]))
    assert_raises(ValueError, lambda: qq.TwoQuarks('t'))

def test_hadronic_production():
    ch = hh.TwoBodyHadronic('B+', 'K*+')
    mS = np.array([0, 0.1, 0.5, 1, 2, 3, 5])
    assert(np.all(ch.normalized_width(mS) == hh.normalized_decay_width('B', 'K*', mS)))
    assert(np.all(ch.width(mS, {'theta': 0.25}) == 0.25**2 * hh.normalized_decay_width('B', 'K*', mS)))
    assert(np.all(ch.is_open(mS) == [True, True, True, True, True, True, False]))
    assert(np.all(ch.is_valid(mS)))
    assert_equals(str(ch), 'B+ -> S K*+')
    assert_equals(ch.pythia_string(0.42, 9900025), '521:addChannel = 1 0.42 0 9900025 323')
    assert_raises(ValueError, lambda: hh.TwoBodyHadronic('B+', 'e+' ))
    assert_raises(ValueError, lambda: hh.TwoBodyHadronic('B+', 'K*' ))
    assert_raises(ValueError, lambda: hh.TwoBodyHadronic('B' , 'K*0'))
    tau  = hh.get_lifetime('B+')
    wtot = 1 / tau
    epsilon = 1e-14
    assert(abs(ch.parent_width - wtot) <= epsilon * wtot)
    br1 = ch.normalized_branching_ratio(mS)
    w1  = ch.normalized_width(mS)
    assert(np.all(np.abs(wtot*br1 - w1) <= epsilon * w1))
    br = ch.branching_ratio(mS, {'theta': 0.25})
    w  = ch.width(mS, {'theta': 0.25})
    assert(np.all(np.abs(wtot*br - w) <= epsilon * w))

def test_quartic_2body():
    ch = q2.TwoBodyQuartic('B_s0')
    mS = np.array([0, 0.1, 0.5, 1, 2.5, 3, 10])
    assert(np.all(ch.normalized_width(mS) == q2.normalized_decay_width('B_s', mS)))
    assert(np.all(ch.is_open(mS) == [True, True, True, True, True, False, False]))
    assert(np.all(ch.is_valid(mS)))
    assert_equals(str(ch), 'B_s0 -> S S')
    assert_equals(ch.pythia_string(0.42, 9900025), '531:addChannel = 1 0.42 0 9900025 9900025')
    assert_raises(ValueError, lambda: q2.TwoBodyQuartic('Z'))
    assert_raises(ValueError, lambda: q2.TwoBodyQuartic('B+'))
    assert(np.all(ch.width(mS, {'alpha': 0.5}) == 0.5**2 * ch.normalized_width(mS)))
    assert(np.all(ch.branching_ratio(mS, {'alpha': 0.5}) == 0.5**2 * ch.normalized_branching_ratio(mS)))
    # Test internals
    assert_raises(ValueError, lambda: q2._get_decay_constant('D'))

def test_quartic_3body():
    ch = q3.ThreeBodyQuartic('B+', 'K+')
    mS = np.array([0, 0.5, 2.3, 2.5])
    assert(np.all(ch.normalized_width(mS) == q3.normalized_decay_width('B', 'K', mS)))
    assert(np.all(ch.is_open(mS) == [True, True, True, False]))
    assert(np.all(ch.is_valid(mS)))
    assert_equals(str(ch), 'B+ -> S S K+')
    # FIXME: find correct matrix element.
    assert_equals(ch.pythia_string(0.42, 9900025), '521:addChannel = 1 0.42 0 9900025 9900025 321')
    assert_raises(ValueError, lambda: q3.ThreeBodyQuartic('B+', 'e+' ))
    assert_raises(ValueError, lambda: q3.ThreeBodyQuartic('B+', 'K*' ))
    assert_raises(ValueError, lambda: q3.ThreeBodyQuartic('B' , 'K*0'))

def test_neutral_kaons():
    ch1 = hh.TwoBodyHadronic('K_L0', 'pi0', weak_eigenstate='K0')
    mS = np.array([0, 0.1, 0.5, 1, 2, 3, 5])
    assert(np.all(ch1.normalized_width(mS) == hh.normalized_decay_width('K', 'pi', mS)))
    ch2 = hh.TwoBodyHadronic('B0', 'K0')
    assert(np.all(ch2.normalized_width(mS) == hh.normalized_decay_width('B', 'K', mS)))
    ch3 = q2.TwoBodyQuartic('K_L0', weak_eigenstate='K0')
    assert(np.all(ch3.normalized_width(mS) == q2.normalized_decay_width('K', mS)))
    ch4 = q3.ThreeBodyQuartic('K_L0', 'pi0', weak_eigenstate='K0')
    assert(ch4.normalized_width(0.15) == q3.normalized_decay_width('K', 'pi', 0.15))

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
    check_vectorization(q2.TwoBodyQuartic('B0')                , mS)
    check_vectorization(lp.Leptonic('e')                       , mS)
    check_vectorization(q3.ThreeBodyQuartic('B+', 'K+'), [0, 1])
    mS = [0.1, 0.5, 1]
    check_vectorization(pi.TwoPions('neutral'), mS)
    mS = [2, 3, 5]
    check_vectorization(gg.TwoGluons()   , mS)
    check_vectorization(qq.TwoQuarks('c'), mS)
