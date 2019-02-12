# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..data.particles import *
from ..decay import leptonic as lp
from ..decay import two_pions as tp
from ..decay import two_gluons as gg
from ..decay import two_quarks as qq

def test_leptonic_width():
    eps = 1e-12
    mS = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0])
    w = lp.normalized_decay_width('e'  , mS)
    assert(all(w[mS > 2*get_mass('e'  )] >  0))
    assert(all(w[mS < 2*get_mass('e'  )] == 0))
    target = np.array([
        1.6900126864964634e-15, 1.7165714271571945e-14, 8.584148208244031e-14,
        1.7168377110602744e-13, 3.433679456830234e-13, 6.867360931015804e-13,
        1.7168403739688331e-12])
    assert(np.all(np.abs(w - target) <= eps * target))
    w = lp.normalized_decay_width('mu' , mS)
    assert(all(w[mS > 2*get_mass('mu' )] >  0))
    assert(all(w[mS < 2*get_mass('mu' )] == 0))
    target = np.array([
        0, 0, 2.732025043654036e-9, 6.853907708050072e-9, 1.443491869260822e-8,
        2.9237286606176158e-8, 7.335112420583204e-8])
    assert(np.all(np.abs(w - target) <= eps * target))
    w = lp.normalized_decay_width('tau', mS)
    assert(all(w[mS > 2*get_mass('tau')] >  0))
    assert(all(w[mS < 2*get_mass('tau')] == 0))
    target = np.array([
        0, 0, 0, 0, 0, 8.028575393027576e-7, 0.00001695364846943955])
    assert(np.all(np.abs(w - target) <= eps * target))
    assert_raises(ValueError, lambda: lp.normalized_decay_width('K', mS))

def test_two_pion_width():
    mS = np.array([0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 2.0, 4.0, 10.0])
    wn = tp.normalized_decay_width('neutral', mS)
    assert(all(wn[mS < 2*get_mass('pi')]) == 0)
    assert(all(wn[np.logical_and(mS > 2*get_mass('pi'), mS < 1.0)]) > 0)
    assert(all(np.isnan(wn[mS > 1.0])))
    wc = tp.normalized_decay_width('charged', mS)
    assert(all(wc[mS < 2*get_mass('pi')]) == 0)
    assert(all(wc[np.logical_and(mS > 2*get_mass('pi'), mS < 1.0)]) > 0)
    assert(all(np.isnan(wc[mS > 1.0])))
    # Now test that Γ(S -> pi+ pi-) = 2 Γ(S -> pi0 pi0)
    eps = 1e-12
    finite = np.isfinite(wn)
    assert(np.all(np.abs(wc - 2*wn)[finite] <= eps * wc[finite]))
    assert_raises(ValueError, lambda: tp.normalized_decay_width('test', mS))
    # To test the numerical values, we use the masses at the interpolation knots.
    # This way, the different interpolations between scipy and Mathematica do not
    # introduce an additional error.
    mS_precise = np.array([0.3934299772377165, 0.7008357666750972, 0.9969868714505782])
    w = tp.normalized_decay_width('charged', mS_precise)
    target = np.array([
        9.131368964469796e-9, 4.948045885831599e-8, 9.521085542274886e-7])
    assert(np.all(np.abs(w - target) <= eps * target))
    # Test masses away from the interpolation knots.
    # If using linear interpolation, we actually expect the same result as the
    # Mathematica version.
    eps = 1e-12
    target = np.array([
        0, 0, 0, 2.4299966755579365e-9, 1.92145326982196e-8,
        7.959362839919174e-8, 8.801586892729695e-07])
    assert(np.all(np.abs(wc[mS <= 1.0] - target) <= eps * target))

def test_two_gluon_width():
    mS = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0])
    w = gg.normalized_decay_width(mS)
    assert(all(w[mS >= 2.0] > 0))
    assert(all(np.isnan(w[mS < 2.0])))
    eps = 1e-8
    target = np.array([
        5.346506003380509e-8,3.366555346862895e-7,1.7387372280010782e-6])
    # Here we compute the error relative to the maximum value, in order to
    # account for numerical cancellations in small decay widths, which reduce
    # the relative precision.
    assert(np.all(np.abs(w[mS >= 2] - target) <= eps * np.max(target)))

def test_two_quark_width():
    mS = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0])
    valid = (mS >= 2.0) & (mS < 2*get_mass('B'))
    w = qq.normalized_decay_width('s', mS)
    assert(np.all(w[valid] > 0))
    assert(np.all(np.isnan(w[~valid])))
    eps = 1e-8
    target = np.array([
        6.850535126425071e-8, 7.463155026546353e-8, 8.312529294924812e-8,
        9.208282350175197e-8, 1.359595676119765e-7])
    assert(np.all(np.abs(w[valid] - target) <= eps * target))
    w = qq.normalized_decay_width('c', mS)
    assert(np.all(w[valid & (mS > 2*get_mass('D'))] > 0))
    assert(np.all(np.isnan(w[~valid])))
    target = np.array([0.000012532775792826198, 0.000018519975220329018])
    assert(np.all(np.abs(w[valid & (mS >= 5.0)] - target) <= eps * target))
    assert_raises(ValueError, lambda: qq.normalized_decay_width('b', mS))
