# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..data.particles import *
from ..decay import leptonic as lp
from ..decay import two_pions as tp
from ..decay import two_kaons as kk
from ..decay import multimeson as mm
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
    mS = np.array([0.01, 0.2, 0.3, 0.5, 0.9, 1.0, 1.5, 2.0, 2.5, 4.0, 10.0])
    wn = tp.normalized_decay_width('neutral', mS)
    assert(all(wn[mS < 2*get_mass('pi')] == 0))
    assert(all(wn[(mS > 2*get_mass('pi')) & (mS <= 2.0)] > 0))
    assert(all(np.isnan(wn[mS > 2.0])))
    wc = tp.normalized_decay_width('charged', mS)
    assert(all(wc[mS < 2*get_mass('pi')] == 0))
    assert(all(wc[(mS > 2*get_mass('pi')) & (mS <= 2.0)] > 0))
    assert(all(np.isnan(wc[mS > 2.0])))
    # Now test that Γ(S -> pi+ pi-) = 2 Γ(S -> pi0 pi0)
    eps = 1e-12
    finite = np.isfinite(wn)
    assert(np.all(np.abs(wc - 2*wn)[finite] <= eps * wc[finite]))
    assert_raises(ValueError, lambda: tp.normalized_decay_width('test', mS))
    # To test the numerical values, we use the masses at the interpolation knots.
    # This way, the different interpolations between scipy and Mathematica do not
    # introduce an additional error.
    mS = np.array([
        0.2808915847265062, 0.5011797711548195, 1.0036678784980348,
        1.4995286492412911, 1.9618148657454992])
    w = tp.normalized_decay_width('charged', mS)
    target = np.array([
        4.880143779482717e-10, 1.4857373356991338e-8, 3.604336514028455e-7,
        2.319233832129673e-9, 2.310540370746613e-9])
    assert(np.all(np.abs(w - target) <= eps * target))
    # Test masses away from the interpolation knots.
    # If using linear interpolation, we actually expect the same result as the
    # Mathematica version.
    eps = 1e-12
    mS = np.array([0.2, 0.3, 0.5, 1.0, 1.4, 2.0])
    w = tp.normalized_decay_width('charged', mS)
    target = np.array([
        0, 1.8927616589680657e-9, 1.475091932044681e-8, 4.019482951592148e-7,
        6.6426603227451354e-9, 2.39245459053496e-9])
    assert(np.all(np.abs(w - target) <= eps * target))

def test_two_kaon_width():
    mS = np.array([0.01, 0.2, 0.3, 0.5, 0.9, 1.0, 1.5, 2.0, 2.5, 4.0, 10.0])
    w = kk.normalized_decay_width(mS)
    assert(np.all(w[ mS < 2*get_mass('K')               ] == 0))
    assert(np.all(w[(mS > 2*get_mass('K')) & (mS <= 2.0)] >  0))
    assert(np.all(np.isnan(w[mS > 2.0])))
    # Test numerical values at the interpolation knots.
    eps = 1e-12
    mS = np.array([1.0029956973063714, 1.5221480119166657, 1.992527315035567])
    w = kk.normalized_decay_width(mS)
    target = np.array([
        5.3034678644569654e-8, 4.121879878937285e-8, 1.7589704763603936e-8])
    assert(np.all(np.abs(w - target) <= eps * target))
    # Test masses away from the interpolation knots.
    eps = 1e-12 # For linear interpolation, we can use a small ε
    mS = np.array([0.9, 1.0, 1.4, 2.0])
    w = kk.normalized_decay_width(mS)
    target = np.array([
        0, 3.25398123949564e-8, 4.558339609849428e-8, 1.743794584685908e-8])
    assert(np.all(np.abs(w - target) <= eps * target))

def test_multimeson():
    threshold = 4 * get_mass('pi')
    assert_equals(mm.normalized_decay_width(threshold), 0)
    Lambda_S = 2.0
    eps = 1e-14
    total_width_below = (
        tp.normalized_decay_width('neutral', Lambda_S) +
        tp.normalized_decay_width('charged', Lambda_S) +
        kk.normalized_decay_width(Lambda_S) * 2 +
        mm.normalized_decay_width(Lambda_S)
    )
    total_width_above = (
        gg.normalized_decay_width(Lambda_S) +
        qq.normalized_decay_width('s', Lambda_S) +
        qq.normalized_decay_width('c', Lambda_S)
    )
    assert(abs(total_width_below - total_width_above) <= eps * total_width_above)
    mS = np.array([0.1, 0.5, 0.6, 1, 1.4, 1.7, 2])
    w = mm.normalized_decay_width(mS)
    assert(np.all(mm.normalized_total_width(mS) == w))
    eps = 1e-8
    target = np.array([
        0, 0, 9.444526857355288e-10, 8.981725642908731e-9,
        2.711286641711554e-8, 4.993756076043946e-8, 8.261294432080519e-8])
    assert(np.all(np.abs(w - target) <= eps * target))

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
    # Test S -> s sbar (always far from the threshold).
    w = qq.normalized_decay_width('s', mS)
    assert(np.all(w[valid] > 0))
    assert(np.all(np.isnan(w[~valid])))
    eps = 1e-8
    target = np.array([
        6.76124578665207e-8, 7.427821148146254e-8, 8.292873563795307e-8,
        9.195446037707259e-8, 1.3592134596330357e-7])
    assert(np.all(np.abs(w[valid] - target) <= eps * target))
    # Test the range of validity of the formula for S -> c cbar.
    w = qq.normalized_decay_width('c', mS)
    assert(np.all(w[valid & (mS > 2*get_mass('D'))] > 0))
    assert(np.all(np.isnan(w[~valid])))
    # Test S -> c cbar near the threshold.
    mS = np.array([3.75, 4, 4.5, 5, 10])
    w = qq.normalized_decay_width('c', mS)
    eps = 1e-8
    target = np.array([
        5.745753285992689e-6, 7.521215316538118e-6, 9.090351049835501e-6,
        0.000010189686857190758, 0.00001780299310120903])
    assert(np.all(np.abs(w - target) <= eps * target))
    # Test QCD corrections
    beta = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    eps = 1e-13
    target = np.array([
        492.5263641647134, 48.6432489800492, 15.641285460481056,
        8.300452608332847, 4.022675132174538, -0.44454292663622885,
        -5.426657197539063])
    assert(np.all(np.abs(qq._Delta_H(beta) - target) <= eps * np.abs(target)))
    assert_raises(ValueError, lambda: qq.normalized_decay_width('b', mS))

def test_dilogarithm():
    y = np.array([-1000, -100, -10, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    eps = 1e-14
    target = np.array([
        -25.502475813889966,-12.23875517731494,-4.198277886858104,
        -0.8224670334241132, -0.6797815878346812, -0.5281071740446666,
        -0.3658325775124496, -0.1908001377775357, 0., 0.21100377543970492,
        0.4492829744712819, 0.7275863077163336, 1.0747946000082484,
        1.6449340668482264])
    # Since the target can be negative, we need to take its absolute value.
    assert(np.all(np.abs(qq.Li2(y) - target) <= eps * np.abs(target)))
