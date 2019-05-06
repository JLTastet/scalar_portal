# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..data.particles import *
from ..production import hadronic_common as hc
from ..production import two_body_hadronic as hh
from ..production import two_body_quartic as q2
from ..production import three_body_quartic as q3

def test_xi():
    assert_raises(ValueError, lambda: hc.xi('Z', 's', 'd'))
    assert_raises(ValueError, lambda: hc.xi('D', 's', 'z'))
    assert_raises(ValueError, lambda: hc.xi('D', 'a', 'd'))
    # Compare numerical values to Mathematica implementation (with on-shell
    # masses for b and t , and u, d, s, c are assumed to be massless).
    xi_dds, xi_uuc, xi_ddb, xi_dsb = (
        3.0278603568988333e-6, 9.103687490947078e-10, 0.00007830938334212971,
        0.0003809123090962853)
    epsilon = 1e-15
    assert(abs(hc.xi('D', 's', 'd') - xi_dds) / xi_dds < epsilon)
    assert(abs(hc.xi('D', 'b', 'd') - xi_ddb) / xi_ddb < epsilon)
    assert(abs(hc.xi('D', 'b', 's') - xi_dsb) / xi_dsb < epsilon)
    assert(abs(hc.xi('U', 'c', 'u') - xi_uuc) / xi_uuc < epsilon)

def test_two_body_hadronic_width():
    mS = np.array([0, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0])
    w = hh.normalized_decay_width('K', 'pi'        , mS)
    assert(all(np.isfinite(w)))
    assert(all(w[mS >= get_mass('K') - get_mass('pi')] == 0))
    w = hh.normalized_decay_width('B', 'pi'        , mS)
    assert(all(np.isfinite(w)))
    assert(all(w[mS >= get_mass('B') - get_mass('pi')] == 0))
    w = hh.normalized_decay_width('B', 'K'         , mS)
    w = hh.normalized_decay_width('B', 'K*'        , mS)
    w = hh.normalized_decay_width('B', 'K*(1410)'  , mS)
    w = hh.normalized_decay_width('B', 'K_1(1270)' , mS)
    w = hh.normalized_decay_width('B', 'K*_0(700)' , mS)
    w = hh.normalized_decay_width('B', 'K*_2(1430)', mS)
    assert(all(np.isfinite(w)))
    assert(all(w[mS >= get_mass('B') - get_mass('K*_2(1430)')] == 0))
    assert_raises(ValueError, lambda: hh.normalized_decay_width('K', 'B', mS))
    assert_raises(ValueError, lambda: hh.normalized_decay_width('K', 'e', mS))
    # Compare numerical values to the Mathematica implementation.
    # (Scale-invariant top and bottom masses, all other quark masses set to zero.)
    mS = np.array([0, 0.1, 0.3, 1, 2, 4])
    epsilon = 1e-14
    # B -> S K
    target = np.array([
        1.8602870173762396e-13, 1.8605806557632914e-13, 1.8629312904592422e-13,
        1.889850436964034e-13, 1.9802185907651862e-13, 2.1948725218597987e-13])
    w = hh.normalized_decay_width('B', 'K'         , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S pi
    target = np.array([
        4.9254300664076316e-15, 4.92625148889779e-15, 4.93282936535902e-15,
        5.0084557092617675e-15, 5.267439948779629e-15, 6.225493338710085e-15])
    w = hh.normalized_decay_width('B', 'pi'        , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # K -> S pi
    target = np.array([
        6.678263262442073e-20, 6.333404019226674e-20, 3.2299591402536834e-20,
        0, 0, 0])
    w = hh.normalized_decay_width('K', 'pi'        , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*_0(700)
    mS = np.array([0., 1., 2., 4.])
    target = np.array([
        3.552507496745926e-13, 3.5531025129903633e-13, 3.424451657396583e-13,
        7.21993072953889e-14])
    w = hh.normalized_decay_width('B', 'K*_0(700)' , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*_0(1430)
    target = np.array([
        3.8607701226967446e-14, 4.711607522540698e-14, 8.989628771823396e-14, 0])
    w = hh.normalized_decay_width('B', 'K*_0(1430)', mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*
    epsilon = 1e-14
    target = np.array([
        2.249612487030174e-13, 2.249565572911129e-13, 2.22912559568721e-13,
        1.0360227019980757e-13])
    w = hh.normalized_decay_width('B', 'K*'        , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*(1410)
    target = np.array([
        9.87876145856704e-14, 9.185209948881761e-14, 7.070325146331022e-14, 0])
    w = hh.normalized_decay_width('B', 'K*(1410)'  , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*(1680)
    target = np.array([
        6.27051097254405e-14, 5.722507299151126e-14, 4.066968571938808e-14, 0])
    w = hh.normalized_decay_width('B', 'K*(1680)'  , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K_1(1270)
    target = np.array([
        3.960000320317645e-13, 3.898188329844353e-13, 3.601523297769955e-13,
        3.2219512061024887e-16])
    w = hh.normalized_decay_width('B', 'K_1(1270)' , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K_1(1400)
    target = np.array([
        6.907888865453182e-15, 5.405930157795572e-15, 1.6549846870201926e-15, 0])
    w = hh.normalized_decay_width('B', 'K_1(1400)' , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*_2(1430)
    target = np.array([
        1.451262487781669e-13, 1.3542292574122081e-13, 1.0164406039681797e-13, 0])
    w = hh.normalized_decay_width('B', 'K*_2(1430)', mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # Exceptions
    assert_raises(ValueError, lambda: hh.normalized_decay_width('B', 'D', mS))
    assert_raises(ValueError, lambda: hc._get_quark_transition('B', 'D'))
    assert_raises(ValueError, lambda: hc.get_matrix_element('B', 'D'))

def test_two_body_quartic_width():
    mS = np.array([0, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0])
    w = q2.normalized_decay_width('B', mS)
    assert(np.all(np.isfinite(w)))
    assert(np.all(w[2 * mS >= get_mass('B')] == 0))
    assert(np.all(w[2 * mS <  get_mass('B')] >  0))
    w = q2.normalized_decay_width('K', mS)
    assert(np.all(np.isfinite(w)))
    assert(np.all(w[2 * mS >= get_mass('K')] == 0))
    assert(np.all(w[2 * mS <  get_mass('K')] >  0))
    w = q2.normalized_decay_width('B_s', mS)
    assert(np.all(np.isfinite(w)))
    assert(np.all(w[2 * mS >= get_mass('B_s')] == 0))
    assert(np.all(w[2 * mS <  get_mass('B_s')] >  0))
    # Compare numerical values to the Mathematica implementation.
    # (Scale-invariant top and bottom masses, all other quark masses set to zero.)
    epsilon = 1e-14
    mS = np.array([0, 0.1, 0.245, 0.3, 1])
    target = np.array([4.792772703622526e-30, 4.385868760087906e-30,
                       7.432231682506842e-31, 0, 0])
    assert(np.all(np.abs(q2.normalized_decay_width('K', mS) - target) <= epsilon * target))
    mS = np.array([0, 1, 2.6, 3, 10])
    target = np.array([5.450294263421434e-24, 5.0439974303906796e-24,
                       9.393809623524414e-25, 0, 0])
    assert(np.all(np.abs(q2.normalized_decay_width('B', mS) - target) <= epsilon * target))
    mS = np.array([0, 1, 2.65, 3, 10])
    target = np.array([1.985777687153304e-22, 1.8427478974450345e-22,
                       3.127933995192633e-23, 0, 0])
    assert(np.all(np.abs(q2.normalized_decay_width('B_s', mS) - target) <= epsilon * target))

def test_three_body_quartic_width():
    eps = 1e-13
    w = q3.normalized_decay_width('B', 'K', [0, 1, 2.35, 2.5], eps=eps)
    target = np.array([
        5.864813137291421e-23, 3.7761205048911344e-23, 8.811002889266824e-26, 0])
    assert(np.all(np.isfinite(w)))
    assert(np.all(np.abs(w - target) <= eps * np.max(target)))
    w = q3.normalized_decay_width('B', 'pi', [0, 1, 2.5, 2.6], eps=eps)
    target = np.array([
        1.8708923449104718e-24, 1.285452250616117e-24, 6.65590506978302e-27, 0])
    assert(np.all(np.isfinite(w)))
    assert(np.all(np.abs(w - target) <= eps * np.max(target)))
    w = q3.normalized_decay_width('B', 'K*', [0, 1, 2.1, 2.2], eps=eps)
    target = np.array([
        4.282516077558582e-23, 2.2125750925998816e-23, 6.961437471254587e-26, 0])
    assert(np.all(np.isfinite(w)))
    assert(np.all(np.abs(w - target) <= eps * np.max(target)))
    w = q3.normalized_decay_width('B', 'K*(1410)', [0, 1, 1.9, 2.0], eps=eps)
    target = np.array([
        8.92640419964295e-24, 2.755185932435649e-24, 1.9488533048430092e-28, 0])
    assert(np.all(np.isfinite(w)))
    assert(np.all(np.abs(w - target) <= eps * np.max(target)))
    w = q3.normalized_decay_width('B', 'K*(1680)', [0, 1, 1.7, 1.8], eps=eps)
    target = np.array([
        4.687384822781292e-24, 1.1381159882913425e-24, 2.3497197326151174e-27, 0])
    assert(np.all(np.isfinite(w)))
    assert(np.all(np.abs(w - target) <= eps * np.max(target)))
    w = q3.normalized_decay_width('B', 'K_1(1270)', [0, 1, 2, 2.1], eps=eps)
    target = np.array([
        5.129426387752128e-23, 2.03947298773292e-23, 2.7069264949652506e-30, 0])
    assert(np.all(np.isfinite(w)))
    assert(np.all(np.abs(w - target) <= eps * np.max(target)))
    w = q3.normalized_decay_width('B', 'K_1(1400)', [0, 1, 1.9, 2], eps=eps)
    target = np.array([
        3.140488910017822e-25, 7.279745127485522e-26, 1.0549841832137387e-28, 0])
    assert(np.all(np.isfinite(w)))
    assert(np.all(np.abs(w - target) <= eps * np.max(target)))
    w = q3.normalized_decay_width('B', 'K*_2(1430)', [0, 1, 1.9, 2], eps=eps)
    target = np.array([
        1.1878140839320772e-23, 3.15219398483355e-24, 1.0108597041349455e-29, 0])
    assert(np.all(np.isfinite(w)))
    assert(np.all(np.abs(w - target) <= eps * np.max(target)))
    w = q3.normalized_decay_width('B', 'K*_0(700)', [0, 1, 2.2, 2.3], eps=eps)
    target = np.array([
        5.492986232383979e-23, 2.4909603623676532e-23, 3.9395035121516126e-27, 0])
    assert(np.all(np.isfinite(w)))
    assert(np.all(np.abs(w - target) <= eps * np.max(target)))
    w = q3.normalized_decay_width('B', 'K*_0(1430)', [0, 1, 1.9, 2], eps=eps)
    target = np.array([
        1.6612014059946842e-23, 8.953536899181432e-24, 1.685067040498652e-27, 0])
    assert(np.all(np.isfinite(w)))
    assert(np.all(np.abs(w - target) <= eps * np.max(target)))
