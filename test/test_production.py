# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..data.particles import *
from ..production import two_body_hadronic as hh

def test_xi():
    epsilon = 0.15
    assert(abs(hh.xi('D', 2, 1) - hh._xi_ref[('D', 2, 1)]) / hh._xi_ref[('D', 2, 1)] < epsilon)
    assert(abs(hh.xi('D', 3, 1) - hh._xi_ref[('D', 3, 1)]) / hh._xi_ref[('D', 3, 1)] < epsilon)
    assert(abs(hh.xi('D', 3, 2) - hh._xi_ref[('D', 3, 2)]) / hh._xi_ref[('D', 3, 2)] < epsilon)
    assert(abs(hh.xi('U', 2, 1) - hh._xi_ref[('U', 2, 1)]) / hh._xi_ref[('U', 2, 1)] < epsilon)
    assert_raises(ValueError, lambda: hh.xi('D', 1, 2))
    assert_raises(ValueError, lambda: hh.xi('Z', 2, 1))
    assert_raises(ValueError, lambda: hh.xi('D', 2, 0))
    assert_raises(ValueError, lambda: hh.xi('D', 4, 1))
    # Compare numerical values to Mathematica implementation (with on-shell
    # masses for b and t , and u, d, s, c are assumed to be massless).
    xi_dds, xi_uuc, xi_ddb, xi_dsb = (
        2.9931573728902494e-6, 1.2004587795118054e-9, 0.00007741186200444578,
        0.0003765465880216251)
    epsilon = 1e-15
    assert(abs(hh.xi('D', 2, 1) - xi_dds) / xi_dds < epsilon)
    assert(abs(hh.xi('D', 3, 1) - xi_ddb) / xi_ddb < epsilon)
    assert(abs(hh.xi('D', 3, 2) - xi_dsb) / xi_dsb < epsilon)
    assert(abs(hh.xi('U', 2, 1) - xi_uuc) / xi_uuc < epsilon)

def test_two_body_hadronic_amplitude():
    mS = np.array([0, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0])
    A = hh._normalized_amplitude('K', 'pi'        , mS)
    assert(all(np.isfinite(A)))
    assert(all(A[mS >= get_mass('K') - get_mass('pi')] == 0))
    A = hh._normalized_amplitude('B', 'pi'        , mS)
    assert(all(np.isfinite(A)))
    assert(all(A[mS >= get_mass('B') - get_mass('pi')] == 0))
    A = hh._normalized_amplitude('B', 'K'         , mS)
    A = hh._normalized_amplitude('B', 'K*'        , mS)
    A = hh._normalized_amplitude('B', 'K*(1410)'  , mS)
    A = hh._normalized_amplitude('B', 'K_1(1270)' , mS)
    A = hh._normalized_amplitude('B', 'K*_0(700)' , mS)
    A = hh._normalized_amplitude('B', 'K*_2(1430)', mS)
    assert(all(np.isfinite(A)))
    assert(all(A[mS >= get_mass('B') - get_mass('K*_2(1430)')] == 0))
    assert_raises(ValueError, lambda: hh._normalized_amplitude('K', 'B', mS))
    assert_raises(ValueError, lambda: hh._normalized_amplitude('K', 'e', mS))
    # Check that trying to compute χ_Y' for a spin other than 0, 1 or 2 fails.
    assert_raises(ValueError, lambda: hh._chi('B', 'e', 0.))

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
    # Compare numerical values to Mathematica implementation (on-shell t and b masses).
    mS = np.array([0, 0.1, 0.3, 1, 2, 4])
    epsilon = 1e-15
    # B -> S K
    target = np.array([
        1.8178890537824978e-13, 1.8181759998314708e-13,
        1.8204730610071367e-13, 1.8467786908971138e-13, 1.9350872562266444e-13,
        2.1448489908640465e-13])
    w = hh.normalized_decay_width('B', 'K'         , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S pi
    target = np.array([
        4.8131741603627665e-15, 4.81397686174134e-15, 4.82040482124668e-15,
        4.8943075585523985e-15, 5.147389265689037e-15, 6.083607577286032e-15])
    w = hh.normalized_decay_width('B', 'pi'        , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # K -> S pi
    target = np.array([
        6.526058382213544e-20, 6.189058856075263e-20, 3.1563448598987664e-20,
        0, 0, 0])
    w = hh.normalized_decay_width('K', 'pi'        , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*_0(700)
    mS = np.array([0., 1., 2., 4.])
    target = np.array([
        3.3687694103831563e-13, 3.36933365198568e-13, 3.2473367055074707e-13,
        6.846511037354604e-14])
    w = hh.normalized_decay_width('B', 'K*_0(700)' , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*_0(1430)
    target = np.array([
        3.9483657603365625e-14, 4.818507506774215e-14, 9.193591255832439e-14, 0])
    w = hh.normalized_decay_width('B', 'K*_0(1430)', mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*
    epsilon = 1e-14
    target = np.array([
        2.1983413727159672e-13, 2.198295527820699e-13, 2.1783213998996578e-13,
        1.0124106182758784e-13])
    w = hh.normalized_decay_width('B', 'K*'        , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*(1410)
    target = np.array([
        9.653613744929648e-14, 8.975869028165762e-14, 6.909184749526634e-14, 0])
    w = hh.normalized_decay_width('B', 'K*(1410)'  , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*(1680)
    target = np.array([
        6.127599210302626e-14, 5.5920851364052253e-14, 3.9742779366554684e-14, 0])
    w = hh.normalized_decay_width('B', 'K*(1680)'  , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K_1(1270)
    target = np.array([
        3.869747607782547e-13, 3.8093443797729307e-13, 3.519440666307941e-13,
        3.148519435272946e-16])
    w = hh.normalized_decay_width('B', 'K_1(1270)' , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K_1(1400)
    target = np.array([
        6.750450568087554e-15, 5.2827231322776586e-15, 1.6172657867359548e-15, 0])
    w = hh.normalized_decay_width('B', 'K_1(1400)' , mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
    # B -> S K*_2(1430)
    target = np.array([
        1.418186637900873e-13, 1.3232936663964834e-13, 9.922995648284822e-14, 0])
    w = hh.normalized_decay_width('B', 'K*_2(1430)', mS)
    assert(np.all(np.abs(w - target) <= epsilon * target))
