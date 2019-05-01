# -*- coding: utf-8 -*-

from __future__ import absolute_import, division

from nose.tools import assert_equals, assert_raises

from ..data.form_factors import *

def test_form_factors_at_zero():
    # Scalar form factors
    f = get_form_factor('B', 'K')
    assert(abs(f(0) - 0.33) < 1e-2)
    f = get_form_factor('B', 'pi')
    assert(abs(f(0) - 0.258) < 1e-3)
    f = get_form_factor('K', 'pi')
    assert(abs(f(0) - 0.96) < 1e-2)
    # Pseudoscalar form factors
    f = get_form_factor('B', 'K*_0(700)')
    assert(abs(f(0) - 0.46) < 1e-2)
    f = get_form_factor('B', 'K*_0(1430)')
    assert(abs(f(0) - 0.17) < 1e-2)
    # Vector form factors
    f = get_form_factor('B', 'K*')
    assert(abs(f(0) - 0.374) < 0.033)
    f = get_form_factor('B', 'K*(1410)')
    assert(abs(f(0) - 0.300) < 0.036)
    f = get_form_factor('B', 'K*(1680)')
    assert(abs(f(0) - 0.22) < 0.04)
    # Pseudo-vector form factors
    f = get_form_factor('B', 'K_1(1270)')
    assert(abs(f(0) - (-0.52)) < 0.13)
    f = get_form_factor('B', 'K_1(1400)')
    assert(abs(f(0) - (-0.07)) < 0.033)
    # Tensor form factors
    f = get_form_factor('B', 'K*_2(1430)')
    assert(abs(f(0) - 0.23) < 1e-2)
    # Exceptions
    assert_raises(ValueError, lambda: get_form_factor('K', 'B'))
    assert_raises(ValueError, lambda: get_form_factor('B', 'D'))
