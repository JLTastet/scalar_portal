# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises

from .. import constants as c

def test_m_Q():
    assert_equals(c.m_Q('U', 1), c.m_u)
    assert_equals(c.m_Q('U', 2), c.m_c)
    assert_equals(c.m_Q('U', 3), c.m_t)
    assert_equals(c.m_Q('D', 1), c.m_d)
    assert_equals(c.m_Q('D', 2), c.m_s)
    assert_equals(c.m_Q('D', 3), c.m_b)
    assert_raises(ValueError, lambda: c.m_Q('U', 0))
    assert_raises(ValueError, lambda: c.m_Q('U', 4))
    assert_raises(ValueError, lambda: c.m_Q('D', 0))
    assert_raises(ValueError, lambda: c.m_Q('D', 4))
    assert_raises(ValueError, lambda: c.m_Q('Z', 1))

def test_ckm():
    assert_equals(c.ckm(1,1), c.Vud)
    assert_equals(c.ckm(1,2), c.Vus)
    assert_equals(c.ckm(1,3), c.Vub)
    assert_equals(c.ckm(2,1), c.Vcd)
    assert_equals(c.ckm(2,2), c.Vcs)
    assert_equals(c.ckm(2,3), c.Vcb)
    assert_equals(c.ckm(3,1), c.Vtd)
    assert_equals(c.ckm(3,2), c.Vts)
    assert_equals(c.ckm(3,3), c.Vtb)
    assert_raises(ValueError, lambda: c.ckm(0,1))
    assert_raises(ValueError, lambda: c.ckm(4,1))
    assert_raises(ValueError, lambda: c.ckm(1,0))
    assert_raises(ValueError, lambda: c.ckm(1,4))

def test_xi():
    epsilon=0.11
    assert(abs(c.xi('D', 1, 2) - c.xi_d_ds) / c.xi_d_ds < epsilon)
    assert(abs(c.xi('D', 1, 3) - c.xi_d_db) / c.xi_d_db < epsilon)
    assert(abs(c.xi('D', 2, 3) - c.xi_d_sb) / c.xi_d_sb < epsilon)
    assert(abs(c.xi('U', 1, 2) - c.xi_u_uc) / c.xi_u_uc < epsilon)
    assert_raises(ValueError, lambda: c.xi('D', 2, 1))
    assert_raises(ValueError, lambda: c.xi('Z', 1, 2))
    assert_raises(ValueError, lambda: c.xi('D', 0, 2))
    assert_raises(ValueError, lambda: c.xi('D', 1, 4))
