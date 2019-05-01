# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises

from ..data import constants as c

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

def test_VUD():
    assert_equals(c.VUD('u','d'), c.Vud)
    assert_equals(c.VUD('u','s'), c.Vus)
    assert_equals(c.VUD('u','b'), c.Vub)
    assert_equals(c.VUD('c','d'), c.Vcd)
    assert_equals(c.VUD('c','s'), c.Vcs)
    assert_equals(c.VUD('c','b'), c.Vcb)
    assert_equals(c.VUD('t','d'), c.Vtd)
    assert_equals(c.VUD('t','s'), c.Vts)
    assert_equals(c.VUD('t','b'), c.Vtb)
    assert_raises(ValueError, lambda: c.VUD('d', 'u'))
    assert_raises(ValueError, lambda: c.VUD('a', 'd'))
    assert_raises(ValueError, lambda: c.VUD('u', 'z'))
