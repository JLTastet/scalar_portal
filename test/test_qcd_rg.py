# -*- coding: utf-8 -*-

from __future__ import absolute_import, division

from nose.tools import assert_equals, assert_raises

import numpy as np

from ..data.qcd_rg import *

def test_alpha_s():
    assert(np.isnan(alpha_s(  0.5)))
    assert(abs(alpha_s( 1.0) - 0.430 ) < 1e-3)
    assert(abs(alpha_s( 2.0) - 0.298 ) < 1e-3)
    assert(abs(alpha_s( 5.0) - 0.212 ) < 1e-3)
    assert(abs(alpha_s(10.0) - 0.177 ) < 1e-3)
    assert(abs(alpha_s( 100) - 0.116 ) < 1e-3)
    assert(abs(alpha_s( 1e3) - 0.0868) < 1e-4)
    assert(abs(alpha_s( 1e4) - 0.0693) < 1e-4)

def test_running_mass():
    m_q_msbar_2GeV = np.array([
        m_u_msbar_2GeV, m_d_msbar_2GeV, m_s_msbar_2GeV,
        m_c_msbar_2GeV, m_b_msbar_2GeV, m_t_msbar_2GeV
    ])
    assert(np.all(np.abs(running_mass(m_q_msbar_2GeV, 2.0, 2.0) - m_q_msbar_2GeV) < 1e-14))

def test_get_quark_mass():
    assert(abs(get_quark_mass('u') - m_u_msbar_2GeV) < 1e-14)
    assert(abs(get_quark_mass('d') - m_d_msbar_2GeV) < 1e-14)
    assert(abs(get_quark_mass('s') - m_s_msbar_2GeV) < 1e-14)
    assert(abs(get_quark_mass('c') - m_c_msbar_2GeV) < 1e-14)
    assert(abs(get_quark_mass('b') - m_b_msbar_2GeV) < 1e-14)
    assert(abs(get_quark_mass('t') - m_t_msbar_2GeV) < 1e-14)
    mu = np.array([1.0, 2.0, 3.0, 5.0])
    for q in all_quarks:
        m_q_msbar_2GeV = get_quark_mass(q)
        m_q_msbar = running_mass(m_q_msbar_2GeV, 2.0, mu)
        assert(np.all(np.abs(get_quark_mass(q, mu) - m_q_msbar) < 1e-14))
    assert_raises(ValueError, lambda: get_quark_mass('~t'))

