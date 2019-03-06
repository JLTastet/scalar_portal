# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises

from ..data.particles import *
from ..data.constants import *

def test_get_pdg_id():
    # Test meson PDG IDs
    assert_equals(get_pdg_id('K0'   ), +311)
    assert_equals(get_pdg_id('Kbar0'), -311)
    assert_equals(get_pdg_id('K+'   ), +321)
    assert_equals(get_pdg_id('K-'   ), -321)
    assert_equals(get_pdg_id('K_L0' ), +130)
    assert_equals(get_pdg_id('K_S0' ), +310)
    assert_equals(get_pdg_id('pi0'  ), +111)
    assert_equals(get_pdg_id('K*_0(700)+'    ), +9000321)
    assert_equals(get_pdg_id('K*(1410)-'     ), - 100323)
    assert_equals(get_pdg_id('K_1(1270)bar0' ), -  10313)
    assert_equals(get_pdg_id('K*_2(1430)0'   ), +    315)
    assert_equals(is_meson('K_L0'), True)
    assert_equals(is_meson('K_S0'), True)
    # Test lepton PDG IDs
    assert_equals(get_pdg_id('e+'     ), -11)
    assert_equals(get_pdg_id('mu-'    ), +13)
    assert_equals(get_pdg_id('nu_ebar'), -12)
    # Test quarks & gluon PDG IDs
    assert_equals(get_pdg_id('cbar'), -4)
    assert_equals(get_pdg_id('t'   ), +6)
    assert_equals(get_pdg_id('g'   ), 21)
    # Test γ PDG ID
    assert_equals(get_pdg_id('gamma'), 22)
    # Try with a few non-existent particles...
    assert_raises(ValueError, lambda: get_pdg_id('pibar0')    )
    assert_raises(ValueError, lambda: get_pdg_id('K_L+')      )
    assert_raises(ValueError, lambda: get_pdg_id('K_Sbar0')   )
    assert_raises(ValueError, lambda: get_pdg_id('gamma0')    )
    # Empty particle string
    assert_raises(ValueError, lambda: get_pdg_id(''))

def test_get_name():
    # Test meson names
    assert_equals(get_name(+311), 'K0'   )
    assert_equals(get_name(-311), 'Kbar0')
    assert_equals(get_name(+321), 'K+'   )
    assert_equals(get_name(-321), 'K-'   )
    assert_equals(get_name(+130), 'K_L0' )
    assert_equals(get_name(+310), 'K_S0' )
    assert_equals(get_name(+111), 'pi0'  )
    assert_equals(get_name(+9000321), 'K*_0(700)+'    )
    assert_equals(get_name(- 100323), 'K*(1410)-'     )
    assert_equals(get_name(-  10313), 'K_1(1270)bar0' )
    assert_equals(get_name(+    315), 'K*_2(1430)0'   )
    # Non-existent ID
    assert_raises(ValueError, lambda: get_name(666))

def test_get_mass():
    assert(abs(get_mass('pi' ) - 0.137) < 0.003)
    assert(abs(get_mass('pi0') - 0.137) < 0.003)
    assert(abs(get_mass('pi-') - 0.137) < 0.003)
    assert(abs(get_mass('B'    ) - 5.279) < 0.001)
    assert(abs(get_mass('B+'   ) - 5.279) < 0.001)
    assert(abs(get_mass('Bbar0') - 5.279) < 0.001)
    assert(abs(get_mass('K*') - 0.892) < 0.005)
    assert(abs(get_mass('K*_2(1430)') - 1.426) < 0.002)
    assert(abs(get_mass('K_1(1400)0') - 1.403) < 0.010)
    assert_equals(get_mass('e-' ), m_e  )
    assert_equals(get_mass('mu+'), m_mu )
    assert_equals(get_mass('tau'), m_tau)
    assert_equals(get_mass('g'    ), 0      )
    assert_equals(get_mass('gamma'), 0      )
    assert_equals(get_mass('Z0'   ), 91.1876)
    assert_raises(ValueError, lambda: get_mass('W0' ))
    assert_raises(ValueError, lambda: get_mass('mu0'))

def test_get_lifetime():
    assert(abs(get_lifetime('K+'  )/second - 1.238e-8 ) < 0.002e-8 )
    assert(abs(get_lifetime('K_S0')/second - 8.954e-11) < 0.004e-11)
    assert(abs(get_lifetime('K_L0')/second - 5.116e-8 ) < 0.021e-8 )
    assert(abs(get_lifetime('B+'  )/second - 1.638e-12) < 0.004e-12)
    assert(abs(get_lifetime('B0'  )/second - 1.520e-12) < 0.004e-12)
    assert_raises(ValueError, lambda: get_lifetime('K-') )
    assert_raises(ValueError, lambda: get_lifetime('K*+'))
    assert_raises(ValueError, lambda: get_lifetime('mu+'))

def test_get_spin_code():
    assert_equals(get_spin_code('pi+'       ), 1)
    assert_equals(get_spin_code('K*_0(1430)'), 1)
    assert_equals(get_spin_code('K*(1410)-' ), 3)
    assert_equals(get_spin_code('K*_2(1430)'), 5)
    assert_equals(get_spin_code('Bbar0'     ), 1)
    assert_equals(get_spin_code('e-'        ), 2)
    assert_equals(get_spin_code('tau'       ), 2)
    assert_raises(ValueError, lambda: get_spin_code('~Gravitino'))

def test_get_parity():
    assert_equals(get_parity('K'            ), -1)
    assert_equals(get_parity('K*_0(700)'    ), +1)
    assert_equals(get_parity('K*(1680)'     ), -1)
    assert_equals(get_parity('K_1(1270)bar0'), +1)
    assert_equals(get_parity('K*_2(1430)0'  ), +1)
    assert_equals(get_parity('B'            ), -1)

def test_quark_quantum_numbers():
    assert_equals(get_abs_strangeness('K*'), 1)
    assert_equals(get_abs_charm('K*')      , 0)
    assert_equals(get_abs_beauty('K*')     , 0)
    assert_equals(get_abs_strangeness('B'), 0)
    assert_equals(get_abs_charm('B')      , 0)
    assert_equals(get_abs_beauty('B')     , 1)

def test_is_meson():
    assert_equals(is_meson('pi'        ), True)
    assert_equals(is_meson('K*_0(1430)'), True)
    assert_equals(is_meson('Bbar0'     ), True)
    assert_equals(is_meson('K*-'       ), True)
    assert_equals(is_meson('K_1(1400)+'), True)
    assert_equals(is_meson('e+')  , False)
    assert_equals(is_meson('mu')  , False)
    assert_equals(is_meson('tau-'), False)

def test_get_qcd_state():
    assert_equals(get_qcd_state('K+'   ), 'K'  )
    assert_equals(get_qcd_state('Kbar0'), 'K'  )
    assert_equals(get_qcd_state('K_L0' ), 'K_L')
    assert_equals(get_qcd_state('K*_0(700)bar0'), 'K*_0(700)')
    assert_raises(ValueError, lambda: get_qcd_state('K'        ))
    assert_raises(ValueError, lambda: get_qcd_state('K*_0(700)'))
    assert_raises(ValueError, lambda: get_qcd_state('e+'       ))
    assert_raises(ValueError, lambda: get_qcd_state('gamma'    ))

def test_is_lepton():
    assert_equals(is_lepton('pi'        ), False)
    assert_equals(is_lepton('K*_0(1430)'), False)
    assert_equals(is_lepton('Bbar0'     ), False)
    assert_equals(is_lepton('K*-'       ), False)
    assert_equals(is_lepton('K_1(1400)+'), False)
    assert_equals(is_lepton('e+')  , True)
    assert_equals(is_lepton('mu')  , True)
    assert_equals(is_lepton('tau-'), True)

def test_alpha_s():
    # The allowed error corresponds to the difference between 4- and 5-loop
    # calculations.
    assert(abs(alpha_s(  3., 4) - 0.2539    ) < 3e-4)
    assert(abs(alpha_s(100., 5) - 0.11647473) < 3e-8)
    assert(abs(alpha_s(500., 6) - 0.095273  ) < 5e-6)
    assert_equals(alpha_s(M_Z, 5), alpha_s_MZ)

def test_on_shell_mass():
    assert_raises(ValueError, lambda: on_shell_mass('u'))
    assert_raises(ValueError, lambda: on_shell_mass('d'))
    assert_raises(ValueError, lambda: on_shell_mass('s'))
    assert(abs(on_shell_mass('c') -   1.5) < 1.5)
    assert(abs(on_shell_mass('b') -   5. ) < 0.5)
    assert(abs(on_shell_mass('t') - 173.0) < 0.4)
    assert_raises(ValueError, lambda: on_shell_mass('~t'))

def test_msbar_mass():
    mu = np.array([1.0, 1.28, 2.0, 3.0, 4.18])
    assert_raises(ValueError, lambda: msbar_mass('u', mu, 3))
    assert_raises(ValueError, lambda: msbar_mass('d', mu, 3))
    assert_raises(ValueError, lambda: msbar_mass('t', mu, 6))
    assert(np.all(msbar_mass('s', mu, 3) > 0))
    assert(np.all(msbar_mass('c', mu, 4) > 0))
    assert(np.all(msbar_mass('b', mu, 5) > 0))
    eps = 1e-14
    assert(abs(msbar_mass('s', 2.0, 3) - m_s_msbar_2GeV) < eps)
    assert(abs(msbar_mass('c', m_c_si, 4) - m_c_si) < eps)
    assert(abs(msbar_mass('b', m_b_si, 5) - m_b_si) < eps)
    assert_raises(ValueError, lambda: msbar_mass('b', mu, 2))
    assert_raises(ValueError, lambda: msbar_mass('t', mu, 7))
    assert_raises(ValueError, lambda: msbar_mass('~t', mu, 6))
    # Not implemented
    assert_raises(ValueError, lambda: msbar_mass('s', mu, 6))
    assert_raises(ValueError, lambda: msbar_mass('c', mu, 6))

def test_scale_invariant_mass():
    assert_raises(ValueError, lambda: scale_invariant_mass('s' ))
    assert_raises(ValueError, lambda: scale_invariant_mass('~t'))
    assert(abs(scale_invariant_mass('c') -   1.28) <  0.05)
    assert(abs(scale_invariant_mass('b') -   4.18) <  0.05)
    assert(abs(scale_invariant_mass('t') - 174.0 ) < 15.  )
