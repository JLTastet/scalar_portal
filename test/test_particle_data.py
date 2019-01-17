# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises

from ..data.particle import *

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
    # Try with a few non-existent particles...
    assert_raises(ValueError, lambda: get_pdg_id('pibar0')    )
    assert_raises(ValueError, lambda: get_pdg_id('K_L+')      )
    assert_raises(ValueError, lambda: get_pdg_id('K_Sbar0')   )
    assert_raises(ValueError, lambda: get_pdg_id('~Gravitino'))
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

def test_get_spin_code():
    assert_equals(get_spin_code('pi+'       ), 1)
    assert_equals(get_spin_code('K*_0(1430)'), 1)
    assert_equals(get_spin_code('K*(1410)-' ), 3)
    assert_equals(get_spin_code('K*_2(1430)'), 5)
    assert_equals(get_spin_code('Bbar0'     ), 1)

def test_get_parity():
    assert_equals(get_parity('K'            ), -1)
    assert_equals(get_parity('K*_0(700)'    ), +1)
    assert_equals(get_parity('K*(1680)'     ), -1)
    assert_equals(get_parity('K_1(1270)bar0'), +1)
    assert_equals(get_parity('K*_2(1430)0'  ), +1)
    assert_equals(get_parity('B'            ), -1)

