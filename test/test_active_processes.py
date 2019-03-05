# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises

from ..api.active_processes import *
from ..decay.leptonic import Leptonic
from ..decay.two_pions import TwoPions
from ..decay.two_gluons import TwoGluons
from ..decay.two_quarks import TwoQuarks

def _make_channels_and_groups():
    channels = [
        Leptonic('e'),
        Leptonic('mu'),
        TwoPions('neutral'),
        TwoPions('charged'),
        TwoGluons(),
        TwoQuarks('s'),
        TwoQuarks('c'),
    ]
    groups = {
        'S -> l+ l-': [
            'S -> e+ e-'  ,
            'S -> mu+ mu-',
        ],
        'S -> pi pi': [
            'S -> pi0 pi0',
            'S -> pi+ pi-',
        ],
        'S -> q qbar': [
            'S -> s sbar',
            'S -> c cbar',
        ],
        'HardQCD': [
            'S -> s sbar',
            'S -> c cbar',
            'S -> g g'   ,
        ],
    }
    return channels, groups

def test_list():
    channels, groups = _make_channels_and_groups()
    ps = ActiveProcesses(channels, groups)
    assert_equals(set(ps.list_available()), set(str(ch) for ch in channels))
    assert_equals(set(ps.list_available_groups()), set(groups.keys()))

def test_enable_disable():
    channels, groups = _make_channels_and_groups()
    ps = ActiveProcesses(channels, groups, default_selection=['All'])
    assert_equals(set(ps.list_enabled()), set(ps.list_available()))
    ps.disable_all()
    assert_equals(set(ps.list_enabled()), set())
    ps.enable('S -> c cbar')
    assert_equals(set(ps.list_enabled()), set(['S -> c cbar']))
    ps.enable('S -> l+ l-')
    assert_equals(set(ps.list_enabled()), set(['S -> c cbar', 'S -> e+ e-', 'S -> mu+ mu-']))
    ps.disable('S -> c cbar')
    ps.disable('S -> s sbar') # If the process is not enabled, do nothing.
    assert_equals(set(ps.list_enabled()), set(['S -> e+ e-', 'S -> mu+ mu-']))
    ps.disable_all()
    ps.enable('HardQCD')
    ps.disable('S -> q qbar')
    assert_equals(set(ps.list_enabled()), set(['S -> g g']))
    ps.enable_all()
    assert_equals(set(ps.list_enabled()), set(ps.list_available()))
    assert_raises(ValueError, lambda: ps.enable('S -> tau+ tau-'))

def test_recursion():
    channels, groups = _make_channels_and_groups()
    groups['AllQCD'] = ['S -> pi pi', 'HardQCD']
    ps = ActiveProcesses(channels, groups)
    ps.enable('AllQCD')
    assert_equals(set(ps.list_enabled()), set([
        'S -> pi0 pi0', 'S -> pi+ pi-', 'S -> s sbar', 'S -> c cbar', 'S -> g g']))
    ps.enable_all()
    ps.disable('AllQCD')
    assert_equals(set(ps.list_enabled()), set(['S -> e+ e-', 'S -> mu+ mu-']))
