# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..api.branching_ratios import *
from ..decay.leptonic import Leptonic
from ..decay.two_gluons import TwoGluons
from ..production.two_body_hadronic import TwoBodyHadronic

def test_decay_branching_ratios():
    channels = [Leptonic(l) for l in ['e', 'mu', 'tau']]
    mS = np.array([0.1, 1.5, 3])
    br = DecayBranchingRatios(channels, mS, 1)
    ref_widths = {str(ch): ch.normalized_width(mS) for ch in channels}
    ref_total_width = sum(ref_widths.values())
    epsilon = 1e-14
    assert(np.all(np.abs(br.total_width - ref_total_width) <= epsilon * ref_total_width))
    for ch in ref_widths.keys():
        assert(np.all(np.abs(br.width[ch] - ref_widths[ch]) <= epsilon * ref_widths[ch]))
        assert(np.all(np.abs(ref_total_width*br.branching_ratios[ch] - ref_widths[ch])
                      <= epsilon * ref_widths[ch]))
    assert_raises(ValueError, lambda: br.pythia_strings())
    assert_raises(ValueError, lambda: br.pythia_particle_string())
    assert(np.all(np.abs(br.lifetime_si * second * ref_total_width - 1) <= epsilon))

def test_invalid_branching_ratio():
    mS = np.array([0.5, 1.5, 2.5])
    br0 = DecayBranchingRatios([Leptonic('e')             ], mS, 1)
    br1 = DecayBranchingRatios([Leptonic('e'), TwoGluons()], mS, 1)
    br2 = DecayBranchingRatios([Leptonic('e'), TwoGluons()], mS, 1, ignore_invalid=True)
    w0 = br0.total_width
    w1 = br1.total_width
    w2 = br2.total_width
    assert_equals(list(np.isnan(w1)), [True, True, False])
    assert(np.all(~np.isnan(w2)))
    assert_equals(list(w0 == w2), [True, True, False])

def test_decay_pythia_strings():
    channels = [Leptonic(l) for l in ['e', 'mu', 'tau']]
    br = DecayBranchingRatios(channels, 0.5, 1)
    strings = br.pythia_strings()
    assert_equals(strings['S -> e+ e-'    ], '9900025:addChannel = 1 3.14194722291e-05 0 -11 11')
    assert_equals(strings['S -> mu+ mu-'  ], '9900025:addChannel = 1 0.999968580528 0 -13 13'   )
    assert_equals(strings['S -> tau+ tau-'], '9900025:addChannel = 1 0.0 0 -15 15'              )
    assert_equals(br.pythia_particle_string(),
                  '9900025:new = S S 1 0 0 0.5 0.0 0.0 0.0 7.22250988672e-05\n' +
                  '9900025:isResonance = false\n' +
                  '9900025:mayDecay = true\n' +
                  '9900025:isVisible = false')
    assert_equals(br.pythia_particle_string(new=False),
                  '9900025:all = S S 1 0 0 0.5 0.0 0.0 0.0 7.22250988672e-05\n' +
                  '9900025:isResonance = false\n' +
                  '9900025:mayDecay = true\n' +
                  '9900025:isVisible = false')

def test_production_branching_ratios():
    channels = [TwoBodyHadronic('B+', 'pi+'), TwoBodyHadronic('B+', 'K*_2(1430)+')]
    mS = np.array([1, 4])
    br = ProductionBranchingRatios(channels, mS, 1)
    br_pi = br.branching_ratios['B+ -> S pi+'        ]
    br_K2 = br.branching_ratios['B+ -> S K*_2(1430)+']
    maxbr = br.maximum_branching_ratio
    assert_equals(list(br_K2 > br_pi), [True, False])
