# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..api.branching_ratios import *
from ..decay.leptonic import Leptonic
from ..decay.two_gluons import TwoGluons
from ..production.two_body_hadronic import TwoBodyHadronic
from ..production.two_body_quartic import TwoBodyQuartic

def test_decay_branching_ratios():
    channels = [Leptonic(l) for l in ['e', 'mu', 'tau']]
    mS = np.array([0.1, 1.5, 3])
    br = DecayBranchingRatios(channels, mS, {'theta': 1})
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
    br0 = DecayBranchingRatios([Leptonic('e')             ], mS, {'theta': 1})
    br1 = DecayBranchingRatios([Leptonic('e'), TwoGluons()], mS, {'theta': 1})
    br2 = DecayBranchingRatios([Leptonic('e'), TwoGluons()], mS, {'theta': 1}, ignore_invalid=True)
    w0 = br0.total_width
    w1 = br1.total_width
    w2 = br2.total_width
    assert_equals(list(np.isnan(w1)), [True, True, False])
    assert(np.all(~np.isnan(w2)))
    assert_equals(list(w0 == w2), [True, True, False])

def test_decay_pythia_strings():
    channels = [Leptonic(l) for l in ['e', 'mu', 'tau']]
    br = DecayBranchingRatios(channels, 0.5, {'theta': 1})
    strings = br.pythia_strings()
    assert_equals(strings['S -> e+ e-'    ], '9900025:addChannel = 1 3.14194722291e-05 0 -11 11')
    assert_equals(strings['S -> mu+ mu-'  ], '9900025:addChannel = 1 0.999968580528 0 -13 13'   )
    assert_equals(strings['S -> tau+ tau-'], '9900025:addChannel = 1 0.0 0 -15 15'              )
    assert_equals(br.pythia_particle_string(),
                  '9900025:new = S void 1 0 0 0.5 0.0 0.0 0.0 7.22250988672e-05\n' +
                  '9900025:isResonance = false\n' +
                  '9900025:mayDecay = true\n' +
                  '9900025:isVisible = false')
    assert_equals(br.pythia_particle_string(new=False),
                  '9900025:all = S void 1 0 0 0.5 0.0 0.0 0.0 7.22250988672e-05\n' +
                  '9900025:isResonance = false\n' +
                  '9900025:mayDecay = true\n' +
                  '9900025:isVisible = false')

def test_production_branching_ratios():
    channels = [TwoBodyHadronic('B+', 'pi+'), TwoBodyHadronic('B+', 'K*_2(1430)+')]
    mS = np.array([1, 4])
    br = ProductionBranchingRatios(channels, mS, {'theta': 1})
    br_pi = br.branching_ratios['B+ -> S pi+'        ]
    br_K2 = br.branching_ratios['B+ -> S K*_2(1430)+']
    assert_equals(list(br_K2 > br_pi), [True, False])

def test_branching_ratio_result():
    production_channels = [TwoBodyHadronic('B+', 'pi+'), TwoBodyHadronic('B+', 'K*_2(1430)+')]
    decay_channels = [Leptonic(l) for l in ['e', 'mu', 'tau']]
    mS = np.array([0.5, 1.5, 2.5])
    production_br = ProductionBranchingRatios(production_channels, mS, {'theta': 1})
    decay_br = DecayBranchingRatios(decay_channels, mS, {'theta': 1})
    res = BranchingRatiosResult(production_br, decay_br)
    assert(np.all(res.total_width == decay_br.total_width))
    assert(np.all(res.lifetime_si == decay_br.lifetime_si))
    assert(res.production is production_br)
    assert(res.decays     is decay_br     )

def test_result_strings():
    production_channels = [TwoBodyHadronic('B+', 'pi+'), TwoBodyHadronic('B+', 'K*_2(1430)+')]
    decay_channels = [Leptonic(l) for l in ['e', 'mu', 'tau']]
    production_br = ProductionBranchingRatios(production_channels, 0.5, {'theta': 1})
    decay_br = DecayBranchingRatios(decay_channels, 0.5, {'theta': 1})
    res = BranchingRatiosResult(production_br, decay_br)
    assert_equals(res.pythia_particle_string(), decay_br.pythia_particle_string())
    assert_equals(res.pythia_full_string(), '''\
9900025:new = S void 1 0 0 0.5 0.0 0.0 0.0 7.22250988672e-05
9900025:isResonance = false
9900025:mayDecay = true
9900025:isVisible = false
521:addChannel = 1 0.0123084672076 0 9900025 211
521:addChannel = 1 0.355312987843 0 9900025 325
9900025:addChannel = 1 3.14194722291e-05 0 -11 11
9900025:addChannel = 1 0.999968580528 0 -13 13
9900025:addChannel = 1 0.0 0 -15 15'''
    )

def test_invalid():
    br = DecayBranchingRatios([TwoGluons()], 1.5, {'theta': 1})
    assert_raises(ValueError, lambda: br.pythia_strings())
    assert_raises(ValueError, lambda: DecayBranchingRatios([TwoGluons()], 1.5, 1))

def test_broadcasting():
    channels = [TwoBodyHadronic('B0', 'pi0'), TwoBodyQuartic('B0')]
    ProductionBranchingRatios(channels, 1., {'theta': [0.1, 1], 'alpha': 0       })
    ProductionBranchingRatios(channels, 1., {'theta': 0       , 'alpha': [0.1, 1]})
    ProductionBranchingRatios(channels, 1., {'theta': [0.1, 1], 'alpha': [0.1, 1]})
    ProductionBranchingRatios(channels, [0., 1.], {'theta': 1 , 'alpha': 0       })
    assert_raises(ValueError, lambda: ProductionBranchingRatios(
        channels, [0., 0.5, 1.], {'theta': [0.1, 1], 'alpha': 0}))
    assert_raises(ValueError, lambda: ProductionBranchingRatios(
        channels, 1., {'theta': [0.1, 0.25, 1], 'alpha': [0.1, 0.5]}))
