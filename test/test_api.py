# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises

from .. import (Model, Channel, ProductionChannel, DecayChannel, ActiveProcesses,
                BranchingRatios, DecayBranchingRatios, ProductionBranchingRatios)

def test_api():
    assert('Model'                     in globals())
    assert('Channel'                   in globals())
    assert('ProductionChannel'         in globals())
    assert('DecayChannel'              in globals())
    assert('ActiveProcesses'           in globals())
    assert('BranchingRatios'           in globals())
    assert('DecayBranchingRatios'      in globals())
    assert('ProductionBranchingRatios' in globals())
