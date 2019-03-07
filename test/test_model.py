# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..api.model import Model
from ..api.branching_ratios import BranchingRatiosResult
from ..data.constants import default_scalar_id

def test_model():
    m = Model()
    assert_equals(m.scalar_pdg_id, default_scalar_id)
    m.decays.enable('LightScalar')
    m.production.disable('K -> S pi')

def test_groups():
    m_ref = Model()
    m_ref.production.enable_all()
    m_ref.decays.enable_all()
    m = Model()
    m.production.enable('K -> S pi')
    m.production.enable('B -> S pi')
    m.production.enable('B -> S K?')
    m.decays.enable('LightScalar')
    m.decays.enable('HeavyScalar')
    lp = m.production.list_enabled()
    lp_ref = m_ref.production.list_enabled()
    assert_equals(set(lp), set(lp_ref))
    ld = m.decays.list_enabled()
    ld_ref = m_ref.decays.list_enabled()
    assert_equals(set(ld), set(ld_ref))

def test_channels():
    m = Model()
    m.production.enable_all()
    m.decays.enable_all()
    prod_ch  = m.production.get_active_processes()
    decay_ch = m.decays.get_active_processes()
    mS = np.array([0.5, 1.5, 3])
    for ch in prod_ch:
        ch.normalized_width(mS)
    for ch in decay_ch:
        ch.normalized_width(mS)

def test_result():
    m = Model()
    m.production.enable_all()
    m.decays.enable('LightScalar')
    mS = np.array([0.1, 0.5, 1])
    res = m.compute_branching_ratios(mS, 0.25)
    assert(isinstance(res, BranchingRatiosResult))
