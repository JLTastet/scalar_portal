# -*- coding: utf-8 -*-

from __future__ import absolute_import

from nose.tools import assert_equals, assert_raises
import numpy as np

from ..api.model import Model

def test_model():
    m = Model()
    assert_equals(m.scalar_pdg_id, 9900025)
    m.decays.disable_all()
    m.decays.enable('LightScalar')
    m.production.disable('K -> S pi')

def test_groups():
    m_ref = Model()
    m = Model()
    m.production.disable_all()
    m.decays.disable_all()
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
    prod_ch  = m.production.get_active_processes()
    decay_ch = m.decays.get_active_processes()
    mS = np.array([0.5, 1.5, 3])
    for ch in prod_ch:
        ch.normalized_width(mS)
    for ch in decay_ch:
        ch.normalized_width(mS)
