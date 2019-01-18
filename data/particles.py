# -*- coding: utf-8 -*-

from __future__ import division

import os
import pandas

_srcdir = os.path.dirname(__file__)
_meson_df = pandas.read_table(os.path.join(_srcdir, 'meson_properties.dat'))

# Special cases for K_S0 and K_L0
_k0_codes = {'K_S': 310, 'K_L': 130}
_k0_names = {val: key for key, val in _k0_codes.items()}

def _get_meson(feature, value):
    query = _meson_df[_meson_df[feature] == value]
    assert(len(query) <= 1)
    if len(query) < 1:
        raise(ValueError('No meson with {} == {}'.format(feature, value)))
    return query.iloc[0]

def _get_meson_by_name(meson_name):
    try:
        # Handle mesons specified without the electric charge (e.g. `K*`)
        return _get_meson('Name', meson_name)
    except:
        # Handle mesons specified with the charge (e.g. `K*0`)
        qcd_state, charge = _split_meson_charge(meson_name)
        return _get_meson('Name', qcd_state)

def _get_meson_by_id(pdg_id):
    # Handle the special case of K_S0 and K_L0
    if pdg_id in _k0_names:
        return _k0_names[pdg_id] + '0'
    # Handle the general case
    query = _meson_df[(_meson_df['IdZero'] == abs(pdg_id))
                      | (_meson_df['IdPlus'] == abs(pdg_id))]
    assert(len(query) <= 1)
    if len(query) < 1:
        raise(ValueError('No meson corresponding to PDG code {}'.format(pdg_id)))
    else:
        record = query.iloc[0]
    qcd_state = record.Name
    # Infer the charge (and 'bar') from the PDG code
    if +record.IdZero == pdg_id:
        charge = '0'
    elif -record.IdZero == pdg_id:
        charge = 'bar0'
    elif +record.IdPlus == pdg_id:
        charge = '+'
    elif -record.IdPlus == pdg_id:
        charge = '-'
    else:
        assert(False) # pragma: no cover
    fullname = qcd_state + charge
    return fullname

def _split_meson_charge(meson_name):
    # Separates the name of the QCD state and its charge
    if len(meson_name) >= 4 and meson_name[-4:] == 'bar0':
        qcd_state, charge = meson_name[:-4], meson_name[-4:]
    elif len(meson_name) >= 1 and meson_name[-1:] in ['0', '+', '-']:
        qcd_state, charge = meson_name[:-1], meson_name[-1:]
    else:
        raise(ValueError('Unrecognised meson name {}'.format(meson_name)))
    return qcd_state, charge

def _get_meson_pdg_id(meson_name):
    qcd_state, charge = _split_meson_charge(meson_name)
    # Handle special PDG codes for K_S0 and K_L0
    if qcd_state in _k0_codes:
        if charge != '0':
            raise(ValueError('Invalid particle string {}'.format(meson_name)))
        return _k0_codes[qcd_state]
    # Now handle the generic case
    record = _get_meson_by_name(qcd_state)
    # Infer the PDG code from the charge (and the 'bar')
    if charge == '0':
        code = +record.IdZero
    elif charge == 'bar0':
        if record.SelfConjugate:
            raise(ValueError('{}0 is self-conjugate!'.format(qcd_state)))
        code = -record.IdZero
    elif charge == '+':
        code = +record.IdPlus
    elif charge == '-':
        code = -record.IdPlus
    else:
        assert(False) # pragma: no cover
    return code

def _get_meson_mass(meson_name):
    record = _get_meson_by_name(meson_name)
    return record.Mass

def _get_meson_spin_code(meson_name):
    record = _get_meson_by_name(meson_name)
    return record.SpinCode

def _get_meson_parity(meson_name):
    record = _get_meson_by_name(meson_name)
    return record.Parity

def _get_abs_meson_strangeness(meson_name):
    record = _get_meson_by_name(meson_name)
    return abs(record.S)

def _get_abs_meson_charm(meson_name):
    record = _get_meson_by_name(meson_name)
    return abs(record.C)

def _get_abs_meson_beauty(meson_name):
    record = _get_meson_by_name(meson_name)
    return abs(record.B)

def get_pdg_id(particle):
    # NOTE: Only mesons are handled so far.
    return _get_meson_pdg_id(particle)

def get_name(pdg_id):
    # NOTE: Only mesons are handled so far.
    return _get_meson_by_id(pdg_id)

def get_mass(particle):
    # NOTE: Only mesons are handled so far.
    return _get_meson_mass(particle)

def get_spin_code(particle):
    """
    Returns a positive integer 2S+1 representing the spin S of the particle.
    """
    # NOTE: Only mesons are handled so far.
    return _get_meson_spin_code(particle)

def get_parity(particle):
    # NOTE: Only mesons are handled so far.
    return _get_meson_parity(particle)

def get_abs_strangeness(particle):
    # NOTE: Only mesons are handled so far.
    return _get_abs_meson_strangeness(particle)

def get_abs_charm(particle):
    # NOTE: Only mesons are handled so far.
    return _get_abs_meson_charm(particle)

def get_abs_beauty(particle):
    # NOTE: Only mesons are handled so far.
    return _get_abs_meson_beauty(particle)
