# -*- coding: utf-8 -*-

from __future__ import division

import os
import pandas
import numpy as np
from particletools.tables import PYTHIAParticleData

from . import constants as cst
from . import qcd

_srcdir = os.path.dirname(__file__)

# Meson properties
# ----------------

# Special cases for K_S0 and K_L0
_k0_codes = {'K_S': 310, 'K_L': 130}
_k0_names = {val: key for key, val in _k0_codes.items()}

_meson_df = pandas.read_csv(os.path.join(_srcdir, 'meson_properties.dat'),
                            delim_whitespace=True)
_meson_names = list(_meson_df.Name) + list(_k0_codes.keys())

def _get_meson(feature, value):
    query = _meson_df[_meson_df[feature] == value]
    assert(len(query) <= 1)
    if len(query) < 1:
        raise(ValueError('No meson with {} == {}'.format(feature, value)))
    return query.iloc[0]

def _split_meson_charge(meson_name):
    # Separates the name of the QCD state and its charge
    if len(meson_name) >= 4 and meson_name[-4:] == 'bar0':
        qcd_state, charge = meson_name[:-4], meson_name[-4:]
    elif len(meson_name) >= 1 and meson_name[-1:] in ['0', '+', '-']:
        qcd_state, charge = meson_name[:-1], meson_name[-1:]
    else:
        raise(ValueError('Unrecognised meson name {}'.format(meson_name)))
    return qcd_state, charge

def _get_meson_by_name(meson_name):
    # Special case for neutral kaons
    if meson_name in ['K_S0', 'K_L0']:
        return _get_meson_by_name('K')
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
    if code == 0:
        raise(ValueError('No PDG code for {}.'.format(meson_name)))
    return code

def _get_meson_mass(meson_name):
    record = _get_meson_by_name(meson_name)
    return record.Mass

def _get_meson_spin_code(meson_name):
    record = _get_meson_by_name(meson_name)
    return record.SpinCode

_charges = {
    '0'   :  0,
    'bar0':  0,
    '+'   : +1,
    '-'   : -1,
}

def _get_meson_charge(meson_name):
    _, charge_str = _split_meson_charge(meson_name)
    return _charges[charge_str]

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

def _get_meson_lifetime(meson_name):
    try:
        return cst.meson_lifetimes[meson_name]
    except KeyError:
        raise(ValueError('Lifetime of {} is unknown.'.format(meson_name)))

# Lepton properties
# -----------------

_lepton_masses = {
    'e'  : cst.m_e  ,
    'mu' : cst.m_mu ,
    'tau': cst.m_tau,
}

def _get_lepton_mass(lepton_name):
    if len(lepton_name) >= 1 and lepton_name[-1] in ['-', '+']:
        basename = lepton_name[:-1]
    else:
        basename = lepton_name
    try:
        return _lepton_masses[basename]
    except KeyError:
        raise(ValueError('Unknown lepton {}.'.format(lepton_name)))

def _get_lepton_spin_code(lepton_name):
    return 2

# Generic particle properties
# ---------------------------

_pdata = PYTHIAParticleData()

def _get_generic_pdg_id(particle):
    try:
        return _pdata.pdg_id(particle)
    except:
        raise(ValueError("Particle '{}' not found in PYTHIA database.".format(particle)))

def _get_generic_mass(particle):
    try:
        return _pdata.mass(particle)
    except:
        raise(ValueError("Particle '{}' not found in PYTHIA database.".format(particle)))

# Public API
# ----------

def is_meson(particle):
    if particle in _meson_names:
        return True
    else:
        try:
            basename, charge = _split_meson_charge(particle)
            if basename in _meson_names:
                return True
        except:
            return False
    return False

def get_qcd_state(particle):
    if is_meson(particle):
        qcd_state, charge = _split_meson_charge(particle)
        return qcd_state
    else:
        raise(ValueError('{} is not a meson.'.format(particle)))

def is_lepton(particle):
    if particle in _lepton_masses:
        return True
    elif len(particle) >= 1 and particle[:-1] in _lepton_masses:
        return True
    else:
        return False

def get_pdg_id(particle):
    if is_meson(particle):
        # The PYTHIA database is sometimes inaccurate for mesons, so we
        # manually override it in this specific case.
        return _get_meson_pdg_id(particle)
    else:
        return _get_generic_pdg_id(particle)

def get_name(pdg_id):
    # NOTE: Only mesons are handled so far.
    return _get_meson_by_id(pdg_id)

def get_mass(particle):
    if is_lepton(particle):
        return _get_lepton_mass(particle)
    elif is_meson(particle):
        return _get_meson_mass(particle)
    else:
        try:
            return _get_generic_mass(particle)
        except ValueError:
            raise(ValueError('Mass of {} is unknown.'.format(particle)))

def get_lifetime(particle):
    """
    Returns the particle lifetime (average lifetime in its rest frame), in
    natural units (GeV⁻¹).
    """
    # NOTE: Only mesons are handled so far.
    if is_meson(particle):
        return _get_meson_lifetime(particle)
    else:
        raise(ValueError('Operation not supported for {}.'.format(particle)))

def get_spin_code(particle):
    """
    Returns a positive integer 2S+1 representing the spin S of the particle.
    """
    if is_lepton(particle):
        return _get_lepton_spin_code(particle)
    elif is_meson(particle):
        return _get_meson_spin_code(particle)
    else:
        raise(ValueError('Spin of {} is unknown.'.format(particle)))

def get_charge(particle):
    # NOTE: Only mesons are handled so far.
    if is_meson(particle):
        return _get_meson_charge(particle)
    else:
        raise(ValueError('Operation not supported for {}.'.format(particle)))

def get_parity(particle):
    # NOTE: Only mesons are handled so far.
    if is_meson(particle):
        return _get_meson_parity(particle)
    else:
        raise(ValueError('Operation not supported for {}.'.format(particle)))

def get_abs_strangeness(particle):
    # NOTE: Only mesons are handled so far.
    if is_meson(particle):
        return _get_abs_meson_strangeness(particle)
    else:
        raise(ValueError('Operation not supported for {}.'.format(particle)))

def get_abs_charm(particle):
    # NOTE: Only mesons are handled so far.
    if is_meson(particle):
        return _get_abs_meson_charm(particle)
    else:
        raise(ValueError('Operation not supported for {}.'.format(particle)))

def get_abs_beauty(particle):
    # NOTE: Only mesons are handled so far.
    if is_meson(particle):
        return _get_abs_meson_beauty(particle)
    else:
        raise(ValueError('Operation not supported for {}.'.format(particle)))

# Quark masses and strong coupling constant
# -----------------------------------------

def alpha_s(mu, nf):
    """
    Computes the strong coupling constant α_s at scale μ with nf dynamical flavors
    using the `rundec` package, through the `qcd` wrapper from Wilson.

    Note: we use the *non-squared* scale μ, which has dimension +1, instead of μ².

    Running is computed at 5 loops, and decoupling at 4 loops.

    numpy.vectorize is used to emulate NumPy broadcast rules in `mu`, but is not
    as fast as native vectorization.

    RunDec references:
    * Chetyrkin, K. G., J. H. Kuehn, and M. Steinhauser.
      “RunDec: A Mathematica Package for Running and Decoupling of the Strong Coupling and Quark Masses.”
      Computer Physics Communications 133, no. 1 (December 2000): 43–65.
      https://doi.org/10.1016/S0010-4655(00)00155-7.
    * Schmidt, Barbara, and Matthias Steinhauser.
      “CRunDec: A C++ Package for Running and Decoupling of the Strong Coupling and Quark Masses.”
      Computer Physics Communications 183, no. 9 (September 2012): 1845–48.
      https://doi.org/10.1016/j.cpc.2012.03.023.
    * Herren, Florian, and Matthias Steinhauser.
      “Version 3 of {\tt RunDec} and {\tt CRunDec}.”
      Computer Physics Communications 224 (March 2018): 333–45.
      https://doi.org/10.1016/j.cpc.2017.11.014.

    Wilson references:
    * Website: https://wilson-eft.github.io/
    * Source code: https://github.com/wilson-eft/wilson
    * Paper (for the Wilson RG running & matching, not used here):
      Aebischer, Jason, Jacky Kumar, and David M. Straub. :
      “: A Python Package for the Running and Matching of Wilson Coefficients above and below the Electroweak Scale.”
      The European Physical Journal C 78, no. 12 (December 19, 2018): 1026.
      https://doi.org/10.1140/epjc/s10052-018-6492-7.
    """
    return np.vectorize(lambda _mu: qcd.alpha_s(_mu, nf, alphasMZ=cst.alpha_s_MZ, loop=5), cache=True)(mu)

_pole_masses = {
    'u': None,
    'd': None,
    's': None,
    'c': 1.5,
    'b': 4.8,
    't': cst.m_t_os
}

def on_shell_mass(q):
    """
    Returns the approximate pole mass of the chosen quark.

    This really only makes sense for the top quark.
    """
    try:
        M_q = _pole_masses[q]
    except KeyError:
        raise(ValueError('Unknown quark {}.'.format(q)))
    if M_q is None:
        raise(ValueError('The pole mass is ill-defined for {}.'.format(q)))
    else:
        return M_q

def msbar_mass(q, mu, nf):
    """
    Returns the running quark mass in the MSbar scheme at a scale μ, in a
    theory with nf dynamical flavors.

    We use CRunDec through a slightly modified version of the `wilson.util.qcd` wrapper.
    """
    if q in ['u', 'd', 't']:
        raise(ValueError('MSbar mass not implemented for {} quark.'.format(q)))
    elif q == 's':
        return np.vectorize(
            lambda _mu: qcd.m_s(cst.m_s_msbar_2GeV, _mu, nf, alphasMZ=cst.alpha_s_MZ, loop=5),
            cache=True)(mu)
    elif q == 'c':
        return np.vectorize(
            lambda _mu: qcd.m_c(cst.m_c_si, _mu, nf, alphasMZ=cst.alpha_s_MZ, loop=5),
            cache=True)(mu)
    elif q == 'b':
        return np.vectorize(
            lambda _mu: qcd.m_b(cst.m_b_si, _mu, nf, alphasMZ=cst.alpha_s_MZ, loop=5),
            cache=True)(mu)
    else:
        raise(ValueError('Unknown quark {}.'.format(q)))

_si_masses = {
    'c': cst.m_c_si,
    'b': cst.m_b_si,
    't': cst.m_t_si,
}

def scale_invariant_mass(q):
    """
    Returns the scale-invariant mass of the heavy quark, in the MS-bar scheme.

    For the c and b quarks, we use the 2018 PDG values [1], while for the top
    quark we look for the fixed point of the MS-bar mass with Nf=6, using the
    calculation from [2], which is accurate to order O(αs³) + O(α) + O(α αs),
    and the Higgs mass from the PDG [1].

    [1] M. Tanabashi et al. (Particle Data Group), Phys. Rev. D 98, 030001 (2018)
        DOI: 10.1103/PhysRevD.98.030001

    [2] Jegerlehner, Fred, Mikhail Yu Kalmykov, and Bernd A. Kniehl.
        “On the Difference between the Pole and the MSbar Masses of the Top Quark at the Electroweak Scale.”
        Physics Letters B 722, no. 1–3 (May 2013): 123–29.
        https://doi.org/10.1016/j.physletb.2013.04.012.
    """
    if q in ['u', 'd', 's']:
        raise(ValueError('Scale-invariant mass not implemented for the {} quark.'.format(q)))
    elif q in ['c', 'b', 't']:
        return _si_masses[q]
    else:
        raise(ValueError('Unknown quark {}.'.format(q)))
