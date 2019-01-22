# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import scipy.interpolate as si

from .constants import *

_srcdir = os.path.dirname(__file__)
_table = np.loadtxt(os.path.join(_srcdir, 'alpha_s.dat'))

_interpolator = si.PchipInterpolator(_table[:,0], _table[:,1], extrapolate=False)

def alpha_s(mu):
    """
    Evaluates the strong coupling constant α_s in the MSbar scheme at the scale μ.

    Note: we use the *non-squared* scale μ, which has dimension +1, instead of μ².

    α_s is interpolated using a piecewise cubic Hermite interpolator (PCHIP).
    Evaluating α_s outside the interpolation interval will result in NaNs.

    Source: extracted from LHAPDF 6.1.6, with the CT10nlo PDF set.
        Eur.Phys.J. C75 (2015) 3, 132 (arXiv: 1412.7420)
        Phys.Rev. D82 (2010) 074024 (arXiv:1007.2241)
    """
    return _interpolator(mu)

def _c_sc(x):
    return (9/2*x)**(4/9) * (1 + 0.895*x + 1.371*x**2 + 1.952*x**3)

def _c_cb(x):
    return (25/6*x)**(12/25) * (1 + 1.014*x + 1.389*x**2 + 1.091*x**3)

def _c_bt(x):
    return (23/6*x)**(12/23) * (1 + 1.175*x + 1.501*x**2 + 0.1725*x**3)

def _c_t(x):
    return (7/2*x)**(4/7) * (1 + 1.398*x + 1.793*x**2 - 0.6834*x**3)

# Quark masses in MSbar, evaluated at μ = 2 GeV.
#   Source: F. Sanfilippo, Quark Masses from Lattice QCD, PoS LATTICE2014 (2015) 014, (arXiv: 1505.02794)
m_u_msbar_2GeV =   2.3e-3
m_d_msbar_2GeV =   4.8e-3
m_s_msbar_2GeV =   0.0924
m_c_msbar_2GeV =   1.23
m_b_msbar_2GeV =   4.2
m_t_msbar_2GeV = 173.

_m_q_msbar_2GeV = {
    'u': m_u_msbar_2GeV,
    'd': m_d_msbar_2GeV,
    's': m_s_msbar_2GeV,
    'c': m_c_msbar_2GeV,
    'b': m_b_msbar_2GeV,
    't': m_t_msbar_2GeV,
}

def _c(x, mu):
    """
    Coefficient c(x) used to evaluate the quark mass at a different scale in MSbar.
    """
    # Cast everything to NumPy arrays for the masking to work properly.
    x = np.array(x)
    mu = np.array(mu)
    mask_sc = (mu > m_s_msbar_2GeV) & (mu <= m_c_msbar_2GeV)
    mask_cb = (mu > m_c_msbar_2GeV) & (mu <= m_b_msbar_2GeV)
    mask_bt = (mu > m_b_msbar_2GeV) & (mu <= m_t_msbar_2GeV)
    mask_t =  (mu > m_t_msbar_2GeV)
    res = np.zeros_like(np.array(x))
    res[mask_sc] = _c_sc(x[mask_sc])
    res[mask_cb] = _c_cb(x[mask_cb])
    res[mask_bt] = _c_bt(x[mask_bt])
    res[mask_t ] = _c_t( x[mask_t ])
    res[~(mask_sc | mask_cb | mask_bt | mask_t)] = float('nan')
    return res

def running_mass(mq0, mu0, mu1):
    """
    Evaluates the MSbar quark mass m_q at a new scale μ_1, in terms of the mass m_q(μ_0).

    FIXME: this currently fails validation!
    """
    return mq0 * ( _c(alpha_s(mu1)/pi, mu1) / _c(alpha_s(mu0)/pi, mu0) )

all_quarks = ['u', 'd', 's', 'c', 'b', 't']

def get_quark_mass(q, mu=2.0):
    """
    Returns the mass of quark `q` at scale `mu` (default = 2.0 GeV).
    """
    if q in all_quarks:
        m_q_msbar_2GeV = _m_q_msbar_2GeV[q]
    else:
        raise(ValueError('Unknown quark {}.'.format(q)))
    return running_mass(m_q_msbar_2GeV, 2.0, mu)
