# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

from .constants import *
from .particles import *

from numpy import sqrt
from math import sin, cos

INF = float('inf')

mB = get_mass('B+')

_form_factors = {}

def get_form_factor(Y, Yprime):
    try:
        return _form_factors[(Y, Yprime)]
    except KeyError:
        raise(ValueError('No form factor for the {} -> {} transition.'.format(Y, Yprime)))

# Pseudoscalar form factors (section C.1.1)
# -----------------------------------------

def _pseudoscalar_f0_form_factor(F0, m_fit):
    return lambda q2: F0 / (1 - q2/m_fit**2)

_form_factors[('B', 'K' )] = _pseudoscalar_f0_form_factor(0.33 , sqrt(38))
_form_factors[('B', 'pi')] = _pseudoscalar_f0_form_factor(0.258, sqrt(38))
_form_factors[('K', 'pi')] = _pseudoscalar_f0_form_factor(0.96 , INF     )

# Scalar form factors (section C.1.2)
# -----------------------------------

def _scalar_fplus_form_factor(F0, a, b):
    return lambda q2: F0 / ( 1 - a*(q2/mB**2) + b*(q2/mB**2)**2 )

def _scalar_f0_form_factor(F0, a, b, mS):
    fplus = _scalar_fplus_form_factor(F0, a, b)
    return lambda q2: fplus(q2) * (1 - q2 / (mB**2 - mS**2))

_form_factors[('B', 'K*_0(700)' )] = \
    _scalar_f0_form_factor(0.46, 1.6, 1.35, get_mass('K*_0(700)' ))
_form_factors[('B', 'K*_0(1430)')] = \
    _scalar_f0_form_factor(0.17, 4.4, 6.4 , get_mass('K*_0(1430)'))

# Vector form factors (section C.2.1)
# -----------------------------------

def _vector_A0_Kstar892_form_factor(r1, r2, m_fit):
    return lambda q2: r1 / (1 - q2/mB**2) + r2 / ( 1 - q2/m_fit**2 )

def _xi(xi0):
    return lambda q2: xi0 / (1 - q2/mB**2)

def _vector_A0_Kstar_form_factor(xi0_perp, xi0_par, mV):
    xi_perp = _xi(xi0_perp)
    xi_par  = _xi(xi0_par )
    def A0(q2):
        EV = mB/2 * ( 1 - q2/mB**2 + (mV/mB)**2 )
        return (1 - mV**2/(mB*EV)) * xi_par(q2) + (mV/mB) * xi_perp(q2)
    return A0

_form_factors[('B', 'K*')] = \
     _vector_A0_Kstar892_form_factor(1.364, -0.99, sqrt(36.8))
_form_factors[('B', 'K*(1410)')] = \
    _vector_A0_Kstar_form_factor(0.28, 0.22, get_mass('K*(1410)'))
_form_factors[('B', 'K*(1680)')] = \
    _vector_A0_Kstar_form_factor(0.24, 0.18, get_mass('K*(1680)'))

# Pseudo-vector (section C.2.2)
# -----------------------------

def _pseudovector_VAB_form_factor(F0, a, b):
    return lambda q2: F0 / (1 - a*(q2/mB**2) + b*(q2/mB**2)**2)

# Masses of the K_1A and K_1B (assuming zero quark masses)
m_K1A = 1.31 # GeV
m_K1B = 1.34 # GeV

# Mixing angle between K_1(1270), K_1(1400) and K_1A, K_2A.
theta_K1 = -34 * degree

_V0A = _pseudovector_VAB_form_factor( 0.22, 2.4 , 1.78)
_V0B = _pseudovector_VAB_form_factor(-0.45, 1.34, 0.69)

def _pseudovector_V_B_K1_form_factor(m_K1, xA, xB):
    return lambda q2: (xA * m_K1A * _V0A(q2) + xB * m_K1B * _V0B(q2) ) / m_K1

_form_factors[('B', 'K_1(1270)')] = \
    _pseudovector_V_B_K1_form_factor(get_mass('K_1(1270)'), sin(theta_K1),  cos(theta_K1))
_form_factors[('B', 'K_1(1400)')] = \
    _pseudovector_V_B_K1_form_factor(get_mass('K_1(1400)'), cos(theta_K1), -sin(theta_K1))

# Tensor final meson state
# ------------------------

def _tensor_A0_form_factor(F0, aT, bT):
    return lambda q2: F0 / ( (1-q2/mB**2) * (1-aT*(q2/mB**2)+bT*(q2/mB**2)**2) )

_form_factors[('B', 'K*_2(1430)')] = _tensor_A0_form_factor(0.23, 1.23, 0.76)
