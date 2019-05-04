# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import

from ..data.constants import *
from ..data.particles import *
from ..api.channel import ProductionChannel
from . import hadronic_common as h

import numpy as np
import scipy.integrate
from warnings import warn


def normalized_decay_width(X, X1, mS, eps=1e-3):
    '''
    Computes the decay width for the process X -> X' S S, divided by the
    coefficient α.
    '''
    mS = np.asarray(mS, dtype='float')
    mX = get_mass(X)
    mX1 = get_mass(X1)
    xi = h._get_xi(X, X1)
    # Part of the prefactor has been absorbed in the normalized amplitude.
    prefactor = xi**2 / (512*pi**3 * mX**3 * v**2 * M_h**4)
    # Integration
    lower_bound = 4*mS**2
    upper_bound = (mX-mX1)**2
    def E2(q2):
        return np.sqrt(q2) / 2
    def E3(q2):
        return (mX**2 - q2 - mX1**2) / (2*np.sqrt(q2))
    M = h.get_matrix_element(X, X1)
    def integrand(q2, mS):
        A = M(q2)
        return np.real(A*np.conj(A)) * np.sqrt(E2(q2)**2 - mS**2) * np.sqrt(E3(q2)**2 - mX1**2)
    closing_mass = (mX - mX1) / 2
    def width(mS, low, high):
        if mS < closing_mass:
            val, _ = scipy.integrate.quad(lambda q2: integrand(q2, mS),
                                          low, high, epsabs=0, epsrel=eps)
            return prefactor * val
        else:
            return 0.
    return np.vectorize(width, otypes=[float])(mS, lower_bound, upper_bound)


class ThreeBodyQuartic(ProductionChannel):
    '''
    Quartic scalar production through the exclusive 3-body hadronic decay H ->
    H' S S.

    If the parent particle is not a weak eigenstate (e.g. due to neutral kaon
    mixing), then a weak eigenstate (either particle or antiparticle) must be
    specified separately.

    `eps` is the relative accuracy target for the numerical integration.
    '''
    def __init__(self, H, H1, weak_eigenstate=None, eps=1e-3):
        if weak_eigenstate is None:
            weak_eigenstate = H
        if not (is_meson(weak_eigenstate) and is_meson(H1)):
            raise(ValueError('{} and {} must be mesons.'.format(weak_eigenstate, H1)))
        super(ThreeBodyQuartic, self).__init__(H, [H1], coefficient='alpha', NS=2)
        try:
            self._X  = get_qcd_state(weak_eigenstate)
            self._X1 = get_qcd_state(H1)
        except:
            raise(ValueError('The charges of {} and {} must be specified.'.format(weak_eigenstate, H1)))
        self._eps = eps

    def normalized_width(self, mS):
        return normalized_decay_width(self._X, self._X1, mS, eps=self._eps)

    def pythia_string(self, *args, **kwargs):
        warn('Assuming pure phase-space decay for {}'.format(str(self)))
        return super(ThreeBodyQuartic, self).pythia_string(*args, **kwargs)
