# -*- coding: utf-8 -*-

from __future__ import absolute_import, division
from future.utils import viewitems, with_metaclass
import abc # Abstract Base Classes

import numpy as np

from ..api.channel import Channel
from ..data.constants import second, c_si, default_scalar_id


def format_pythia_particle_string(
        pdg_id, name, antiname, spin_type, charge_type, mass, lifetime_si,
        new=True, may_decay=True, is_visible=False):
    lifetime_mm = 1e3 * lifetime_si * c_si
    all_prop = '{}:{} = {} {} {} {} 0 {} 0.0 0.0 0.0 {}'.format(
        pdg_id, 'new' if new else 'all', name, antiname, spin_type, charge_type,
        mass, lifetime_mm)
    is_resonance = '{}:isResonance = false'.format(pdg_id)
    may_decay = '{}:mayDecay = {}'.format(pdg_id, str(may_decay).lower())
    is_visible = '{}:isVisible = {}'.format(pdg_id, str(is_visible).lower())
    return '\n'.join([all_prop, is_resonance, may_decay, is_visible])


class BranchingRatios(with_metaclass(abc.ABCMeta, object)):
    '''
    Represents a set of computed branching ratios.
    '''
    def __init__(self, channels, mass, coupling,
                 ignore_invalid=False,
                 scalar_id=default_scalar_id):
        self._channels = {str(ch): ch for ch in channels}
        self._mS = np.asarray(mass, dtype='float')
        self._coupling = np.asarray(coupling, dtype='float')
        self._scalar_id = scalar_id
        self._width = {str(ch): ch.width(mass, coupling) for ch in channels}
        if ignore_invalid:
            for w in self._width.values():
                w[np.isnan(w)] = 0

    @property
    def width(self):
        return self._width

    @property
    @abc.abstractmethod
    def branching_ratios(self):
        pass # pragma: no cover

    def pythia_strings(self):
        if self._mS.ndim > 0 or self._coupling.ndim > 0:
            raise(ValueError('Can only generate PYTHIA strings for a single mass and coupling.'))
        return {ch_str: channel.pythia_string(self.branching_ratios[ch_str], self._scalar_id)
                for ch_str, channel in viewitems(self._channels)}


class DecayBranchingRatios(BranchingRatios):
    '''
    Represents a set of decay branching ratios for the scalar.
    '''
    def __init__(self, *args, **kwargs):
        super(DecayBranchingRatios, self).__init__(*args, **kwargs)
        self._total_width = sum(self._width.values())
        self._br = {
            ch_str: w / self._total_width for ch_str, w in viewitems(self._width)}

    @property
    def total_width(self):
        return self._total_width

    @property
    def lifetime_si(self):
        tau = 1 / self._total_width
        return tau / second

    @property
    def branching_ratios(self):
        return self._br

    def pythia_particle_string(self, new=True):
        '''
        Returns a string which can be directly read by the PYTHIA event
        generator to add the scalar particle.
        '''
        if self._mS.ndim > 0:
            raise(ValueError('Can only generate a PYTHIA string for a single scalar mass.'))
        return format_pythia_particle_string(
            pdg_id=self._scalar_id, name='S', antiname='S', spin_type=1,
            charge_type=0, mass=self._mS, lifetime_si=self.lifetime_si,
            new=new, may_decay=True, is_visible=False)


class ProductionBranchingRatios(BranchingRatios):
    '''
    Represents a set of production branching ratios for the scalar.
    '''
    def __init__(self, *args, **kwargs):
        super(ProductionBranchingRatios, self).__init__(*args, **kwargs)
        self._br = {
            st: w / self._channels[st].parent_width for st, w in viewitems(self._width)}

    @property
    def branching_ratios(self):
        return self._br


class BranchingRatiosResult(object):
    '''
    Utility class wrapping the result of a computation of both production and
    decay branching ratios.

    Aggregates `ProductionBranchingRatios` and `DecayBranchingRatios`, and
    provides shortcuts for methods related to the scalar particle itself.
    '''
    def __init__(self, prod, decay):
        self._prod  = prod
        self._decay = decay

    @property
    def production(self):
        return self._prod

    @property
    def decays(self):
        return self._decay

    @property
    def total_width(self):
        return self._decay.total_width

    @property
    def lifetime_si(self):
        return self._decay.lifetime_si

    def pythia_particle_string(self, new=True):
        return self._decay.pythia_particle_string(new)

    def pythia_full_string(self):
        particle_str = self.pythia_particle_string()
        production_strs = self.production.pythia_strings()
        decay_strs = self.decays.pythia_strings()
        full_string = '\n'.join(
            [particle_str] + production_strs.values() + decay_strs.values())
        return full_string
