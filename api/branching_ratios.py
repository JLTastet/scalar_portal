# -*- coding: utf-8 -*-

from __future__ import absolute_import, division
from future.utils import viewitems, with_metaclass
import abc # Abstract Base Classes

# We use OrderedDict to obtain stable, deterministic results, and to make sure
# particle definitions always come before decay channel ones.
from collections import OrderedDict
import numpy as np

from ..api.channel import Channel
from ..data.constants import second, c_si, default_scalar_id


def format_pythia_particle_string(
        pdg_id, name, antiname, spin_type, charge_type, mass, lifetime_si,
        new=True, may_decay=True, is_visible=False):
    lifetime_mm = 1e3 * lifetime_si * c_si
    all_prop = '{}:{} = {} {} {} {} 0 {} 0.0 0.0 0.0 {:.12}'.format(
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
    def __init__(self, channels, mass, couplings,
                 ignore_invalid=False,
                 scalar_id=default_scalar_id):
        self._channels = OrderedDict((str(ch), ch) for ch in channels)
        self._mS = np.asarray(mass, dtype='float')
        try:
            self._couplings = { k: np.array(v, dtype='float')
                                for k, v in viewitems(couplings) }
        except AttributeError:
            raise(ValueError("'couplings' should be a dictionary (e.g. `{'theta': 1}`)."))
        try:
            bc = np.broadcast(self._mS, *self._couplings.values())
        except ValueError:
            raise(ValueError('Mass and coupling arrays could not be broadcast together.'))
        self._ndim = bc.nd
        self._scalar_id = scalar_id
        self._widths = OrderedDict((str(ch), ch.width(mass, self._couplings)) for ch in channels)
        if ignore_invalid:
            for w in self._widths.values():
                w[np.isnan(w)] = 0

    @property
    def widths(self):
        return self._widths

    @property
    @abc.abstractmethod
    def branching_ratios(self):
        pass # pragma: no cover

    def pythia_strings(self):
        if self._ndim > 0:
            raise(ValueError('Can only generate PYTHIA strings for a single mass and coupling.'))
        for ch, br in viewitems(self.branching_ratios):
            if not np.isfinite(br):
                raise(ValueError('Cannot generate PYTHIA string: invalid channel {} for m = {}.'.format(ch, self._mS)))
        return OrderedDict(
            (ch_str, channel.pythia_string(self.branching_ratios[ch_str], self._scalar_id))
            for ch_str, channel in viewitems(self._channels))


class DecayBranchingRatios(BranchingRatios):
    '''
    Represents a set of decay branching ratios for the scalar.
    '''
    def __init__(self, *args, **kwargs):
        super(DecayBranchingRatios, self).__init__(*args, **kwargs)
        self._total_width = sum(self.widths.values())
        with np.errstate(invalid='ignore'):
            self._br = OrderedDict(
                (ch_str, w / self._total_width) for ch_str, w in viewitems(self.widths))

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
            pdg_id=self._scalar_id, name='S', antiname='void', spin_type=1,
            charge_type=0, mass=self._mS, lifetime_si=self.lifetime_si,
            new=new, may_decay=True, is_visible=False)


class ProductionBranchingRatios(BranchingRatios):
    '''
    Represents a set of production branching ratios for the scalar.
    '''
    def __init__(self, *args, **kwargs):
        super(ProductionBranchingRatios, self).__init__(*args, **kwargs)
        self._br = OrderedDict(
            (st, w / self._channels[st].parent_width) for st, w in viewitems(self.widths))

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
    def decay(self):
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
        decay_strs = self.decay.pythia_strings()
        full_string = '\n'.join(
            [particle_str] +
            list(st for st in production_strs.values() if st is not None) +
            list(st for st in decay_strs.values()      if st is not None))
        return full_string
