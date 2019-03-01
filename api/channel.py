# -*- coding: utf-8 -*-

from __future__ import absolute_import
from future.utils import with_metaclass
import abc # Abstract Base Classes

from ..data.particles import *


def _to_channel_str(parent, children):
    return parent + ' -> ' + ' '.join(children)

def _from_channel_str(string):
    try:
        parents_str, children_str = string.split(' -> ')
        parents = parents_str.split(' ')
        if len(parents) != 1:
            raise(ValueError('A decay channel can only contain one parent particle.'))
        children = children_str.split()
    except:
        raise(ValueError("Failed to parse channel '{}'.".format(string)))
    return parents[0], children


class Channel(with_metaclass(abc.ABCMeta, object)):
    '''
    Wraps a decay channel containing the scalar particle either in the initial
    or final state.
    '''
    def __init__(self, parent, children):
        self._parent = parent
        self._children = children
        self._str = _to_channel_str(parent, children)

    @abc.abstractmethod
    def is_open(self, mS):
        '''
        Whether the decay channel is kinematically open.

        The default implementation assumes all particles to be on-shell, and
        should be overridden if this is not the case (e.g. S -> q qbar).
        '''
        pass # pragma: no cover

    @abc.abstractmethod
    def normalized_width(self, mS):
        '''
        Returns the width for this channel, assuming a unit mixing angle: θ = 1.
        '''
        pass # pragma: no cover

    def width(self, mS, theta):
        '''
        Returns the width for this channel for an arbitrary mixing angle θ.

        The default implementation assumes θ² scaling. It should be overridden
        if this is not the case.
        '''
        return theta**2 * self.normalized_width(mS)

    @abc.abstractmethod
    def pythia_string(self, branching_ratio, scalar_id):
        '''
        Returns a string which can be directly read by the PYTHIA event
        generator to enable the channel.

        The default implementation assumes pure phase space decay. It should be
        overridden if the final state hadronizes or if its multiplicity is
        larger than 2.
        '''
        pass # pragma: no cover


class DecayChannel(Channel):
    '''
    Wraps a decay channel of the scalar particle.
    '''
    def __init__(self, children):
        super(DecayChannel, self).__init__('S', children)

    def is_open(self, mS):
        threshold = sum(get_mass(child) for child in self._children)
        return mS > threshold

    def pythia_string(self, branching_ratio, scalar_id):
        return '{}:addChannel = 1 {} 0 {}'.format(
            scalar_id, branching_ratio,
            ' '.join(str(get_pdg_id(child)) for child in self._children))
