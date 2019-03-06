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

def format_pythia_string(parent_id, children_ids, branching_ratio, matrix_element=0):
    '''
    Format the decay parameters into a string readily usable to add the decay
    channel to the PYTHIA event generator.
    '''
    return '{}:addChannel = 1 {} {} {}'.format(
        parent_id, branching_ratio, matrix_element,
        ' '.join(str(child_id) for child_id in children_ids))


class Channel(with_metaclass(abc.ABCMeta, object)):
    '''
    Wraps a decay channel containing the scalar particle either in the initial
    or final state.
    '''
    def __init__(self, parent, children):
        self._parent = parent
        self._children = children
        self._str = _to_channel_str(parent, children)

    def __str__(self):
        return _to_channel_str(self._parent, self._children)

    @abc.abstractmethod
    def is_open(self, mS):
        '''
        Whether the decay channel is kinematically open.

        The default implementation assumes all particles to be on-shell, and
        should be overridden if this is not the case (e.g. S -> q qbar).
        '''
        pass # pragma: no cover

    def is_valid(self, mS):
        '''
        Whether the decay width calculation is valid for the considered mass.

        This is equivalent to checking whether the result is finite and not NaN.
        '''
        return np.isfinite(self.normalized_width(mS))

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


class ProductionChannel(Channel):
    '''
    Wraps a decay channel containing the scalar particle in the final state.
    '''
    def __init__(self, parent, other_children):
        super(ProductionChannel, self).__init__(parent, ['S'] + other_children)
        self._other_children = other_children
        self._parent_width = 1 / get_lifetime(self._parent)

    def is_open(self, mS):
        threshold = ( get_mass(self._parent)
                      - sum(get_mass(child) for child in self._other_children) )
        return mS < threshold

    def pythia_string(self, branching_ratio, scalar_id):
        return format_pythia_string(
            get_pdg_id(self._parent),
            [scalar_id] + [get_pdg_id(child) for child in self._other_children],
            branching_ratio)

    def normalized_branching_ratio(self, mS):
        return self.normalized_width(mS) / self._parent_width

    def branching_ratio(self, mS, theta):
        return self.width(mS, theta) / self._parent_width

    @property
    def parent_width(self):
        return self._parent_width


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
        return format_pythia_string(
            scalar_id, [get_pdg_id(child) for child in self._children],
            branching_ratio)
