# -*- coding: utf-8 -*-

from __future__ import absolute_import
from future.utils import viewitems

from copy import copy
# We use OrderedDict to obtain stable, deterministic results.
from collections import OrderedDict


class ActiveProcesses(object):
    '''
    Represents a set of available and active processes.
    '''
    def __init__(self, process_list, process_groups=dict(), default_selection=[]):
        # Shallow copy because processes are immutable
        self._process_list = copy(process_list)
        self._process_dict = OrderedDict((str(process), process) for process in process_list)
        # Similarly, process groups are immutable `frozenset`'s
        self._process_groups = process_groups
        all_process_groups = OrderedDict((str(process), None) for process in process_list)
        all_process_groups.update(process_groups)
        all_process_groups.update({'All': self._process_dict.keys()})
        self._all_process_groups = all_process_groups
        # Active processes
        self._active = set()
        for proc in default_selection:
            self.enable(proc)

    def _get_subprocesses(self, process):
        try:
            return self._all_process_groups[process]
        except KeyError:
            raise(ValueError("Process '{}' not available.".format(process)))

    def enable(self, process):
        'Enable one process or group.'
        subprocesses = self._get_subprocesses(process)
        if subprocesses is None: # Elementary process
            self._active.add(process)
        else: # Enable all subprocesses
            for subproc in subprocesses:
                self.enable(subproc)

    def disable(self, process):
        '''
        Disable one process or group.

        If the process or group is already disabled, do nothing.
        '''
        subprocesses = self._get_subprocesses(process)
        if subprocesses is None: # Elementary process
            self._active.discard(process)
        else: # Disable all subprocesses
            for subproc in subprocesses:
                self.disable(subproc)

    def enable_all(self):
        'Enable all processes.'
        self._active = set(self._process_dict.keys())

    def disable_all(self):
        'Disable all processes.'
        self._active = set()

    def list_enabled(self):
        'List all enabled processes.'
        return list(self._active)

    def list_available(self):
        'List all available processes.'
        return list(self._process_dict.keys())

    def list_available_groups(self):
        'List all available groups of processes.'
        return list(self._process_groups)

    def get_active_processes(self):
        'Return the `Channel` objects for all active processes.'
        lst = [self._process_dict[ch] for ch in self._active]
        lst.sort()
        return lst
