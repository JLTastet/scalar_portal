# -*- coding: utf-8 -*-

from __future__ import absolute_import, division
from future.utils import viewitems

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Run in headless mode
import matplotlib.pyplot as plt

from ..api.model import Model

_srcdir = os.path.dirname(__file__)
_outdir = os.path.join(_srcdir, '..', 'plots')
if not os.path.isdir(_outdir):
    os.mkdir(_outdir)

def _plot_production(process='All', pred=lambda ch: True, mS_max=5.25, res=800, prefix=''):
    m = Model()
    if process == 'All':
        m.production.enable_all()
    else:
        m.production.enable(process)
    mS = np.linspace(0, mS_max, res)
    result = m.compute_branching_ratios(mS, theta=1, alpha=0)
    widths = result.production.width
    brs = result.production.branching_ratios
    # Plot decay widths of heavy hadrons to the Scalar
    figw, axw = plt.subplots(figsize=(15,8))
    for ch, w in viewitems(widths):
        if not pred(ch):
            continue
        axw.plot(mS, w, label=ch)
    axw.set_xlabel(r'$m_S\;[\mathrm{GeV}]$')
    axw.set_ylabel(r'$\Gamma / \theta^2\;[\mathrm{GeV}]$')
    axw.set_yscale('log')
    axw.grid(color='grey', linestyle=':')
    box = axw.get_position()
    axw.autoscale(tight=True)
    axw.set_position([box.x0, box.y0, 0.75*box.width, box.height])
    axw.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figw.savefig(os.path.join(_outdir, '{}production_widths.pdf'.format(prefix)))
    plt.close()
    # Plot branching ratios of heavy hadrons to the Scalar
    figbr, axbr = plt.subplots(figsize=(15,8))
    for ch, br in viewitems(brs):
        if not pred(ch):
            continue
        axbr.plot(mS, br, label=ch)
    axbr.set_xlabel(r'$m_S\;[\mathrm{GeV}]$')
    axbr.set_ylabel(r'$\mathrm{Br} / \theta^2$')
    axbr.set_yscale('log')
    axbr.grid(color='grey', linestyle=':')
    box = axbr.get_position()
    axbr.autoscale(tight=True)
    axbr.set_position([box.x0, box.y0, 0.75*box.width, box.height])
    axbr.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figbr.savefig(os.path.join(_outdir, '{}production_br.pdf'.format(prefix)))
    plt.close()

def test_charged_production_plot():
    _plot_production('All', lambda ch: ch[-1] == '+', prefix='charged_')

def test_neutral_production_plot():
    _plot_production('All', lambda ch: ch[-1] == '0', prefix='neutral_')

def test_decay_plot(res=800):
    threshold = 2.0 # GeV
    m_low = Model()
    m_low.decays.enable('LightScalar')
    m_high = Model()
    m_high.decays.enable('HeavyScalar')
    mS_low = np.linspace(0, threshold, res)
    mS_high = np.linspace(threshold, 5, res)
    res_low = m_low.compute_branching_ratios(mS_low, theta=1)
    res_high = m_high.compute_branching_ratios(mS_high, theta=1)
    widths_low = res_low.decays.width
    widths_high = res_high.decays.width
    brs_low = res_low.decays.branching_ratios
    brs_high = res_high.decays.branching_ratios
    # Plot decay widths of the Scalar
    figw, axw = plt.subplots(figsize=(15,8))
    axw.plot(mS_low, res_low.total_width, label='Total', color='k', linewidth=2)
    axw.plot(mS_high, res_high.total_width, color='k', linewidth=2)
    for ch, w in viewitems(widths_low):
        axw.plot(mS_low, w, label=ch)
    for ch, w in viewitems(widths_high):
        axw.plot(mS_high, w, label=ch)
    axw.set_xlabel(r'$m_S\;[\mathrm{GeV}]$')
    axw.set_ylabel(r'$\Gamma\;[\mathrm{GeV}]$')
    axw.set_yscale('log')
    axw.grid(color='grey', linestyle=':')
    axw.autoscale(tight=True)
    box = axw.get_position()
    axw.set_position([box.x0, box.y0, 0.75*box.width, box.height])
    axw.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figw.savefig(os.path.join(_outdir, 'decay_widths.pdf'))
    plt.close()
    # Plot branching ratios of the Scalar
    figbr, axbr = plt.subplots(figsize=(15,8))
    for ch, br in viewitems(brs_low):
        axbr.plot(mS_low, br, label=ch)
    for ch, br in viewitems(brs_high):
        axbr.plot(mS_high, br, label=ch)
    axbr.set_xlabel('BR')
    axbr.set_ylabel(r'$\Gamma\;[\mathrm{GeV}]$')
    axbr.set_yscale('log')
    axbr.grid(color='grey', linestyle=':')
    axbr.autoscale(tight=True)
    box = axbr.get_position()
    axbr.set_position([box.x0, box.y0, 0.75*box.width, box.height])
    axbr.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    with np.errstate(invalid='ignore'):
        figbr.savefig(os.path.join(_outdir, 'decay_br.pdf'))
    plt.close()
