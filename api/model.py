# -*- coding: utf-8 -*-

from __future__ import absolute_import, division

import numpy as np

from ..api.active_processes import ActiveProcesses
from ..api.branching_ratios import *
from ..data.constants import default_scalar_id
from ..production.two_body_hadronic import TwoBodyHadronic
from ..production.two_body_quartic import TwoBodyQuartic
from ..decay.leptonic import Leptonic
from ..decay.two_pions import TwoPions
from ..decay.two_kaons import TwoKaons
from ..decay.multimeson import Multimeson
from ..decay.two_gluons import TwoGluons
from ..decay.two_quarks import TwoQuarks


# All supported production channels
_production_channels = [
    # Kaon decays
    TwoBodyHadronic('K+'  , 'pi+'),
    TwoBodyHadronic('K_L0', 'pi0', weak_eigenstate='K0'),
    TwoBodyHadronic('K_S0', 'pi0', weak_eigenstate='K0'),
    TwoBodyQuartic('K_L0', weak_eigenstate='K0'),
    TwoBodyQuartic('K_S0', weak_eigenstate='K0'),
    # B meson decays
    TwoBodyHadronic('B+', 'pi+'        ),
    TwoBodyHadronic('B0', 'pi0'        ),
    TwoBodyHadronic('B+', 'K+'         ),
    TwoBodyHadronic('B0', 'K0'         ), # Final-state mixing handled by PYTHIA
    TwoBodyHadronic('B+', 'K*+'        ),
    TwoBodyHadronic('B0', 'K*0'        ),
    TwoBodyHadronic('B+', 'K*(1410)+'  ),
    TwoBodyHadronic('B0', 'K*(1410)0'  ),
    TwoBodyHadronic('B+', 'K*(1680)+'  ),
    TwoBodyHadronic('B0', 'K*(1680)0'  ),
    TwoBodyHadronic('B+', 'K_1(1270)+' ),
    TwoBodyHadronic('B0', 'K_1(1270)0' ),
    TwoBodyHadronic('B+', 'K_1(1400)+' ),
    TwoBodyHadronic('B0', 'K_1(1400)0' ),
    TwoBodyHadronic('B+', 'K*_0(700)+' ),
    TwoBodyHadronic('B0', 'K*_0(700)0' ),
    TwoBodyHadronic('B+', 'K*_0(1430)+'),
    TwoBodyHadronic('B0', 'K*_0(1430)0'),
    TwoBodyHadronic('B+', 'K*_2(1430)+'),
    TwoBodyHadronic('B0', 'K*_2(1430)0'),
    TwoBodyQuartic('B0'  ),
    TwoBodyQuartic('B_s0'),
]

_production_groups = {
    # Kaon decays
    'K -> S pi': [
        'K+ -> S pi+'  ,
        'K_L0 -> S pi0',
        'K_S0 -> S pi0',
    ],
    # 2-body quartic production from kaons
    'K -> S S': [
        'K_L0 -> S S',
        'K_S0 -> S S',
    ],
    # Pions
    'B -> S pi': [
        'B+ -> S pi+',
        'B0 -> S pi0',
    ],
    # Pseudoscalar mesons
    'B -> S K': [
        'B+ -> S K+',
        'B0 -> S K0',
    ],
    # Vector mesons
    'B -> S K*': [
        'B+ -> S K*+'      ,
        'B0 -> S K*0'      ,
        'B+ -> S K*(1410)+',
        'B0 -> S K*(1410)0',
        'B+ -> S K*(1680)+',
        'B0 -> S K*(1680)0',
    ],
    # Axial-vector mesons
    'B -> S K_1': [
        'B+ -> S K_1(1270)+',
        'B0 -> S K_1(1270)0',
        'B+ -> S K_1(1400)+',
        'B0 -> S K_1(1400)0',
    ],
    # Scalar mesons
    'B -> S K*_0': [
        'B+ -> S K*_0(700)+' ,
        'B0 -> S K*_0(700)0' ,
        'B+ -> S K*_0(1430)+',
        'B0 -> S K*_0(1430)0',
    ],
    # Tensor mesons
    'B -> S K*_2': [
        'B+ -> S K*_2(1430)+',
        'B0 -> S K*_2(1430)0',
    ],
    # All decays to kaons
    'B -> S K?': [
        'B -> S K'   ,
        'B -> S K*'  ,
        'B -> S K_1' ,
        'B -> S K*_0',
        'B -> S K*_2',
    ],
    # 2-body quartic production from B/B_s mesons
    'B -> S S': [
        'B0 -> S S'  ,
        'B_s0 -> S S',
    ]
}

# All supported decay channels
_decay_channels = [
    Leptonic('e'),       # S -> e+ e-
    Leptonic('mu'),      # S -> mu+ mu-
    Leptonic('tau'),     # S -> tau+ tau-
    TwoPions('neutral'), # S -> pi0 pi0
    TwoPions('charged'), # S -> pi+ pi-
    TwoKaons('neutral'), # S -> K0 Kbar0
    TwoKaons('charged'), # S -> K+ K-
    Multimeson(),        # S -> 4π, ηη, ρρ, …
    TwoGluons(),         # S -> g g
    TwoQuarks('s'),      # S -> s sbar
    TwoQuarks('c'),      # S -> c cbar
]

_decay_groups = {
    # Leptonic decays
    'S -> l+ l-' : ['S -> e+ e-'  , 'S -> mu+ mu-' , 'S -> tau+ tau-'],
    # Decays to pions
    'S -> pi pi' : ['S -> pi0 pi0', 'S -> pi+ pi-'],
    # Decays to kaons
    'S -> K K'   : ['S -> K0 Kbar0', 'S -> K+ K-' ],
    # Decays to heavy quarks
    'S -> q qbar': ['S -> s sbar' , 'S -> c cbar' ],
    # All hard QCD processes, which subsequently shower and hadronize
    'HardQCD'    : ['S -> g g'    , 'S -> q qbar' ],
    # All valid and relevant processes below 2 GeV
    'LightScalar': ['S -> e+ e-'  , 'S -> mu+ mu-' , 'S -> pi pi', 'S -> K K',
                    'S -> mesons...'],
    # All valid and relevant processes above 2 GeV
    'HeavyScalar': ['S -> l+ l-'  , 'HardQCD'     ],
}


class Model(object):
    '''
    Phenomenological model of a GeV-scale Higgs-like scalar particle.

    This class is the main entry point of the `scalar_portal` module. It wraps
    all the necessary logic to compute partial production and decay widths and
    branching ratios, as well as the total width / lifetime for the scalar
    particle. It can also be used to set up the PYTHIA event generator.

    Main reference:
        Boiarska, I., Bondarenko, K., Boyarsky, A., Gorkavenko, V.,
        Ovchynnikov, M., Sokolenko, A., 2019.
        Phenomenology of GeV-scale scalar portal.
        arXiv:1904.10447 [hep-ex, physics:hep-ph].
    '''
    def __init__(self, scalar_id=default_scalar_id):
        self._production = ActiveProcesses(_production_channels, _production_groups)
        self._decays = ActiveProcesses(_decay_channels, _decay_groups)
        self._scalar_id = scalar_id

    @property
    def production(self):
        'Production channels for the scalar.'
        return self._production

    @property
    def decays(self):
        'Decay channels for the scalar.'
        return self._decays

    @property
    def scalar_pdg_id(self):
        'The PDG ID for the scalar particle.'
        return self._scalar_id

    def compute_branching_ratios(self, mass, coupling, ignore_invalid=False):
        '''
        Compute the production and decay branching ratios of the scalar
        particle, and return a `BranchingRatiosResult` object containing the
        result.
        '''
        prod_channels  = self._production.get_active_processes()
        decay_channels = self._decays.get_active_processes()
        prod_br  = ProductionBranchingRatios(
            prod_channels , mass, coupling, ignore_invalid, scalar_id=self._scalar_id)
        decay_br = DecayBranchingRatios(
            decay_channels, mass, coupling, ignore_invalid, scalar_id=self._scalar_id)
        res = BranchingRatiosResult(prod_br, decay_br)
        return res
