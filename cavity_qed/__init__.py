"""Cavity QED simulation package.

This package provides tools for simulating cavity quantum electrodynamics (QED) 
systems, with a focus on Rydberg atoms in microwave cavities.
"""

__version__ = "0.1.0"

from .models import (
    CavityParameters,
    RydbergParameters,
    AtomicCloud,
    CouplingRegime,
    SimulationResult
)

from .core.physics import CavityQEDCalculator
from .config import ConfigManager, get_default_config
from .visualization.plotting import (
    plot_spectrum,
    plot_mode_profile,
    plot_coupling_analysis
)

__all__ = [
    'CavityParameters',
    'RydbergParameters',
    'AtomicCloud',
    'CouplingRegime',
    'SimulationResult',
    'CavityQEDCalculator',
    'ConfigManager',
    'get_default_config',
    'plot_spectrum',
    'plot_mode_profile',
    'plot_coupling_analysis'
]
