"""Core data models for cavity QED simulations."""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from enum import Enum

class CouplingRegime(str, Enum):
    STRONG = "strong"
    WEAK = "weak"
    INTERMEDIATE = "intermediate"

@dataclass
class CavityParameters:
    """Parameters describing a microwave cavity."""
    frequency: float  # Hz
    quality_factor: float
    geometry: str
    dimensions: Tuple[float, ...]  # (width, height, length) in meters
    mode_indices: Tuple[int, int, int] = (2, 0, 1)
    polarization_efficiency: float = 1.0
    grid_points: int = 40
    kappa_port1_fraction: float = 0.25
    kappa_port2_fraction: float = 0.25

@dataclass
class RydbergParameters:
    """Parameters for Rydberg state calculations."""
    n1_eff: float
    n2_eff: float
    dipole_prefactor: float = 1.5
    lifetime_0K: float = 1.0e-9
    bbr_factor_300K: float = 0.6
    dephasing_rate: float = 0.2e6  # Hz
    cg_factor: float = 0.7
    use_arc: bool = False
    isotope: str = "Rb85"
    state1: Optional[Tuple[int, int, float, float]] = None
    state2: Optional[Tuple[int, int, float, float]] = None
    temperature: float = 300.0  # K

@dataclass
class AtomicCloud:
    """Parameters for the atomic cloud distribution."""
    total_atoms: int
    sigma_xyz: Tuple[float, float, float]  # meters
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # meters

@dataclass
class SimulationResult:
    """Container for simulation results and analysis."""
    parameters: Dict[str, Any]
    data: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "parameters": self.parameters,
            "data": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in self.data.items()},
            "metadata": self.metadata
        }
