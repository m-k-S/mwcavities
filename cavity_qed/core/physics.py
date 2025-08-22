"""Core physics calculations for cavity QED simulations."""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import asdict

from ..models import (
    CavityParameters,
    RydbergParameters,
    AtomicCloud,
    CouplingRegime,
    SimulationResult
)

# Physical constants from scipy if available, otherwise use these values
try:
    from scipy.constants import hbar, epsilon_0 as eps0, pi, c
    HBAR = hbar
    EPS0 = eps0
    PI = pi
    C_LIGHT = c
    QE = 1.602176634e-19  # Elementary charge in C
    A0 = 5.29177210903e-11  # Bohr radius in m
except ImportError:
    HBAR = 1.054571817e-34
    EPS0 = 8.8541878128e-12
    PI = np.pi
    C_LIGHT = 299792458.0
    QE = 1.602176634e-19
    A0 = 5.29177210903e-11

class CavityQEDCalculator:
    """Main class for performing cavity QED calculations."""
    
    def __init__(self, cavity: CavityParameters, rydberg: RydbergParameters):
        self.cavity = cavity
        self.rydberg = rydberg
        self._mode_function = None
        self._effective_volume = None
        
    def calculate_mode_volume(self) -> float:
        """Calculate the effective mode volume of the cavity."""
        if self._effective_volume is not None:
            return self._effective_volume
            
        # This is a placeholder - implement actual mode volume calculation
        a, b, d = self.cavity.dimensions
        m, n, p = self.cavity.mode_indices
        self._effective_volume = (a * b * d) / 8  # Simple approximation
        return self._effective_volume
    
    def calculate_electric_field(self, x: float, y: float, z: float) -> float:
        """Calculate the normalized electric field at position (x,y,z)."""
        a, b, d = self.cavity.dimensions
        m, n, p = self.cavity.mode_indices
        
        # Simple TE mode pattern - replace with actual mode solver
        kx = m * PI / a
        ky = n * PI / b
        kz = p * PI / d
        
        return np.sin(kx * x) * np.cos(ky * y) * np.sin(kz * z)
    
    def calculate_coupling(self, dipole_moment: float) -> float:
        """Calculate the vacuum Rabi frequency for a given dipole moment."""
        V_eff = self.calculate_mode_volume()
        omega_c = 2 * PI * self.cavity.frequency
        E_zpf = np.sqrt(HBAR * omega_c / (2 * EPS0 * V_eff)) * self.cavity.polarization_efficiency
        return dipole_moment * E_zpf / HBAR
    
    def analyze_coupling_regime(self, g: float) -> Dict[str, Any]:
        """Analyze the coupling regime based on the coupling strength."""
        kappa = 2 * PI * self.cavity.frequency / self.cavity.quality_factor
        gamma = self.rydberg.dephasing_rate
        
        cooperativity = 4 * g**2 / (kappa * gamma)
        strong_coupling = g > (kappa + gamma) / 4
        
        regime = (
            CouplingRegime.STRONG if strong_coupling 
            else CouplingRegime.INTERMEDIATE if cooperativity > 1 
            else CouplingRegime.WEAK
        )
        
        return {
            'cooperativity': cooperativity,
            'regime': regime,
            'kappa': kappa,
            'gamma': gamma,
            'g': g
        }

    def simulate_cw_response(
        self, 
        detuning_range: Tuple[float, float], 
        num_points: int = 1001
    ) -> SimulationResult:
        """Simulate the cavity transmission spectrum."""
        f_min, f_max = detuning_range
        f_off = np.linspace(f_min, f_max, num_points)
        
        kappa = 2 * PI * self.cavity.frequency / self.cavity.quality_factor
        kappa1 = self.cavity.kappa_port1_fraction * kappa
        kappa2 = self.cavity.kappa_port2_fraction * kappa
        
        # Simple Lorentzian response - replace with actual calculation
        D = (kappa/2)**2 + (2 * PI * f_off)**2
        S21 = np.sqrt(kappa1 * kappa2) / (kappa/2 - 1j * 2 * PI * f_off)
        P21 = np.abs(S21)**2
        
        return SimulationResult(
            parameters={
                'cavity': asdict(self.cavity),
                'rydberg': asdict(self.rydberg),
                'detuning_range': detuning_range,
                'num_points': num_points
            },
            data={
                'frequencies': f_off,
                'transmission': P21,
                'S21': S21
            },
            metadata={
                'analysis': self.analyze_coupling_regime(1e6)  # Example coupling
            }
        )
