"""Configuration management for cavity QED simulations."""

from typing import Dict, Any, Optional
import json
from pathlib import Path
from dataclasses import asdict

from ..models import CavityParameters, RydbergParameters, AtomicCloud

class ConfigManager:
    """Manages simulation configuration and parameters."""
    
    @staticmethod
    def load_config(filepath: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_config(config: Dict[str, Any], filepath: str) -> None:
        """Save configuration to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary."""
        required_sections = ['cavity', 'rydberg', 'atoms']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        return True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create parameter objects from configuration dictionary."""
        cls.validate_config(config_dict)
        
        # Create parameter objects
        cavity = CavityParameters(**config_dict['cavity'])
        rydberg = RydbergParameters(**config_dict['rydberg'])
        
        # Handle atomic cloud
        atoms_config = config_dict['atoms'].copy()
        atoms_config['total_atoms'] = atoms_config.pop('N_total', 1e6)
        atoms = AtomicCloud(**atoms_config)
        
        return {
            'cavity': cavity,
            'rydberg': rydberg,
            'atoms': atoms,
            'simulation': config_dict.get('simulation', {}),
            'cw': config_dict.get('cw', {})
        }
    
    @classmethod
    def to_dict(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameter objects back to dictionary."""
        return {
            'cavity': asdict(params['cavity']),
            'rydberg': asdict(params['rydberg']),
            'atoms': {
                'N_total': params['atoms'].total_atoms,
                'sigma_xyz': params['atoms'].sigma_xyz,
                'center': params['atoms'].center
            },
            'simulation': params.get('simulation', {}),
            'cw': params.get('cw', {})
        }

# Default configuration
def get_default_config() -> Dict[str, Any]:
    """Get default configuration parameters."""
    return {
        "cavity": {
            "frequency": 13.6713e9,  # Hz
            "quality_factor": 5.0e4,
            "geometry": "rect",
            "dimensions": (0.016447, 0.008223, 0.029420),  # m
            "mode_indices": (2, 0, 1),
            "polarization_efficiency": 1.0,
            "grid_points": 36,
            "kappa_port1_fraction": 0.25,
            "kappa_port2_fraction": 0.25
        },
        "rydberg": {
            "n1_eff": 50.0,
            "n2_eff": 51.0,
            "dipole_prefactor": 1.5,
            "lifetime_0K": 1.0e-9,
            "bbr_factor_300K": 0.6,
            "dephasing_rate": 0.2e6,  # Hz
            "cg_factor": 0.7,
            "use_arc": False,
            "isotope": "Rb85",
            "temperature": 300.0  # K
        },
        "atoms": {
            "N_total": 1e6,
            "sigma_xyz": (1e-3, 1e-3, 1e-3),  # m
            "center": (0.0, 0.0, 0.0)  # m
        },
        "simulation": {
            "seed": 1,
            "make_plots": True
        },
        "cw": {
            "detuning_range": (-20e6, 20e6),  # Hz
            "num_points": 1001
        }
    }
