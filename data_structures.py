

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CavityParams:
    
    f_c: float                 
    Q_loaded: float            
    geometry: str              
    dims: Tuple[float, ...]    
    eta_pol: float = 1.0       
    grid_N: int = 40           
    rect_mode_indices: Optional[Tuple[int,int,int]] = None
    kappa_port1_frac: float = 0.25  
    kappa_port2_frac: float = 0.25  


@dataclass
class RydbergParams:
    
    
    n1_eff: float
    n2_eff: float
    
    mu_prefactor: float = 1.5
    tau0_0K: float = 1.0e-9
    bbr_factor_300K: float = 0.6
    gamma_phi_Hz: float = 0.2e6
    cg_factor: float = 0.7
    
    use_arc: bool = False
    isotope: str = "Rb85"      
    
    state1: Optional[Tuple[int,int,float,float]] = None
    state2: Optional[Tuple[int,int,float,float]] = None
    temperature_K: float = 300.0


@dataclass
class AtomicCloud:
    
    N_total: int
    sigma_xyz: Tuple[float, float, float]  
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    def sample_positions(self, N: int, rng=np.random) -> np.ndarray:
        
        sx, sy, sz = self.sigma_xyz
        cx, cy, cz = self.center
        xs = rng.normal(cx, sx, size=N)
        ys = rng.normal(cy, sy, size=N)
        zs = rng.normal(cz, sz, size=N)
        return np.stack([xs, ys, zs], axis=-1)


@dataclass
class GaussianBeam:
    
    I0: float  
    w:  float  
    
    def intensity(self, x, y):
        
        r2 = x**2 + y**2
        return self.I0 * np.exp(-2*r2/self.w**2)
