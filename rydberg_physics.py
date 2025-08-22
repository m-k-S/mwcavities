

import numpy as np
from constants import PI, QE, A0, HBAR
from data_structures import RydbergParams


def try_arc_dipole_and_lifetimes(rp: RydbergParams):
    
    if not rp.use_arc or rp.state1 is None or rp.state2 is None:
        return None, None
    try:
        from arc import Rubidium85, Rubidium87
        atom = Rubidium85() if rp.isotope == "Rb85" else Rubidium87()
        n1, l1, j1, m1 = rp.state1
        n2, l2, j2, m2 = rp.state2
        
        try:
            d_SI = atom.getDipoleMatrixElement(n1, l1, j1, m1, n2, l2, j2, m2, s=0, q=0, SI=True)
            dipole = float(d_SI)
        except Exception:
            d_au = atom.getDipoleMatrixElement(n1, l1, j1, m1, n2, l2, j2, m2, s=0, q=0)
            dipole = float(d_au) * QE * A0
        tau1 = atom.getStateLifetime(n1, l1, j1, rp.temperature_K)
        tau2 = atom.getStateLifetime(n2, l2, j2, rp.temperature_K)
        Gamma_pop = 0.5 * (1.0/tau1 + 1.0/tau2) * 2 * PI
        return dipole, Gamma_pop
    except Exception:
        return None, None


def hydro_dipole_Cm(n1_eff, n2_eff, mu_prefactor=1.5):
    
    nbar2 = 0.5 * (n1_eff**2 + n2_eff**2)
    return mu_prefactor * QE * A0 * nbar2


def hydro_gamma_population(n1_eff, n2_eff, tau0_0K=1e-9, bbr_factor=0.6):
    
    tau1 = tau0_0K * n1_eff**3 * bbr_factor
    tau2 = tau0_0K * n2_eff**3 * bbr_factor
    return 0.5 * (1.0/tau1 + 1.0/tau2) * 2 * PI


def single_atom_g(Ezpf, dipole_Cm, cg_factor=1.0):
    
    return (dipole_Cm * Ezpf * cg_factor) / HBAR


def estimate_collective_G(positions, u_func, g0, pump_fraction):
    
    u = u_func(positions)
    gi = g0 * u
    G2 = np.sum((gi**2) * pump_fraction)
    return np.sqrt(G2), gi, u
