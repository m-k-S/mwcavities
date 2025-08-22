

import numpy as np
from scipy.integrate import solve_ivp
from constants import PI, HBAR, C0, EPS0, QE, A0
from data_structures import AtomicCloud, GaussianBeam


def ladder_rhs(t, rho_vec, Omega_ge, Omega_er, Delta_ge, Delta_er,
               Gamma_e, Gamma_r, gamma_ge_phi, gamma_er_phi):
    
    rho = rho_vec.reshape((3, 3))
    
    
    H = np.array([[0.0,               0.5*HBAR*Omega_ge, 0.0],
                  [0.5*HBAR*Omega_ge, -HBAR*Delta_ge,    0.5*HBAR*Omega_er],
                  [0.0,               0.5*HBAR*Omega_er, -HBAR*(Delta_ge+Delta_er)]], dtype=np.complex128)
    
    
    comm = (-1j/HBAR) * (H @ rho - rho @ H)
    
    
    L = np.zeros((3, 3), dtype=np.complex128)
    
    
    Ceg = np.zeros((3, 3))
    Ceg[0, 1] = 1.0
    L += Gamma_e * (Ceg @ rho @ Ceg.conj().T - 0.5 * (Ceg.conj().T @ Ceg @ rho + rho @ Ceg.conj().T @ Ceg))
    
    
    Cre = np.zeros((3, 3))
    Cre[1, 2] = 1.0
    L += Gamma_r * (Cre @ rho @ Cre.conj().T - 0.5 * (Cre.conj().T @ Cre @ rho + rho @ Cre.conj().T @ Cre))
    
    
    D = np.zeros((3, 3), dtype=np.complex128)
    D[0, 1] += -gamma_ge_phi * rho[0, 1]
    D[1, 0] += -gamma_ge_phi * rho[1, 0]
    D[1, 2] += -gamma_er_phi * rho[1, 2]
    D[2, 1] += -gamma_er_phi * rho[2, 1]
    D[0, 2] += -(gamma_ge_phi + gamma_er_phi) * rho[0, 2]
    D[2, 0] += -(gamma_ge_phi + gamma_er_phi) * rho[2, 0]
    
    return (comm + L + D).reshape(-1).view(np.float64)


def estimate_pumped_atoms_fast(cloud: AtomicCloud, N_mc: int,
                               beam780: GaussianBeam, beam481: GaussianBeam,
                               Delta_ge, Delta_er,
                               Gamma_e=2*PI*6.07e6,
                               Gamma_r=2*PI*1.0e4,
                               gamma_ge_phi=2*PI*0.5e6,
                               gamma_er_phi=2*PI*0.2e6,
                               mu_er=None,
                               grid_points=5):
    
    
    Isat_780 = 167.0
    def I_to_Omega_780(I): 
        return Gamma_e * np.sqrt(np.maximum(I, 0.0) / (2 * Isat_780))
    
    
    def I_to_Omega_481(I, dipole_Cm=None, prefactor_MHz_per_sqrtWcm2=50.0):
        if dipole_Cm is not None:
            E = np.sqrt(2 * np.maximum(I, 0.0) / (C0 * EPS0))
            return (dipole_Cm * E) / HBAR
        I_Wcm2 = I / 1e4
        return 2 * PI * (prefactor_MHz_per_sqrtWcm2 * 1e6) * np.sqrt(np.maximum(I_Wcm2, 0.0))

    pos = cloud.sample_positions(N_mc)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    I780 = beam780.intensity(x, y)
    I481 = beam481.intensity(x, y)
    Om_ge = I_to_Omega_780(I780)
    Om_er = I_to_Omega_481(I481, dipole_Cm=mu_er)

    ge_grid = np.linspace(float(Om_ge.min()), float(Om_ge.max()), max(2, grid_points))
    er_grid = np.linspace(float(Om_er.min()), float(Om_er.max()), max(2, grid_points))

    
    def solve_ss(oge, oer):
        rho0 = np.zeros((3, 3), dtype=np.complex128)
        rho0[0, 0] = 1.0
        y0 = rho0.reshape(-1).view(np.float64)
        rhs = lambda t, y: ladder_rhs(t, y.view(np.complex128), oge, oer, Delta_ge, Delta_er,
                                      Gamma_e, Gamma_r, gamma_ge_phi, gamma_er_phi)
        sol = solve_ivp(rhs, [0, 3.0e-5], y0, method='RK45', rtol=6e-6, atol=1e-8)
        rho_ss = sol.y[:, -1].view(np.complex128).reshape((3, 3))
        return float(np.real(rho_ss[2, 2]))

    Pr_grid = np.zeros((len(ge_grid), len(er_grid)))
    for i, oge in enumerate(ge_grid):
        for j, oer in enumerate(er_grid):
            Pr_grid[i, j] = solve_ss(oge, oer)

    ge_idx = np.abs(Om_ge[:, None] - ge_grid[None, :]).argmin(axis=1)
    er_idx = np.abs(Om_er[:, None] - er_grid[None, :]).argmin(axis=1)
    pump_frac = Pr_grid[ge_idx, er_idx]
    N_pumped = pump_frac.mean() * cloud.N_total
    return pos, pump_frac, N_pumped
