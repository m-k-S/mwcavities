



import json, os
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Optional

import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp
from scipy.special import j0
import matplotlib.pyplot as plt


EPS0 = 8.8541878128e-12
HBAR = 1.054571817e-34
QE   = 1.602176634e-19
A0   = 5.29177210903e-11
C0   = 299792458.0
CHI_01 = 2.404825557695773  


@dataclass
class CavityParams:
    f_c: float                 
    Q_loaded: float
    geometry: str              
    dims: Tuple[float, ...]    
    rect_mode_indices: Tuple[int,int,int] = (1,0,1)  
    eta_pol: float = 1.0
    grid_N: int = 36
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
    sigma_xyz_m: Tuple[float, float, float]
    center_m: Tuple[float, float, float] = (0.0,0.0,0.0)
    def sample_positions(self, N: int, rng=np.random) -> np.ndarray:
        sx, sy, sz = self.sigma_xyz_m; cx, cy, cz = self.center_m
        xs = rng.normal(cx, sx, size=N); ys = rng.normal(cy, sy, size=N); zs = rng.normal(cz, sz, size=N)
        return np.stack([xs, ys, zs], axis=-1)

@dataclass
class GaussianBeam:
    I0: float  
    w:  float  
    def intensity(self, x, y):
        r2 = x**2 + y**2
        return self.I0 * np.exp(-2*r2/self.w**2)


def rect_mode_u_TEmnp(dims, m, n, p):
    Lx, Ly, Lz = dims
    def sin_or_one(frac, order):
        return np.ones_like(frac) if order == 0 else np.sin(order*np.pi*frac)
    def u(pts: np.ndarray):
        x = (pts[:,0]+Lx/2)/Lx; y = (pts[:,1]+Ly/2)/Ly; z = (pts[:,2]+Lz/2)/Lz
        return np.abs(sin_or_one(x,m)*sin_or_one(y,n)*sin_or_one(z,p))
    vol = Lx*Ly*Lz; bounds = (-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2)
    return u, vol, bounds

def cyl_mode_u_TM010(dims):
    R, L = dims
    def u(pts: np.ndarray):
        r = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
        val = j0(CHI_01*np.clip(r,None,R)/R); val[r>R] = 0.0
        return np.abs(val)
    vol = pi*R*R*L; bounds = (-R,R,-R,R,-L/2,L/2)
    return u, vol, bounds

def make_mode_u(cav: CavityParams):
    if cav.geometry == "rect":
        m,n,p = cav.rect_mode_indices
        return rect_mode_u_TEmnp(cav.dims, m,n,p)
    elif cav.geometry == "cyl":
        return cyl_mode_u_TM010(cav.dims)
    raise ValueError("geometry must be 'rect' or 'cyl'")

def numerical_Veff_from_u(u_func, volume, bounds, N):
    x0,x1,y0,y1,z0,z1 = bounds
    xs = np.linspace(x0,x1,N); ys = np.linspace(y0,y1,N); zs = np.linspace(z0,z1,N)
    XX,YY,ZZ = np.meshgrid(xs,ys,zs, indexing='xy')
    pts = np.column_stack([XX.ravel(),YY.ravel(),ZZ.ravel()])
    uu = u_func(pts); return volume*float(np.mean(uu**2))  


def try_arc_dipole_and_lifetimes(rp: RydbergParams):
    if not rp.use_arc or rp.state1 is None or rp.state2 is None:
        return None, None
    try:
        from arc import Rubidium85, Rubidium87
        atom = Rubidium85() if rp.isotope=="Rb85" else Rubidium87()
        n1,l1,j1,m1 = rp.state1; n2,l2,j2,m2 = rp.state2
        try:
            d_SI = atom.getDipoleMatrixElement(n1,l1,j1,m1, n2,l2,j2,m2, s=0, q=0, SI=True); dip = float(d_SI)
        except Exception:
            d_au = atom.getDipoleMatrixElement(n1,l1,j1,m1, n2,l2,j2,m2, s=0, q=0); dip = float(d_au)*QE*A0
        tau1 = atom.getStateLifetime(n1,l1,j1, rp.temperature_K); tau2 = atom.getStateLifetime(n2,l2,j2, rp.temperature_K)
        Gamma_pop = 0.5*(1.0/tau1 + 1.0/tau2)*2*pi
        return dip, Gamma_pop
    except Exception:
        return None, None


def hydro_dipole_Cm(n1_eff, n2_eff, mu_prefactor=1.5):
    nbar2 = 0.5*(n1_eff**2 + n2_eff**2)
    return mu_prefactor * QE * A0 * nbar2

def hydro_gamma_population(n1_eff, n2_eff, tau0_0K=1e-9, bbr_factor=0.6):
    tau1 = tau0_0K*n1_eff**3*bbr_factor; tau2 = tau0_0K*n2_eff**3*bbr_factor
    return 0.5*(1.0/tau1 + 1.0/tau2)*2*pi


def ladder_rhs(t, rho_vec, Omega_ge, Omega_er, Delta_ge, Delta_er,
               Gamma_e, Gamma_r, gamma_ge_phi, gamma_er_phi):
    rho = rho_vec.reshape((3,3))
    H = np.array([[0.0,               0.5*HBAR*Omega_ge, 0.0],
                  [0.5*HBAR*Omega_ge, -HBAR*Delta_ge,    0.5*HBAR*Omega_er],
                  [0.0,               0.5*HBAR*Omega_er, -HBAR*(Delta_ge+Delta_er)]], dtype=np.complex128)
    comm = (-1j/HBAR)*(H@rho - rho@H)
    L = np.zeros((3,3), dtype=np.complex128)
    Ceg = np.zeros((3,3)); Ceg[0,1]=1.0
    L += Gamma_e*(Ceg@rho@Ceg.conj().T - 0.5*(Ceg.conj().T@Ceg@rho + rho@Ceg.conj().T@Ceg))
    Cre = np.zeros((3,3)); Cre[1,2]=1.0
    L += Gamma_r*(Cre@rho@Cre.conj().T - 0.5*(Cre.conj().T@Cre@rho + rho@Cre.conj().T@Cre))
    D = np.zeros((3,3), dtype=np.complex128)
    D[0,1] += -gamma_ge_phi*rho[0,1]; D[1,0] += -gamma_ge_phi*rho[1,0]
    D[1,2] += -gamma_er_phi*rho[1,2]; D[2,1] += -gamma_er_phi*rho[2,1]
    D[0,2] += -(gamma_ge_phi+gamma_er_phi)*rho[0,2]; D[2,0] += -(gamma_ge_phi+gamma_er_phi)*rho[2,0]
    return (comm+L+D).reshape(-1).view(np.float64)

def estimate_pumped_atoms_fast(cloud: AtomicCloud, N_mc: int,
                               beam780: GaussianBeam, beam481: GaussianBeam,
                               Delta_ge, Delta_er,
                               Gamma_e=2*pi*6.07e6, Gamma_r=2*pi*1.0e4,
                               gamma_ge_phi=2*pi*0.5e6, gamma_er_phi=2*pi*0.2e6,
                               mu_er=None, grid_points=5):
    Isat_780 = 167.0
    def I_to_Omega_780(I): return Gamma_e*np.sqrt(np.maximum(I,0.0)/(2*Isat_780))
    def I_to_Omega_481(I, dipole_Cm=None, prefactor_MHz_per_sqrtWcm2=50.0):
        if dipole_Cm is not None:
            E = np.sqrt(2*np.maximum(I,0.0)/(C0*EPS0)); return (dipole_Cm*E)/HBAR
        I_Wcm2 = I/1e4; return 2*pi*(prefactor_MHz_per_sqrtWcm2*1e6)*np.sqrt(np.maximum(I_Wcm2,0.0))

    pos = cloud.sample_positions(N_mc)
    x,y,z = pos[:,0], pos[:,1], pos[:,2]
    Om_ge = I_to_Omega_780(beam780.intensity(x,y))
    Om_er = I_to_Omega_481(beam481.intensity(x,y), dipole_Cm=mu_er)

    ge_grid = np.linspace(float(Om_ge.min()), float(Om_ge.max()), max(2,grid_points))
    er_grid = np.linspace(float(Om_er.min()), float(Om_er.max()), max(2,grid_points))

    def solve_ss(oge, oer):
        rho0 = np.zeros((3,3), dtype=np.complex128); rho0[0,0]=1.0
        y0 = rho0.reshape(-1).view(np.float64)
        rhs = lambda t, y: ladder_rhs(t, y.view(np.complex128), oge, oer, Delta_ge, Delta_er,
                                      Gamma_e, Gamma_r, gamma_ge_phi, gamma_er_phi)
        sol = solve_ivp(rhs, [0, 3.0e-5], y0, method='RK45', rtol=6e-6, atol=1e-8)
        rho_ss = sol.y[:,-1].view(np.complex128).reshape((3,3))
        return float(np.real(rho_ss[2,2]))

    Pr_grid = np.zeros((len(ge_grid), len(er_grid)))
    for i, oge in enumerate(ge_grid):
        for j, oer in enumerate(er_grid):
            Pr_grid[i,j] = solve_ss(oge, oer)

    ge_idx = np.abs(Om_ge[:,None] - ge_grid[None,:]).argmin(axis=1)
    er_idx = np.abs(Om_er[:,None] - er_grid[None,:]).argmin(axis=1)
    pump_frac = Pr_grid[ge_idx, er_idx]
    N_pumped = pump_frac.mean() * cloud.N_total
    return pos, pump_frac, N_pumped


def single_atom_g(Ezpf, dipole_Cm, cg_factor=1.0):
    return (dipole_Cm * Ezpf * cg_factor) / HBAR  

def collective_G(positions, u_func, g0, pump_fraction):
    u = u_func(positions); gi = g0 * u
    G2 = np.sum((gi**2) * pump_fraction)
    return np.sqrt(G2), gi, u

def cw_S21_spectrum(G, kappa, gamma, delta_ac, kappa1, kappa2, span_Hz=40e6, npts=1201):
    freqs = np.linspace(-span_Hz/2, +span_Hz/2, npts)   
    Dc = 2*pi*freqs                                   
    Da = Dc - delta_ac                                
    D = (kappa/2 + 1j*Dc) + (G**2)/(gamma/2 + 1j*Da)  
    S21 = np.sqrt(kappa1*kappa2) / D
    P21 = np.abs(S21)**2
    if P21.max() > 0: P21 /= P21.max()
    return freqs, P21


def _rydberg_dipole_and_gamma(rp: RydbergParams):
    d_arc, Gpop_arc = try_arc_dipole_and_lifetimes(rp)
    if d_arc is not None:
        return d_arc, Gpop_arc, True
    d = hydro_dipole_Cm(rp.n1_eff, rp.n2_eff, rp.mu_prefactor)
    Gpop = hydro_gamma_population(rp.n1_eff, rp.n2_eff, rp.tau0_0K, rp.bbr_factor_300K)
    return d, Gpop, False


def run_cw_pipeline(params: Dict, seed: int = 1, make_plots: bool = True):
    np.random.seed(seed)

    
    cav = CavityParams(**params["cavity"])
    u_func, volume, bounds = make_mode_u(cav)
    Veff = numerical_Veff_from_u(u_func, volume, bounds, cav.grid_N)
    omega_c = 2*pi*cav.f_c
    Ezpf = np.sqrt(HBAR*omega_c/(2*EPS0*Veff)) * cav.eta_pol
    kappa = omega_c / cav.Q_loaded
    kappa1 = cav.kappa_port1_frac * kappa
    kappa2 = cav.kappa_port2_frac * kappa

    
    rp = RydbergParams(**params["rydberg"])
    dipole, gamma_pop, used_arc = _rydberg_dipole_and_gamma(rp)
    gamma = gamma_pop + 2*pi*rp.gamma_phi_Hz
    g0 = single_atom_g(Ezpf, dipole, cg_factor=rp.cg_factor)

    
    atoms_params = params["atoms"].copy()
    N_mc = atoms_params.pop("N_mc")   
    cloud = AtomicCloud(**atoms_params)

    beam780 = GaussianBeam(params["optics"]["I780_Wm2"], params["optics"]["w780_m"])
    beam481 = GaussianBeam(params["optics"]["I481_Wm2"], params["optics"]["w481_m"])

    pos, pump_frac, N_pumped = estimate_pumped_atoms_fast(
        cloud=cloud, N_mc=N_mc,
        beam780=beam780, beam481=beam481,
        Delta_ge=2*pi*params["optics"]["Delta_ge_Hz"],
        Delta_er=2*pi*params["optics"]["Delta_er_Hz"],
        Gamma_e=2*pi*6.07e6,
        Gamma_r=2*pi*params["optics"].get("Gamma_r_Hz", 1.0e4),
        gamma_ge_phi=2*pi*params["optics"].get("gamma_ge_phi_Hz", 0.5e6),
        gamma_er_phi=2*pi*params["optics"].get("gamma_er_phi_Hz", 0.2e6),
        mu_er=None,
        grid_points=params["optics"].get("rabi_grid_points", 5)
    )

    
    G, gi, u_vals = collective_G(pos, u_func, g0, pump_frac)
    delta_ac = 2*pi*params["cw"].get("delta_ac_Hz", 0.0)
    span_Hz   = params["cw"].get("span_Hz", 40e6)
    npts      = params["cw"].get("npts", 1201)
    f_off_Hz, P21 = cw_S21_spectrum(G, kappa, gamma, delta_ac, kappa1, kappa2, span_Hz, npts)

    out = {
        "geometry": cav.geometry, "dims": cav.dims, "rect_mode_indices": cav.rect_mode_indices,
        "V_eff_m3": Veff, "Ezpf_V_per_m": Ezpf,
        "kappa_Hz": kappa/(2*pi), "kappa1_Hz": kappa1/(2*pi), "kappa2_Hz": kappa2/(2*pi),
        "dipole_Cm": dipole, "used_arc": used_arc,
        "g0_Hz": g0/(2*pi), "gamma_Hz": gamma/(2*pi),
        "N_pumped": float(N_pumped), "G_Hz": G/(2*pi),
        "split_Hz_approx": 2*G/(2*pi)
    }

    
    if make_plots:
        plt.figure()
        plt.plot(f_off_Hz*1e-6, P21)
        plt.xlabel("Drive detuning from cavity (MHz)")
        plt.ylabel("Normalized |S21|^2")
        plt.title("CW cavity transmission (vacuum Rabi splitting when strong-coupled)")
        plt.tight_layout(); plt.show()

        plt.figure()
        plt.bar(["G","κ","γ"], [out["G_Hz"], out["kappa_Hz"], out["gamma_Hz"]])
        plt.ylabel("Rate (Hz)"); plt.title("Key rates (CW)")
        plt.tight_layout(); plt.show()

    
    np.savetxt("rb_cqed_atoms_cw.csv",
               np.column_stack([pos, pump_frac, gi, u_vals]),
               delimiter=",",
               header="x[m],y[m],z[m],pump_fraction,gi[rad/s],u_norm",
               comments="")
    with open("rb_cqed_results_cw.json","w") as f: json.dump(out, f, indent=2)
    return out, (f_off_Hz, P21)


params = {
    "cavity": {
        "f_c": 15.0e9, "Q_loaded": 5.0e4,
        "geometry": "rect",
        "dims": (0.080, 0.020, 0.040),     
        "rect_mode_indices": (2,0,1),      
        "eta_pol": 1.0, "grid_N": 36,
        "kappa_port1_frac": 0.25, "kappa_port2_frac": 0.25
    },
    "rydberg": {
        "n1_eff": 50.0, "n2_eff": 51.0,
        "mu_prefactor": 1.5, "tau0_0K": 1.0e-9, "bbr_factor_300K": 0.6,
        "gamma_phi_Hz": 0.2e6, "cg_factor": 0.7,
        "use_arc": False, "isotope": "Rb85",
        "state1": None, "state2": None, "temperature_K": 300.0
    },
    "atoms": {
        "N_total": 2_000_000,
        "sigma_xyz_m": (0.7e-3, 0.7e-3, 0.7e-3),
        "center_m": (0.0, 0.0, 0.0),
        "N_mc": 120
    },
    "optics": {
        "I780_Wm2": 200.0, "w780_m": 5.0e-3,
        "I481_Wm2": 5.0e3, "w481_m": 3.0e-3,
        "Delta_ge_Hz": -15.0e6, "Delta_er_Hz": 0.0,
        "gamma_ge_phi_Hz": 0.5e6, "gamma_er_phi_Hz": 0.2e6,
        "Gamma_r_Hz": 1.0e4, "rabi_grid_points": 5
    },
    "cw": { "delta_ac_Hz": 0.0, "span_Hz": 40e6, "npts": 1201 }
}


if __name__ == "__main__":
    out, (f_off, P21) = run_cw_pipeline(params, seed=3, make_plots=True)
    print("Summary:")
    for k in ["V_eff_m3","Ezpf_V_per_m","g0_Hz","G_Hz","kappa_Hz","gamma_Hz","split_Hz_approx","N_pumped","used_arc"]:
        print(f"  {k}: {out[k]}")
    print("\nWrote:\n  - rb_cqed_results_cw.json\n  - rb_cqed_atoms_cw.csv")