import json
import numpy as np

from constants import PI, EPS0, HBAR
from data_structures import CavityParams, RydbergParams, AtomicCloud, GaussianBeam
from cavity_modes import make_mode_u, numerical_Veff_from_u
from rydberg_physics import (
    try_arc_dipole_and_lifetimes, hydro_dipole_Cm, hydro_gamma_population,
    single_atom_g, estimate_collective_G
)
from optical_pumping import estimate_pumped_atoms_fast
from visualization import make_plots
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def calculate_vacuum_rabi_splitting(G, kappa, gamma, delta_ac=0.0):
    G_Hz = G / (2 * PI)
    kappa_Hz = kappa / (2 * PI)
    gamma_Hz = gamma / (2 * PI)
    
    
    C = 4 * G**2 / (kappa * gamma)
    
    
    strong_coupling = G > (kappa + gamma) / 2
    
    
    if delta_ac == 0.0:
        
        splitting_Hz = 2 * G_Hz
        
        if strong_coupling:
            eff_splitting_Hz = 2 * np.sqrt(G_Hz**2 - ((kappa_Hz + gamma_Hz)/4)**2)
        else:
            eff_splitting_Hz = 0.0  
    else:
        
        delta_ac_Hz = delta_ac / (2 * PI)
        omega_plus = 0.5 * (delta_ac + np.sqrt(delta_ac**2 + 4*G**2))
        omega_minus = 0.5 * (delta_ac - np.sqrt(delta_ac**2 + 4*G**2))
        splitting_Hz = (omega_plus - omega_minus) / (2 * PI)
        eff_splitting_Hz = splitting_Hz
    
    return {
        'cooperativity': C,
        'strong_coupling': strong_coupling,
        'splitting_Hz': splitting_Hz,
        'effective_splitting_Hz': max(0, eff_splitting_Hz),
        'regime': 'Strong' if strong_coupling else 'Weak'
    }


def cw_S21_spectrum(G, kappa, gamma, delta_ac, kappa1, kappa2, span_Hz=40e6, npts=1201):
    
    freqs = np.linspace(-span_Hz/2, +span_Hz/2, npts)   
    Dc = 2 * PI * freqs                                 
    Da = Dc - delta_ac                                  
    D = (kappa/2 + 1j*Dc) + (G**2)/(gamma/2 + 1j*Da)   
    S21 = np.sqrt(kappa1*kappa2) / D
    P21 = np.abs(S21)**2
    
    peaks, properties = find_peaks(P21, height=0.1*P21.max(), distance=20)
    
    peak_info = {
        'n_peaks': len(peaks),
        'peak_freqs_MHz': freqs[peaks] * 1e-6 if len(peaks) > 0 else [],
        'measured_splitting_MHz': 0.0
    }
    
    if len(peaks) >= 2:
        
        peak_info['measured_splitting_MHz'] = abs(freqs[peaks[-1]] - freqs[peaks[0]]) * 1e-6
    
    
    if P21.max() > 0: 
        P21 /= P21.max()
    
    return freqs, P21, peak_info


def run_cw_pipeline(params: dict, seed: int = 1, make_plots_flag: bool = True):
    
    np.random.seed(seed)

    
    cav = CavityParams(**params["cavity"])
    u_func, volume, bounds = make_mode_u(cav)
    V_eff = numerical_Veff_from_u(u_func, volume, bounds, cav.grid_N)
    omega_c = 2 * PI * cav.f_c
    Ezpf = np.sqrt(HBAR * omega_c / (2 * EPS0 * V_eff)) * cav.eta_pol
    kappa = omega_c / cav.Q_loaded
    
    
    kappa1 = params["cavity"].get("kappa_port1_frac", 0.25) * kappa
    kappa2 = params["cavity"].get("kappa_port2_frac", 0.25) * kappa

    
    rp = RydbergParams(**params["rydberg"])
    d_arc, Gpop_arc = try_arc_dipole_and_lifetimes(rp)
    if d_arc is not None:
        dipole = d_arc
        gamma_pop = Gpop_arc
        used_arc = True
    else:
        dipole = hydro_dipole_Cm(rp.n1_eff, rp.n2_eff, rp.mu_prefactor)
        gamma_pop = hydro_gamma_population(rp.n1_eff, rp.n2_eff, rp.tau0_0K, rp.bbr_factor_300K)
        used_arc = False
    gamma = gamma_pop + 2 * PI * rp.gamma_phi_Hz
    g0 = single_atom_g(Ezpf, dipole, cg_factor=rp.cg_factor)

    
    cloud = AtomicCloud(N_total=params["atoms"]["N_total"],
                        sigma_xyz=tuple(params["atoms"]["sigma_xyz_m"]),
                        center=tuple(params["atoms"].get("center_m", (0, 0, 0))))
    beam780 = GaussianBeam(I0=params["optics"]["I780_Wm2"], w=params["optics"]["w780_m"])
    beam481 = GaussianBeam(I0=params["optics"]["I481_Wm2"], w=params["optics"]["w481_m"])

    pos, pump_frac, N_pumped = estimate_pumped_atoms_fast(
        cloud=cloud, N_mc=params["atoms"]["N_mc"],
        beam780=beam780, beam481=beam481,
        Delta_ge=2 * PI * params["optics"]["Delta_ge_Hz"],
        Delta_er=2 * PI * params["optics"]["Delta_er_Hz"],
        Gamma_e=2 * PI * 6.07e6,
        Gamma_r=2 * PI * params["optics"].get("Gamma_r_Hz", 1.0e4),
        gamma_ge_phi=2 * PI * params["optics"].get("gamma_ge_phi_Hz", 0.5e6),
        gamma_er_phi=2 * PI * params["optics"].get("gamma_er_phi_Hz", 0.2e6),
        mu_er=None,
        grid_points=params["optics"].get("rabi_grid_points", 5)
    )

    
    G, gi, u_vals = estimate_collective_G(pos, u_func, g0, pump_frac)
    
    
    delta_ac = 2 * PI * params["cw"].get("delta_ac_Hz", 0.0)
    splitting_analysis = calculate_vacuum_rabi_splitting(G, kappa, gamma, delta_ac)
    
    
    span_Hz = params["cw"].get("span_Hz", 40e6)
    npts = params["cw"].get("npts", 1201)
    f_off_Hz, P21, peak_info = cw_S21_spectrum(G, kappa, gamma, delta_ac, kappa1, kappa2, span_Hz, npts)

    out = {
        "geometry": cav.geometry,
        "dims": cav.dims,
        "rect_mode_indices": getattr(cav, 'rect_mode_indices', None),
        "V_eff_m3": V_eff,
        "volume_m3": volume,
        "Ezpf_V_per_m": Ezpf,
        "kappa_rad_per_s": kappa,
        "kappa_Hz": kappa / (2 * PI),
        "kappa1_Hz": kappa1 / (2 * PI),
        "kappa2_Hz": kappa2 / (2 * PI),
        "dipole_Cm": dipole,
        "used_arc": used_arc,
        "g0_rad_per_s": g0,
        "g0_Hz": g0 / (2 * PI),
        "gamma_rad_per_s": gamma,
        "gamma_Hz": gamma / (2 * PI),
        "N_pumped": float(N_pumped),
        "G_rad_per_s": G,
        "G_Hz": G / (2 * PI),
        "cooperativity": splitting_analysis['cooperativity'],
        "coupling_regime": splitting_analysis['regime'],
        "vacuum_rabi_splitting_Hz": splitting_analysis['splitting_Hz'],
        "effective_splitting_Hz": splitting_analysis['effective_splitting_Hz'],
        "measured_splitting_MHz": peak_info['measured_splitting_MHz'],
        "n_spectral_peaks": peak_info['n_peaks']
    }

    
    with open("rb_cqed_results_cw.json", "w") as f:
        json.dump(out, f, indent=2)
    np.savetxt("rb_cqed_atoms_cw.csv",
               np.column_stack([pos, pump_frac, gi, u_vals]),
               delimiter=",",
               header="x[m],y[m],z[m],pump_fraction,gi[rad/s],u_norm",
               comments="")

    if make_plots_flag:
        
        fig = plt.figure(figsize=(15, 10))
        
        
        ax1 = plt.subplot(2, 3, (1, 2))
        plt.plot(f_off_Hz * 1e-6, P21, 'b-', linewidth=2)
        
        
        if peak_info['n_peaks'] > 0:
            peak_freqs = np.array(peak_info['peak_freqs_MHz'])
            peak_vals = P21[np.abs(f_off_Hz[:, None] - peak_freqs[None, :] * 1e6).argmin(axis=0)]
            plt.plot(peak_freqs, peak_vals, 'ro', markersize=8, label=f'{peak_info["n_peaks"]} peaks')
            if peak_info['measured_splitting_MHz'] > 0:
                plt.axvspan(peak_freqs[0], peak_freqs[-1], alpha=0.2, color='red', 
                           label=f'Splitting: {peak_info["measured_splitting_MHz"]:.1f} MHz')
        
        plt.xlabel("Drive detuning from cavity (MHz)")
        plt.ylabel("Normalized |S21|²")
        plt.title(f"CW Transmission Spectrum\n{splitting_analysis['regime']} Coupling (C={splitting_analysis['cooperativity']:.2f})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        
        ax2 = plt.subplot(2, 3, 3)
        rates = [out["G_Hz"], out["kappa_Hz"], out["gamma_Hz"]]
        colors = ['red', 'blue', 'green']
        bars = plt.bar(["G", "κ", "γ"], rates, color=colors, alpha=0.7)
        plt.ylabel("Rate (Hz)")
        plt.title("Key Rates")
        plt.yscale('log')
        
        
        for bar, rate in zip(bars, rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                    f'{rate:.0f}', ha='center', va='bottom', fontsize=10)
        
        
        ax3 = plt.subplot(2, 3, 4)
        coupling_data = {
            'g₀ (single atom)': out["g0_Hz"],
            'G (collective)': out["G_Hz"],
            'Theoretical Split': out["vacuum_rabi_splitting_Hz"],
            'Effective Split': out["effective_splitting_Hz"]
        }
        
        bars = plt.bar(range(len(coupling_data)), list(coupling_data.values()), 
                      color=['orange', 'red', 'purple', 'darkred'], alpha=0.7)
        plt.xticks(range(len(coupling_data)), list(coupling_data.keys()), rotation=45)
        plt.ylabel("Frequency (Hz)")
        plt.title("Coupling Analysis")
        plt.yscale('log')
        
        
        ax4 = plt.subplot(2, 3, 5)
        ax4.axis('off')
        param_text = f"""
System Parameters:
• Cavity: {cav.f_c/1e9:.2f} GHz, Q = {cav.Q_loaded:.0e}
• Mode: {cav.geometry} TE{cav.rect_mode_indices[0]}{cav.rect_mode_indices[1]}{cav.rect_mode_indices[2]}
• Rydberg: N={rp.n1_eff:.0f},{rp.n2_eff:.0f}
• Atoms: {out['N_pumped']:.0f} pumped

Results:
• Cooperativity: {splitting_analysis['cooperativity']:.2f}
• Regime: {splitting_analysis['regime']} coupling
• V_eff: {V_eff*1e6:.1f} mm³
• Splitting: {splitting_analysis['effective_splitting_Hz']/1e3:.1f} kHz
"""
        plt.text(0.05, 0.95, param_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        
        ax5 = plt.subplot(2, 3, 6)
        
        x = np.linspace(-cav.dims[0]/2, cav.dims[0]/2, 50)
        y = np.linspace(-cav.dims[1]/2, cav.dims[1]/2, 50)
        X, Y = np.meshgrid(x, y)
        
        
        m, n, p = cav.rect_mode_indices
        if m > 0:
            mode_x = np.sin(m * PI * (X + cav.dims[0]/2) / cav.dims[0])
        else:
            mode_x = np.ones_like(X)
        if n > 0:
            mode_y = np.sin(n * PI * (Y + cav.dims[1]/2) / cav.dims[1])
        else:
            mode_y = np.ones_like(Y)
        
        mode_pattern = np.abs(mode_x * mode_y)
        
        im = plt.imshow(mode_pattern, extent=[-cav.dims[0]/2*1000, cav.dims[0]/2*1000, 
                                             -cav.dims[1]/2*1000, cav.dims[1]/2*1000],
                       origin='lower', cmap='hot', alpha=0.8)
        plt.colorbar(im, ax=ax5, shrink=0.8)
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title(f'TE{m}{n}{p} Mode Pattern\n(z=0 cross-section)')
        
        plt.tight_layout()
        plt.show()

    return out, (f_off_Hz, P21, splitting_analysis, peak_info)



params = {
    "cavity": {
        "f_c": 13.6713e9,            
        "Q_loaded": 5.0e4,
        "geometry": "rect",          
        "dims": (0.016447, 0.008223, 0.029420),  
        "eta_pol": 1.0,
        "grid_N": 36,                 
        "rect_mode_indices": (2, 0, 1),  
        "kappa_port1_frac": 0.25,    
        "kappa_port2_frac": 0.25,    
    },
    "rydberg": {
        "n1_eff": 81.0,              
        "n2_eff": 82.0,              
        "mu_prefactor": 1.5,
        "tau0_0K": 1.0e-9,
        "bbr_factor_300K": 0.6,
        "gamma_phi_Hz": 0.2e6,
        "cg_factor": 0.7,
        
        "use_arc": True,            
        "isotope": "Rb85",
        "state1": (81,2,2.5,0.5),              
        "state2": (82,1,1.5,0.5),              
        "temperature_K": 300.0
    },
    "atoms": {
        "N_total": 2_000_000,
        "sigma_xyz_m": (0.7e-3, 0.7e-3, 0.7e-3),
        "center_m": (0.0, 0.0, 0.0),
        "N_mc": 80                   
    },
    "optics": {
        "I780_Wm2": 200.0,
        "w780_m": 5.0e-3,
        "I481_Wm2": 5.0e3,
        "w481_m": 3.0e-3,
        "Delta_ge_Hz": -15.0e6,
        "Delta_er_Hz": 0.0,
        "gamma_ge_phi_Hz": 0.5e6,
        "gamma_er_phi_Hz": 0.2e6,
        "Gamma_r_Hz": 1.0e4,
        "rabi_grid_points": 5        
    },
    "cw": {
        "delta_ac_Hz": 0.0,          
        "span_Hz": 40e6,             
        "npts": 1201                 
    }
}


if __name__ == "__main__":
    def load_json(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    
    strong_coupling_params = load_json("strong_coupling_params.json")
    params = strong_coupling_params
    out, (f_off_Hz, P21, splitting_analysis, peak_info) = run_cw_pipeline(params, seed=2, make_plots_flag=True)
    
    print("\n" + "="*60)
    print("CW CAVITY-QED SIMULATION RESULTS")
    print("="*60)
    
    print(f"\nSystem Configuration:")
    print(f"  Cavity: {out['geometry']} TE{out['rect_mode_indices'][0]}{out['rect_mode_indices'][1]}{out['rect_mode_indices'][2]} @ {params['cavity']['f_c']/1e9:.3f} GHz")
    print(f"  Rydberg states: N={params['rydberg']['n1_eff']:.0f},{params['rydberg']['n2_eff']:.0f}")
    print(f"  Pumped atoms: {out['N_pumped']:.0f}")
    
    print(f"\nCoupling Analysis:")
    print(f"  Single-atom g₀: {out['g0_Hz']:.0f} Hz")
    print(f"  Collective G: {out['G_Hz']:.0f} Hz")
    print(f"  Cooperativity C: {out['cooperativity']:.2f}")
    print(f"  Coupling regime: {out['coupling_regime']}")
    
    print(f"\nVacuum Rabi Splitting:")
    print(f"  Theoretical splitting: {out['vacuum_rabi_splitting_Hz']/1e3:.1f} kHz")
    print(f"  Effective splitting: {out['effective_splitting_Hz']/1e3:.1f} kHz")
    print(f"  Measured from spectrum: {out['measured_splitting_MHz']*1e3:.1f} kHz")
    print(f"  Spectral peaks found: {out['n_spectral_peaks']}")
    
    print(f"\nDecay Rates:")
    print(f"  Cavity κ: {out['kappa_Hz']:.0f} Hz")
    print(f"  Atomic γ: {out['gamma_Hz']:.0f} Hz")
    print(f"  κ/γ ratio: {out['kappa_Hz']/out['gamma_Hz']:.2f}")
    
    print(f"\nFiles written:")
    print(f"  - rb_cqed_results_cw.json")
    print(f"  - rb_cqed_atoms_cw.csv")
    print("="*60)
