import numpy as np
import json
from constants import PI, EPS0, HBAR, C0
from cavity_modes import make_mode_u, numerical_Veff_from_u
from data_structures import CavityParams, RydbergParams
from rydberg_physics import try_arc_dipole_and_lifetimes, hydro_dipole_Cm, single_atom_g

def optimize_cavity_for_strong_coupling():
    """Find cavity geometries that achieve strong coupling."""
    
    # Target frequency
    f_target = 13.6713e9
    lambda_mw = C0 / f_target
    
    print("CAVITY OPTIMIZATION FOR STRONG COUPLING")
    print("=" * 50)
    print(f"Target frequency: {f_target/1e9:.4f} GHz")
    print(f"Wavelength: {lambda_mw*100:.3f} cm")
    
    # Rydberg parameters (same as original)
    rydberg_params = {
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
    }
    
    rp = RydbergParams(**rydberg_params)
    
    # Get actual dipole moment
    dipole, gamma_pop = try_arc_dipole_and_lifetimes(rp)
    if dipole is None:
        dipole = hydro_dipole_Cm(rp.n1_eff, rp.n2_eff) * rp.mu_prefactor
        print(f"Using hydrogenic dipole: {dipole/1.602e-19/5.29e-11:.0f} ea₀")
    else:
        print(f"Using ARC dipole: {dipole/1.602e-19/5.29e-11:.0f} ea₀")
    
    candidates = []
    
    # Strategy 1: Smaller cavities with higher-order modes
    # TE301 mode - higher m reduces width significantly
    print(f"\nTesting TE301 mode...")
    a1 = lambda_mw / 6  # Much smaller width
    b1 = lambda_mw / 8  # Smaller height  
    # f = (c/2) * sqrt((3/a)^2 + (0/b)^2 + (1/d)^2)
    term1 = (2*f_target/C0)**2 - (3/a1)**2
    if term1 > 0:
        d1 = 1.0 / np.sqrt(term1)
        candidates.append({
            'name': 'TE301 Compact',
            'mode': (3,0,1),
            'dims': (a1, b1, d1),
            'volume': a1*b1*d1
        })
    
    # TE401 mode - even smaller
    print(f"Testing TE401 mode...")
    a2 = lambda_mw / 8
    b2 = lambda_mw / 10
    term2 = (2*f_target/C0)**2 - (4/a2)**2
    if term2 > 0:
        d2 = 1.0 / np.sqrt(term2)
        candidates.append({
            'name': 'TE401 Ultra-compact',
            'mode': (4,0,1),
            'dims': (a2, b2, d2),
            'volume': a2*b2*d2
        })
    
    # TE211 mode - balanced reduction
    print(f"Testing TE211 mode...")
    a3 = lambda_mw / 4
    b3 = lambda_mw / 6
    term3 = (2*f_target/C0)**2 - (2/a3)**2 - (1/b3)**2
    if term3 > 0:
        d3 = 1.0 / np.sqrt(term3)
        candidates.append({
            'name': 'TE211 Balanced',
            'mode': (2,1,1),
            'dims': (a3, b3, d3),
            'volume': a3*b3*d3
        })
    
    # TE103 mode - long thin cavity
    print(f"Testing TE103 mode...")
    a4 = lambda_mw / 3
    b4 = lambda_mw / 12  # Very thin
    term4 = (2*f_target/C0)**2 - (1/a4)**2
    if term4 > 0:
        d4 = 3.0 / np.sqrt(term4)
        candidates.append({
            'name': 'TE103 Thin',
            'mode': (1,0,3),
            'dims': (a4, b4, d4),
            'volume': a4*b4*d4
        })
    
    # Add original for comparison
    candidates.append({
        'name': 'Original TE201',
        'mode': (2,0,1),
        'dims': (0.016447, 0.008223, 0.029420),
        'volume': 0.016447 * 0.008223 * 0.029420
    })
    
    print(f"\nEvaluating {len(candidates)} cavity designs...")
    print("-" * 50)
    
    results = []
    
    for candidate in candidates:
        cavity_params = {
            "f_c": f_target,
            "Q_loaded": 5.0e4,
            "geometry": "rect",
            "dims": candidate['dims'],
            "eta_pol": 1.0,
            "grid_N": 36,
            "rect_mode_indices": candidate['mode'],
            "kappa_port1_frac": 0.25,
            "kappa_port2_frac": 0.25
        }
        
        try:
            cav = CavityParams(**cavity_params)
            
            # Calculate actual coupling using the same method as main.py
            u_func, volume, bounds = make_mode_u(cav)
            V_eff = numerical_Veff_from_u(u_func, volume, bounds, cav.grid_N)
            omega_c = 2 * PI * cav.f_c
            Ezpf = np.sqrt(HBAR * omega_c / (2 * EPS0 * V_eff)) * cav.eta_pol
            g0 = single_atom_g(Ezpf, dipole, rp.cg_factor)
            
            # System rates
            omega_c = 2 * PI * cav.f_c
            kappa = omega_c / cav.Q_loaded
            gamma = rp.gamma_phi_Hz * 2 * PI
            
            # Coupling analysis
            cooperativity = 4 * g0**2 / (kappa * gamma)
            strong_coupling = g0 > (kappa + gamma) / 4
            
            # V_eff already calculated above
            
            result = {
                'name': candidate['name'],
                'mode': candidate['mode'],
                'dims_mm': [d*1000 for d in candidate['dims']],
                'V_eff_mm3': V_eff * 1e9,
                'g0_kHz': g0 / (2 * PI * 1000),
                'kappa_kHz': kappa / (2 * PI * 1000),
                'gamma_kHz': gamma / (2 * PI * 1000),
                'cooperativity': cooperativity,
                'strong_coupling': strong_coupling,
                'regime': 'Strong' if strong_coupling else 'Weak',
                'cavity_params': cavity_params
            }
            
            results.append(result)
            
            print(f"{candidate['name']:20s} | "
                  f"g₀={result['g0_kHz']:6.1f} kHz | "
                  f"C={result['cooperativity']:5.2f} | "
                  f"{result['regime']:6s} | "
                  f"V_eff={result['V_eff_mm3']:6.2f} mm³")
                  
        except Exception as e:
            print(f"{candidate['name']:20s} | ERROR: {str(e)}")
    
    # Sort by coupling strength
    results.sort(key=lambda x: x['g0_kHz'], reverse=True)
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS (sorted by coupling strength)")
    print("=" * 70)
    
    original_g0 = None
    for i, result in enumerate(results):
        if 'Original' in result['name']:
            original_g0 = result['g0_kHz']
        
        improvement = result['g0_kHz'] / original_g0 if original_g0 else 1.0
        
        print(f"\n{i+1}. {result['name']}")
        print(f"   Mode: TE{result['mode'][0]}{result['mode'][1]}{result['mode'][2]}")
        print(f"   Dimensions: {result['dims_mm'][0]:.2f} × {result['dims_mm'][1]:.2f} × {result['dims_mm'][2]:.2f} mm")
        print(f"   g₀: {result['g0_kHz']:.1f} kHz ({improvement:.1f}× improvement)")
        print(f"   Cooperativity: {result['cooperativity']:.2f}")
        print(f"   Regime: {result['regime']}")
        print(f"   V_eff: {result['V_eff_mm3']:.2f} mm³")
    
    # Save the best strong coupling candidate
    strong_candidates = [r for r in results if r['strong_coupling']]
    
    if strong_candidates:
        best = strong_candidates[0]
        
        optimized_params = {
            "cavity": best['cavity_params'],
            "rydberg": rydberg_params,
            "atoms": {
                "N_total": 2_000_000,
                "sigma_xyz_m": [0.7e-3, 0.7e-3, 0.7e-3],
                "center_m": [0.0, 0.0, 0.0],
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
            },
            "description": f"Optimized {best['name']} for strong coupling"
        }
        
        with open('strong_coupling_params.json', 'w') as f:
            json.dump(optimized_params, f, indent=2)
        
        print(f"\n" + "="*50)
        print("SUCCESS: Strong coupling geometry found!")
        print(f"Best candidate saved to 'strong_coupling_params.json'")
        print(f"Expected improvement: {best['g0_kHz']/original_g0:.1f}× stronger coupling")
        print("="*50)
        
        return best
    else:
        print(f"\n" + "="*50)
        print("No strong coupling geometries found with current approach.")
        print("Consider: Higher Q, different Rydberg states, or fabrication constraints.")
        print("="*50)
        return None

if __name__ == "__main__":
    best_geometry = optimize_cavity_for_strong_coupling()
