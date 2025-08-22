import numpy as np
import json
from constants import PI, EPS0, HBAR, C0
from cavity_modes import make_mode_u, numerical_Veff_from_u
from data_structures import CavityParams, RydbergParams
from rydberg_physics import try_arc_dipole_and_lifetimes, hydro_dipole_Cm, single_atom_g

def find_strong_coupling_cavity():
    """Find cavity geometries that achieve strong coupling by systematically reducing mode volume."""
    
    f_target = 13.6713e9
    lambda_mw = C0 / f_target
    
    print("SYSTEMATIC CAVITY OPTIMIZATION FOR STRONG COUPLING")
    print("=" * 60)
    print(f"Target: g₀ > (κ + γ)/4 ≈ 150 kHz for strong coupling")
    print(f"Strategy: Minimize mode volume while maintaining frequency")
    
    # Rydberg parameters
    rydberg_params = {
        "n1_eff": 81.0, "n2_eff": 82.0, "mu_prefactor": 1.5,
        "tau0_0K": 1.0e-9, "bbr_factor_300K": 0.6, "gamma_phi_Hz": 0.2e6,
        "cg_factor": 0.7, "use_arc": True, "isotope": "Rb85",
        "state1": (81,2,2.5,0.5), "state2": (82,1,1.5,0.5), "temperature_K": 300.0
    }
    
    rp = RydbergParams(**rydberg_params)
    dipole, _ = try_arc_dipole_and_lifetimes(rp)
    if dipole is None:
        dipole = hydro_dipole_Cm(rp.n1_eff, rp.n2_eff) * rp.mu_prefactor
    
    candidates = []
    
    # Original for reference
    candidates.append({
        'name': 'Original TE201',
        'mode': (2,0,1),
        'dims': (0.016447, 0.008223, 0.029420)
    })
    
    # Strategy: Use higher-order modes to create smaller cavities
    # For TE_mnp: f = (c/2) * sqrt((m/a)² + (n/b)² + (p/d)²)
    
    # TE301: Reduce width by using higher m
    a_301 = 3 * C0 / (4 * f_target)  # From frequency equation with n=0, p=1
    remaining_301 = (2*f_target/C0)**2 - (3/a_301)**2
    if remaining_301 > 0:
        d_301 = 1.0 / np.sqrt(remaining_301)
        b_301 = lambda_mw / 15  # Make height small
        candidates.append({
            'name': 'TE301 Narrow',
            'mode': (3,0,1),
            'dims': (a_301, b_301, d_301)
        })
    
    # TE401: Even narrower
    a_401 = 4 * C0 / (4 * f_target)
    remaining_401 = (2*f_target/C0)**2 - (4/a_401)**2
    if remaining_401 > 0:
        d_401 = 1.0 / np.sqrt(remaining_401)
        b_401 = lambda_mw / 20
        candidates.append({
            'name': 'TE401 Ultra-narrow',
            'mode': (4,0,1),
            'dims': (a_401, b_401, d_401)
        })
    
    # TE211: Use both m and n modes
    # (2/a)² + (1/b)² + (1/d)² = (2f/c)²
    a_211 = lambda_mw / 3
    b_211 = lambda_mw / 8
    remaining_211 = (2*f_target/C0)**2 - (2/a_211)**2 - (1/b_211)**2
    if remaining_211 > 0:
        d_211 = 1.0 / np.sqrt(remaining_211)
        candidates.append({
            'name': 'TE211 Compact',
            'mode': (2,1,1),
            'dims': (a_211, b_211, d_211)
        })
    
    # TE312: High-order mixed mode
    a_312 = lambda_mw / 4
    b_312 = lambda_mw / 10
    remaining_312 = (2*f_target/C0)**2 - (3/a_312)**2 - (1/b_312)**2
    if remaining_312 > 0:
        d_312 = 2.0 / np.sqrt(remaining_312)
        candidates.append({
            'name': 'TE312 Mixed',
            'mode': (3,1,2),
            'dims': (a_312, b_312, d_312)
        })
    
    # TE103: Long thin cavity
    a_103 = C0 / (4 * f_target)  # From (1/a)² + (3/d)² = (2f/c)²
    remaining_103 = (2*f_target/C0)**2 - (1/a_103)**2
    if remaining_103 > 0:
        d_103 = 3.0 / np.sqrt(remaining_103)
        b_103 = lambda_mw / 25  # Very thin
        candidates.append({
            'name': 'TE103 Thin-long',
            'mode': (1,0,3),
            'dims': (a_103, b_103, d_103)
        })
    
    print(f"\nEvaluating {len(candidates)} cavity designs...")
    print("-" * 60)
    
    results = []
    
    for candidate in candidates:
        try:
            cavity_params = {
                "f_c": f_target, "Q_loaded": 5.0e4, "geometry": "rect",
                "dims": candidate['dims'], "eta_pol": 1.0, "grid_N": 36,
                "rect_mode_indices": candidate['mode'],
                "kappa_port1_frac": 0.25, "kappa_port2_frac": 0.25
            }
            
            cav = CavityParams(**cavity_params)
            
            # Calculate coupling
            u_func, volume, bounds = make_mode_u(cav)
            V_eff = numerical_Veff_from_u(u_func, volume, bounds, cav.grid_N)
            omega_c = 2 * PI * cav.f_c
            Ezpf = np.sqrt(HBAR * omega_c / (2 * EPS0 * V_eff)) * cav.eta_pol
            g0 = single_atom_g(Ezpf, dipole, rp.cg_factor)
            
            # System analysis
            kappa = omega_c / cav.Q_loaded
            gamma = rp.gamma_phi_Hz * 2 * PI
            cooperativity = 4 * g0**2 / (kappa * gamma)
            strong_coupling = g0 > (kappa + gamma) / 4
            
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
                'cavity_params': cavity_params,
                'volume_reduction': (0.016447 * 0.008223 * 0.029420) / (candidate['dims'][0] * candidate['dims'][1] * candidate['dims'][2])
            }
            
            results.append(result)
            
            status = "✓ STRONG" if strong_coupling else "  weak"
            print(f"{candidate['name']:18s} | {status} | g₀={result['g0_kHz']:6.1f} kHz | C={result['cooperativity']:5.2f} | V_eff={result['V_eff_mm3']:6.1f} mm³")
                  
        except Exception as e:
            print(f"{candidate['name']:18s} | ERROR: {str(e)[:50]}")
    
    # Sort by coupling strength
    results.sort(key=lambda x: x['g0_kHz'], reverse=True)
    
    print("\n" + "=" * 80)
    print("DETAILED RESULTS (sorted by coupling strength)")
    print("=" * 80)
    
    original_g0 = next((r['g0_kHz'] for r in results if 'Original' in r['name']), 100)
    
    for i, result in enumerate(results):
        improvement = result['g0_kHz'] / original_g0
        
        print(f"\n{i+1}. {result['name']} ({'STRONG COUPLING' if result['strong_coupling'] else 'weak coupling'})")
        print(f"   Mode: TE{result['mode'][0]}{result['mode'][1]}{result['mode'][2]}")
        print(f"   Dimensions: {result['dims_mm'][0]:.2f} × {result['dims_mm'][1]:.2f} × {result['dims_mm'][2]:.2f} mm")
        print(f"   Volume reduction: {result['volume_reduction']:.1f}×")
        print(f"   g₀: {result['g0_kHz']:.1f} kHz ({improvement:.1f}× improvement)")
        print(f"   Cooperativity: {result['cooperativity']:.2f}")
        print(f"   V_eff: {result['V_eff_mm3']:.1f} mm³")
    
    # Save best strong coupling candidate
    strong_candidates = [r for r in results if r['strong_coupling']]
    
    if strong_candidates:
        best = strong_candidates[0]
        
        # Create complete parameter set
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
                "I780_Wm2": 200.0, "w780_m": 5.0e-3, "I481_Wm2": 5.0e3, "w481_m": 3.0e-3,
                "Delta_ge_Hz": -15.0e6, "Delta_er_Hz": 0.0, "gamma_ge_phi_Hz": 0.5e6,
                "gamma_er_phi_Hz": 0.2e6, "Gamma_r_Hz": 1.0e4, "rabi_grid_points": 5
            },
            "cw": {"delta_ac_Hz": 0.0, "span_Hz": 40e6, "npts": 1201},
            "description": f"Optimized {best['name']} achieving strong coupling with {best['g0_kHz']:.1f} kHz coupling"
        }
        
        with open('strong_coupling_params.json', 'w') as f:
            json.dump(optimized_params, f, indent=2)
        
        print(f"\n" + "="*70)
        print("🎉 SUCCESS: STRONG COUPLING ACHIEVED!")
        print(f"Best geometry: {best['name']}")
        print(f"Coupling strength: {best['g0_kHz']:.1f} kHz ({best['g0_kHz']/original_g0:.1f}× improvement)")
        print(f"Cooperativity: {best['cooperativity']:.2f}")
        print(f"Parameters saved to: strong_coupling_params.json")
        print("="*70)
        
        return best
    else:
        print(f"\n" + "="*50)
        print("No strong coupling achieved with current geometries.")
        print("Best coupling:", max(results, key=lambda x: x['g0_kHz'])['g0_kHz'], "kHz")
        print("="*50)
        return None

if __name__ == "__main__":
    best_geometry = find_strong_coupling_cavity()
