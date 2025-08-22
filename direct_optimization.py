import numpy as np
import json
from constants import PI, EPS0, HBAR, C0
from cavity_modes import make_mode_u, numerical_Veff_from_u
from data_structures import CavityParams, RydbergParams
from rydberg_physics import try_arc_dipole_and_lifetimes, hydro_dipole_Cm, single_atom_g

def create_optimized_geometries():
    """Create specific cavity geometries designed for strong coupling."""
    
    f_target = 13.6713e9
    lambda_mw = C0 / f_target
    
    # Current system: g₀ ≈ 99 kHz, need > 150 kHz for strong coupling
    # Since g ∝ 1/√V_eff, need to reduce V_eff by factor of (150/99)² ≈ 2.3
    
    geometries = []
    
    # Original geometry
    geometries.append({
        'name': 'Original TE201',
        'mode': (2, 0, 1),
        'dims': (0.016447, 0.008223, 0.029420),
        'description': 'Current baseline geometry'
    })
    
    # Geometry 1: TE301 - Higher m mode allows smaller width
    # For TE301: f = (c/2) * sqrt((3/a)² + (1/d)²)
    # Solve: (3/a)² + (1/d)² = (2f/c)²
    target_term = (2 * f_target / C0) ** 2
    
    # Choose smaller width
    a1 = 0.010  # 10 mm width (vs 16.45 mm original)
    term1 = (3 / a1) ** 2
    if target_term > term1:
        d1 = 1.0 / np.sqrt(target_term - term1)
        b1 = 0.004  # 4 mm height (vs 8.22 mm original)
        geometries.append({
            'name': 'TE301 Compact',
            'mode': (3, 0, 1),
            'dims': (a1, b1, d1),
            'description': 'Higher-order mode with reduced width'
        })
    
    # Geometry 2: TE401 - Even higher m mode
    a2 = 0.008  # 8 mm width
    term2 = (4 / a2) ** 2
    if target_term > term2:
        d2 = 1.0 / np.sqrt(target_term - term2)
        b2 = 0.003  # 3 mm height
        geometries.append({
            'name': 'TE401 Ultra-compact',
            'mode': (4, 0, 1),
            'dims': (a2, b2, d2),
            'description': 'Very high-order mode, minimal width'
        })
    
    # Geometry 3: TE211 - Mixed mode
    # (2/a)² + (1/b)² + (1/d)² = (2f/c)²
    a3 = 0.012  # 12 mm
    b3 = 0.015  # 15 mm  
    remaining = target_term - (2/a3)**2 - (1/b3)**2
    if remaining > 0:
        d3 = 1.0 / np.sqrt(remaining)
        geometries.append({
            'name': 'TE211 Mixed',
            'mode': (2, 1, 1),
            'dims': (a3, b3, d3),
            'description': 'Mixed mode with balanced dimensions'
        })
    
    # Geometry 4: TE103 - Long thin cavity
    # (1/a)² + (3/d)² = (2f/c)²
    a4 = 0.020  # 20 mm width
    term4 = (1 / a4) ** 2
    if target_term > term4:
        d4 = 3.0 / np.sqrt(target_term - term4)
        b4 = 0.002  # 2 mm height - very thin
        geometries.append({
            'name': 'TE103 Thin-long',
            'mode': (1, 0, 3),
            'dims': (a4, b4, d4),
            'description': 'Long thin cavity with high longitudinal mode'
        })
    
    # Geometry 5: Extreme miniaturization - TE501
    a5 = 0.006  # 6 mm
    term5 = (5 / a5) ** 2
    if target_term > term5:
        d5 = 1.0 / np.sqrt(target_term - term5)
        b5 = 0.002  # 2 mm
        geometries.append({
            'name': 'TE501 Extreme',
            'mode': (5, 0, 1),
            'dims': (a5, b5, d5),
            'description': 'Extreme miniaturization'
        })
    
    return geometries

def evaluate_all_geometries():
    """Evaluate all candidate geometries for coupling strength."""
    
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
    
    geometries = create_optimized_geometries()
    
    print("DIRECT CAVITY OPTIMIZATION FOR STRONG COUPLING")
    print("=" * 65)
    print(f"Target: g₀ > 150 kHz for strong coupling")
    print(f"Current: g₀ ≈ 99 kHz (weak coupling)")
    print(f"Need: ~2.3× volume reduction")
    print("-" * 65)
    
    results = []
    
    for geom in geometries:
        try:
            cavity_params = {
                "f_c": 13.6713e9, "Q_loaded": 5.0e4, "geometry": "rect",
                "dims": geom['dims'], "eta_pol": 1.0, "grid_N": 36,
                "rect_mode_indices": geom['mode'],
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
            
            # Calculate actual frequency to verify
            m, n, p = geom['mode']
            a, b, d = geom['dims']
            f_actual = (C0/2) * np.sqrt((m/a)**2 + (n/b)**2 + (p/d)**2)
            freq_error = abs(f_actual - 13.6713e9) / 13.6713e9 * 100
            
            result = {
                'name': geom['name'],
                'mode': geom['mode'],
                'dims_mm': [d*1000 for d in geom['dims']],
                'V_eff_mm3': V_eff * 1e9,
                'g0_kHz': g0 / (2 * PI * 1000),
                'cooperativity': cooperativity,
                'strong_coupling': strong_coupling,
                'freq_error_pct': freq_error,
                'cavity_params': cavity_params,
                'description': geom['description']
            }
            
            results.append(result)
            
            status = "✓ STRONG" if strong_coupling else "  weak"
            print(f"{geom['name']:18s} | {status} | g₀={result['g0_kHz']:6.1f} kHz | "
                  f"C={result['cooperativity']:5.2f} | Δf={result['freq_error_pct']:4.1f}%")
                  
        except Exception as e:
            print(f"{geom['name']:18s} | ERROR: {str(e)}")
    
    # Sort by coupling strength
    results.sort(key=lambda x: x['g0_kHz'], reverse=True)
    
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    original_g0 = next((r['g0_kHz'] for r in results if 'Original' in r['name']), 100)
    
    for i, result in enumerate(results):
        improvement = result['g0_kHz'] / original_g0
        
        print(f"\n{i+1}. {result['name']} ({'STRONG COUPLING' if result['strong_coupling'] else 'weak coupling'})")
        print(f"   {result['description']}")
        print(f"   Mode: TE{result['mode'][0]}{result['mode'][1]}{result['mode'][2]}")
        print(f"   Dimensions: {result['dims_mm'][0]:.1f} × {result['dims_mm'][1]:.1f} × {result['dims_mm'][2]:.1f} mm")
        print(f"   g₀: {result['g0_kHz']:.1f} kHz ({improvement:.1f}× improvement)")
        print(f"   Cooperativity: {result['cooperativity']:.2f}")
        print(f"   V_eff: {result['V_eff_mm3']:.1f} mm³")
        print(f"   Frequency error: {result['freq_error_pct']:.2f}%")
    
    # Save best strong coupling result
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
                "I780_Wm2": 200.0, "w780_m": 5.0e-3, "I481_Wm2": 5.0e3, "w481_m": 3.0e-3,
                "Delta_ge_Hz": -15.0e6, "Delta_er_Hz": 0.0, "gamma_ge_phi_Hz": 0.5e6,
                "gamma_er_phi_Hz": 0.2e6, "Gamma_r_Hz": 1.0e4, "rabi_grid_points": 5
            },
            "cw": {"delta_ac_Hz": 0.0, "span_Hz": 40e6, "npts": 1201},
            "description": f"Optimized {best['name']} achieving strong coupling"
        }
        
        with open('strong_coupling_params.json', 'w') as f:
            json.dump(optimized_params, f, indent=2)
        
        print(f"\n" + "🎉" * 25)
        print("SUCCESS: STRONG COUPLING ACHIEVED!")
        print(f"Best geometry: {best['name']}")
        print(f"Coupling: {best['g0_kHz']:.1f} kHz ({best['g0_kHz']/original_g0:.1f}× improvement)")
        print(f"Cooperativity: {best['cooperativity']:.2f}")
        print(f"Saved to: strong_coupling_params.json")
        print("🎉" * 25)
        
        return best
    else:
        print(f"\n⚠️  No strong coupling achieved.")
        print(f"Best result: {max(results, key=lambda x: x['g0_kHz'])['g0_kHz']:.1f} kHz")
        return None

if __name__ == "__main__":
    best = evaluate_all_geometries()
