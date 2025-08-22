import numpy as np
import json
from constants import PI, EPS0, HBAR, C0
from cavity_modes import make_mode_u, numerical_Veff_from_u
from data_structures import CavityParams

def calculate_cavity_dimensions_for_frequency(f_target, mode_indices=(2,0,1)):
    """Calculate cavity dimensions for a target frequency and mode."""
    m, n, p = mode_indices
    lambda_mw = C0 / f_target
    
    # For TE_mnp mode: f = (c/2) * sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)
    # We want to minimize volume while hitting target frequency
    
    # Strategy: Use higher-order modes to reduce one dimension significantly
    # This reduces the mode volume more than it increases other dimensions
    
    geometries = []
    
    # Original TE201 geometry for comparison
    a_orig = 0.016447
    b_orig = 0.008223  
    d_orig = 0.029420
    V_orig = a_orig * b_orig * d_orig
    geometries.append({
        'mode': (2,0,1), 'dims': (a_orig, b_orig, d_orig), 
        'volume': V_orig, 'name': 'Original TE201'
    })
    
    # TE301: Higher m mode - reduces width significantly
    if m >= 3:
        # f = (c/2) * sqrt((3/a)^2 + (0/b)^2 + (1/d)^2)
        # Choose a smaller width, adjust length accordingly
        a_new = 0.75 * lambda_mw / 3  # Smaller width
        b_new = 0.5 * lambda_mw       # Standard height
        # Solve for d: (3/a)^2 + (1/d)^2 = (2f/c)^2
        term = (2*f_target/C0)**2 - (3/a_new)**2
        if term > 0:
            d_new = 1.0 / np.sqrt(term)
            V_new = a_new * b_new * d_new
            geometries.append({
                'mode': (3,0,1), 'dims': (a_new, b_new, d_new),
                'volume': V_new, 'name': 'Optimized TE301'
            })
    
    # TE401: Even higher m mode
    if m >= 4:
        a_new = 0.6 * lambda_mw / 4
        b_new = 0.4 * lambda_mw
        term = (2*f_target/C0)**2 - (4/a_new)**2
        if term > 0:
            d_new = 1.0 / np.sqrt(term)
            V_new = a_new * b_new * d_new
            geometries.append({
                'mode': (4,0,1), 'dims': (a_new, b_new, d_new),
                'volume': V_new, 'name': 'Optimized TE401'
            })
    
    # TE211: Mixed mode with smaller overall volume
    a_new = 0.8 * lambda_mw / 2
    b_new = 0.6 * lambda_mw / 1
    # (2/a)^2 + (1/b)^2 + (1/d)^2 = (2f/c)^2
    term = (2*f_target/C0)**2 - (2/a_new)**2 - (1/b_new)**2
    if term > 0:
        d_new = 1.0 / np.sqrt(term)
        V_new = a_new * b_new * d_new
        geometries.append({
            'mode': (2,1,1), 'dims': (a_new, b_new, d_new),
            'volume': V_new, 'name': 'Optimized TE211'
        })
    
    # TE102: Different orientation
    a_new = 0.9 * lambda_mw / 1
    b_new = 0.3 * lambda_mw
    # (1/a)^2 + (0/b)^2 + (2/d)^2 = (2f/c)^2
    term = (2*f_target/C0)**2 - (1/a_new)**2
    if term > 0:
        d_new = 2.0 / np.sqrt(term)
        V_new = a_new * b_new * d_new
        geometries.append({
            'mode': (1,0,2), 'dims': (a_new, b_new, d_new),
            'volume': V_new, 'name': 'Optimized TE102'
        })
    
    return sorted(geometries, key=lambda x: x['volume'])

def evaluate_coupling_strength(cavity_params, verbose=True):
    """Evaluate the coupling strength for given cavity parameters."""
    cav = CavityParams(**cavity_params)
    
    # Calculate mode volume
    u_func, volume, bounds = make_mode_u(cav)
    V_eff = numerical_Veff_from_u(u_func, volume, bounds, cav.grid_N)
    
    # Calculate coupling parameters
    omega_c = 2 * PI * cav.f_c
    E_zpf = np.sqrt(HBAR * omega_c / (2 * EPS0 * V_eff)) * cav.eta_pol
    
    # Estimate dipole moment (using typical Rydberg values)
    mu_typical = 1000 * 1.602e-19 * 5.29e-11  # ~1000 ea0 in SI units
    g0 = mu_typical * E_zpf / HBAR
    
    # System parameters
    kappa = omega_c / cav.Q_loaded
    gamma = 0.2e6 * 2 * PI  # Typical dephasing
    
    # Coupling analysis
    cooperativity = 4 * g0**2 / (kappa * gamma)
    strong_coupling = g0 > (kappa + gamma) / 4
    
    results = {
        'V_eff_mm3': V_eff * 1e9,  # Convert to mm³
        'g0_MHz': g0 / (2 * PI * 1e6),
        'kappa_MHz': kappa / (2 * PI * 1e6),
        'gamma_MHz': gamma / (2 * PI * 1e6),
        'cooperativity': cooperativity,
        'strong_coupling': strong_coupling,
        'regime': 'Strong' if strong_coupling else 'Weak'
    }
    
    if verbose:
        print(f"Mode: TE{cav.rect_mode_indices[0]}{cav.rect_mode_indices[1]}{cav.rect_mode_indices[2]}")
        print(f"Dimensions: {cav.dims[0]*1000:.2f} × {cav.dims[1]*1000:.2f} × {cav.dims[2]*1000:.2f} mm")
        print(f"V_eff: {results['V_eff_mm3']:.3f} mm³")
        print(f"g₀: {results['g0_MHz']:.1f} MHz")
        print(f"Cooperativity: {results['cooperativity']:.2f}")
        print(f"Regime: {results['regime']}")
        print("-" * 40)
    
    return results

def find_optimal_cavity():
    """Find the optimal cavity geometry for strong coupling."""
    f_target = 13.6713e9  # Target frequency
    
    print("Evaluating cavity geometries for strong coupling...")
    print("=" * 60)
    
    # Get candidate geometries
    candidates = calculate_cavity_dimensions_for_frequency(f_target)
    
    best_candidates = []
    
    for geom in candidates:
        print(f"\n{geom['name']}:")
        
        cavity_params = {
            "f_c": f_target,
            "Q_loaded": 5.0e4,
            "geometry": "rect",
            "dims": geom['dims'],
            "eta_pol": 1.0,
            "grid_N": 36,
            "rect_mode_indices": geom['mode'],
            "kappa_port1_frac": 0.25,
            "kappa_port2_frac": 0.25
        }
        
        try:
            results = evaluate_coupling_strength(cavity_params)
            
            candidate = {
                'name': geom['name'],
                'cavity_params': cavity_params,
                'results': results,
                'improvement_factor': results['g0_MHz'] / 65.827  # Compared to original
            }
            best_candidates.append(candidate)
            
        except Exception as e:
            print(f"Error evaluating {geom['name']}: {e}")
    
    # Sort by coupling strength
    best_candidates.sort(key=lambda x: x['results']['g0_MHz'], reverse=True)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    for i, candidate in enumerate(best_candidates):
        print(f"\n{i+1}. {candidate['name']}")
        print(f"   g₀: {candidate['results']['g0_MHz']:.1f} MHz ({candidate['improvement_factor']:.1f}× improvement)")
        print(f"   Regime: {candidate['results']['regime']}")
        print(f"   Cooperativity: {candidate['results']['cooperativity']:.2f}")
    
    return best_candidates

if __name__ == "__main__":
    best_candidates = find_optimal_cavity()
    
    # Save the best candidate
    if best_candidates and best_candidates[0]['results']['strong_coupling']:
        best = best_candidates[0]
        
        # Create optimized parameters
        optimized_params = {
            "cavity": best['cavity_params'],
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
            "description": f"Optimized {best['name']} for strong coupling - {best['improvement_factor']:.1f}× improvement"
        }
        
        with open('optimized_params.json', 'w') as f:
            json.dump(optimized_params, f, indent=2)
        
        print(f"\nBest geometry saved to 'optimized_params.json'")
        print(f"Expected improvement: {best['improvement_factor']:.1f}× stronger coupling")
