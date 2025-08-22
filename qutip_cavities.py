



import numpy as np
import matplotlib.pyplot as plt
from qutip import *





GHz = 2 * np.pi


omega_c = 5.0 * GHz      
kappa = 0.05 * GHz       


omega_a = 5.0 * GHz      
gamma = 0.01 * GHz       


g = 0.1 * GHz            





print("--- System Parameters ---")
print(f"Coupling Strength (g/2pi): {g / (2 * np.pi):.3f} GHz")
print(f"Cavity Decay Rate (kappa/2pi): {kappa / (2 * np.pi):.3f} GHz")
print(f"Atomic Decay Rate (gamma/2pi): {gamma / (2 * np.pi):.3f} GHz")

if g > kappa and g > gamma:
    print("\nCondition for Strong Coupling (g > kappa, gamma) is MET.")
else:
    print("\nCondition for Strong Coupling is NOT MET. Splitting may not be visible.")
print("-" * 25)







detuning_range = np.linspace(-3 * g, 3 * g, 200)
energies = []






for delta in detuning_range:
    
    H = 0.5 * delta * sigmaz() + g * sigmax()
    
    
    energies.append(H.eigenenergies())


plt.figure(figsize=(8, 6))
plt.plot(detuning_range / GHz, np.array(energies)[:, 0] / GHz, 'r', linewidth=2)
plt.plot(detuning_range / GHz, np.array(energies)[:, 1] / GHz, 'b', linewidth=2)
plt.title('Energy Spectrum: Avoided Crossing', fontsize=16)
plt.xlabel('Atom-Cavity Detuning (Δ/2π) [GHz]', fontsize=12)
plt.ylabel('Eigenenergies (E/ħ)/2π [GHz]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(0, color='black', linestyle=':', linewidth=1.5, label='Resonance (Δ=0)')
plt.legend(['Dressed State |+⟩', 'Dressed State |−⟩', 'Resonance'])
plt.show()







N = 10 


a = tensor(qeye(2), destroy(N))      
sm = tensor(destroy(2), qeye(N))     
sz = tensor(sigmaz(), qeye(N))       


probe_freq_range = np.linspace(omega_c - 5 * g, omega_c + 5 * g, 400)
cavity_photons = []


c_ops = [
    np.sqrt(kappa) * a,  
    np.sqrt(gamma) * sm  
]

for probe_freq in probe_freq_range:
    
    
    
    H_eff = (omega_c - probe_freq) * a.dag() * a + (omega_a - probe_freq) * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
    
    
    
    
    rho_ss = steadystate(H_eff, c_ops)
    
    
    photon_number = expect(a.dag() * a, rho_ss)
    cavity_photons.append(photon_number)


plt.figure(figsize=(8, 6))

plt.plot((probe_freq_range - omega_c) / GHz, cavity_photons, 'k', linewidth=2)
plt.title('Cavity Transmission Spectrum: Vacuum Rabi Splitting', fontsize=16)
plt.xlabel('Probe Detuning (ω_p - ω_c)/2π [GHz]', fontsize=12)
plt.ylabel('Intracavity Photon Number ⟨n⟩', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)


from scipy.signal import find_peaks
peaks, _ = find_peaks(cavity_photons)
if len(peaks) == 2:
    peak_freqs = (probe_freq_range[peaks] - omega_c) / GHz
    splitting = peak_freqs[1] - peak_freqs[0]
    plt.axvline(peak_freqs[0], color='r', linestyle='--', alpha=0.8)
    plt.axvline(peak_freqs[1], color='b', linestyle='--', alpha=0.8)
    plt.text(0, max(cavity_photons)*0.5, f'Splitting ≈ {splitting:.2f} GHz\n(2g/2π = {2*g/GHz:.2f} GHz)', 
             ha='center', va='center', backgroundcolor='white', fontsize=10)

plt.show()
