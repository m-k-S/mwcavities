

import matplotlib.pyplot as plt
from constants import PI


def plot_rates_comparison(results_dict):
    
    plt.figure()
    plt.bar(["G (collective)", "kappa (cavity)", "gamma (atomic)"],
            [results_dict["G_Hz"], results_dict["kappa_Hz"], results_dict["gamma_Hz"]])
    plt.ylabel("Rate (Hz)")
    plt.title("Key rates comparison")
    plt.tight_layout()
    plt.show()


def plot_tc_dynamics(time, P_g1, P_e0, P_g0):
    
    plt.figure()
    plt.plot(time, P_g1, label="P(|g,1>)")
    plt.plot(time, P_e0, label="P(|e,0>)")
    plt.plot(time, P_g0, label="P(|g,0>)")
    plt.xlabel("Time (s)")
    plt.ylabel("Population")
    plt.title("Tavis–Cummings dynamics (single excitation)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def make_plots(results_dict, time, P_g1, P_e0, P_g0):
    
    plot_rates_comparison(results_dict)
    plot_tc_dynamics(time, P_g1, P_e0, P_g0)
