"""Visualization tools for cavity QED simulations."""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..models import SimulationResult, CouplingRegime

def plot_spectrum(
    result: SimulationResult,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Cavity Transmission Spectrum",
    **kwargs
) -> Tuple[Figure, Axes]:
    """Plot the cavity transmission spectrum.
    
    Args:
        result: Simulation result containing spectrum data
        ax: Optional matplotlib axes to plot on
        figsize: Figure size (width, height) in inches
        title: Plot title
        **kwargs: Additional keyword arguments passed to plot()
        
    Returns:
        Tuple of (figure, axes) containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    freqs = result.data['frequencies']
    transmission = result.data['transmission']
    
    # Convert to MHz for plotting
    freqs_mhz = freqs / 1e6
    
    # Plot the spectrum
    ax.plot(freqs_mhz, transmission, **kwargs)
    
    # Add labels and title
    ax.set_xlabel('Detuning from Cavity (MHz)')
    ax.set_ylabel('Transmission |S21|²')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add coupling regime info if available
    analysis = result.metadata.get('analysis', {})
    if analysis:
        regime = analysis.get('regime', CouplingRegime.WEAK).value
        cooperativity = analysis.get('cooperativity', 0)
        ax.text(
            0.02, 0.98, 
            f"Regime: {regime.upper()}\nC = {cooperativity:.2f}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    return fig, ax

def plot_mode_profile(
    cavity_params: Dict[str, Any],
    plane: str = 'xy',
    z: float = 0.0,
    n_points: int = 100,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = 'viridis'
) -> Tuple[Figure, Axes]:
    """Plot the cavity mode profile in a given plane.
    
    Args:
        cavity_params: Dictionary of cavity parameters
        plane: Plane to plot ('xy', 'xz', or 'yz')
        z: z-coordinate for the plane (if applicable)
        n_points: Number of points in each dimension
        ax: Optional matplotlib axes to plot on
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use
        
    Returns:
        Tuple of (figure, axes) containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get cavity dimensions
    a, b, d = cavity_params['dimensions']
    m, n, p = cavity_params.get('mode_indices', (2, 0, 1))
    
    # Create coordinate grid
    if plane == 'xy':
        x = np.linspace(0, a, n_points)
        y = np.linspace(0, b, n_points)
        X, Y = np.meshgrid(x, y)
        Z = z * np.ones_like(X)
        xlabel, ylabel = 'x (m)', 'y (m)'
    elif plane == 'xz':
        x = np.linspace(0, a, n_points)
        z = np.linspace(0, d, n_points)
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)
        xlabel, ylabel = 'x (m)', 'z (m)'
    elif plane == 'yz':
        y = np.linspace(0, b, n_points)
        z = np.linspace(0, d, n_points)
        Y, Z = np.meshgrid(y, z)
        X = np.zeros_like(Y)
        xlabel, ylabel = 'y (m)', 'z (m)'
    else:
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'")
    
    # Calculate mode pattern (simplified)
    kx = m * np.pi / a
    ky = n * np.pi / b
    kz = p * np.pi / d
    
    E = np.sin(kx * X) * np.cos(ky * Y) * np.sin(kz * Z)
    
    # Plot
    im = ax.pcolormesh(X, Y, E, cmap=cmap, shading='auto')
    plt.colorbar(im, ax=ax, label='Normalized E-field')
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'TE{m}{n}{p} Mode Profile ({plane.upper()} Plane)')
    
    return fig, ax

def plot_coupling_analysis(
    analysis: Dict[str, Any],
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 5)
) -> Tuple[Figure, Axes]:
    """Plot a bar chart comparing key rates in the system.
    
    Args:
        analysis: Dictionary containing coupling analysis results
        ax: Optional matplotlib axes to plot on
        figsize: Figure size (width, height) in inches
        
    Returns:
        Tuple of (figure, axes) containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract rates and convert to MHz
    rates = {
        'g/2π': analysis.get('g', 0) / (2 * np.pi * 1e6),  # MHz
        'κ/2π': analysis.get('kappa', 0) / (2 * np.pi * 1e6),  # MHz
        'γ/2π': analysis.get('gamma', 0) / (2 * np.pi * 1e6)  # MHz
    }
    
    # Create bar plot
    x = list(rates.keys())
    y = list(rates.values())
    
    bars = ax.bar(x, y, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.1f} MHz',
            ha='center',
            va='bottom'
        )
    
    # Add labels and title
    ax.set_ylabel('Rate (MHz)')
    ax.set_title('Coupling Rate Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Add coupling regime info
    regime = analysis.get('regime', CouplingRegime.WEAK).upper()
    cooperativity = analysis.get('cooperativity', 0)
    ax.text(
        0.02, 0.98,
        f"Regime: {regime}\nC = {cooperativity:.2f}",
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    return fig, ax
