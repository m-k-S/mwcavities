"""Command-line interface for cavity QED simulations."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

from .core.physics import CavityQEDCalculator
from .config import ConfigManager, get_default_config
from .visualization.plotting import plot_spectrum, plot_mode_profile, plot_coupling_analysis

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Cavity QED Simulation Tool')
    
    # Input/output arguments
    parser.add_argument('-c', '--config', type=str, help='Path to configuration file')
    parser.add_argument('-o', '--output', type=str, default='results', 
                       help='Output directory for results')
    
    # Simulation parameters
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ConfigManager.load_config(args.config)
    else:
        print("Using default configuration")
        config = get_default_config()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run simulation
    try:
        # Initialize calculator with parameters
        params = ConfigManager.from_dict(config)
        calculator = CavityQEDCalculator(params['cavity'], params['rydberg'])
        
        # Run CW simulation
        detuning_range = params['cw'].get('detuning_range', (-20e6, 20e6))  # Hz
        num_points = params['cw'].get('num_points', 1001)
        
        print(f"Running CW simulation from {detuning_range[0]/1e6:.1f} MHz to {detuning_range[1]/1e6:.1f} MHz...")
        result = calculator.simulate_cw_response(detuning_range, num_points)
        
        # Save results if requested
        if args.save:
            result_file = output_dir / 'simulation_results.json'
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Results saved to {result_file}")
        
        # Generate plots if requested
        if args.plot:
            print("Generating plots...")
            
            # Create plots directory
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Plot spectrum
            fig, ax = plot_spectrum(result)
            fig.savefig(plots_dir / 'transmission_spectrum.png', dpi=300, bbox_inches='tight')
            
            # Plot mode profile
            fig, _ = plot_mode_profile(asdict(params['cavity']))
            fig.savefig(plots_dir / 'mode_profile.png', dpi=300, bbox_inches='tight')
            
            # Plot coupling analysis
            if 'analysis' in result.metadata:
                fig, _ = plot_coupling_analysis(result.metadata['analysis'])
                fig.savefig(plots_dir / 'coupling_analysis.png', dpi=300, bbox_inches='tight')
            
            plt.close('all')
            print(f"Plots saved to {plots_dir}")
        
        print("Simulation completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
