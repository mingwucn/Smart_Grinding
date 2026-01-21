#!/usr/bin/env python3
"""
Generate Prediction vs. Ground Truth Plot with Physical Context

This script creates Panel A: A time-series plot showing the model's predicted 
surface roughness (Ra) overlaid on the measured ground truth, with a color-coded 
background indicating the dominant physical regime (BDI > 1: ductile, BDI < 1: brittle).

Reuses existing code from postprocessing/plot_prediction_time_series_with_physics.py
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path to import from postprocessing
sys.path.append(str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(
        description="Generate Prediction vs. Ground Truth plot with physical context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Default: ae_features model, 100 samples
  %(prog)s --model vib_features+pp  # Use vib_features+pp model
  %(prog)s --model all              # Generate plots for all model types
  %(prog)s --samples 50             # Use first 50 samples
  %(prog)s --no-save --show         # Display plot without saving
  %(prog)s --output custom_plot.png # Save with custom filename
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='ae_features',
        help='Model type to use for predictions (default: ae_features). '
             'Options: ae_spec, ae_features, ae_features+pp, vib_spec, vib_features, '
             'vib_features+pp, ae_spec+ae_features, vib_spec+vib_features, '
             'ae_spec+ae_features+vib_spec+vib_features, all'
    )
    
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=100,
        help='Number of samples to plot (default: 100)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename (default: prediction_time_series_{model}.png)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save the plot to file'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='Display the plot (default: True)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_false',
        dest='show',
        help='Do not display the plot'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved plot (default: 300)'
    )
    
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[12, 6],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 12 6)'
    )
    
    args = parser.parse_args()
    
    # Import the plotting function
    try:
        from postprocessing.plot_prediction_time_series_with_physics import (
            create_physics_informed_plot,
            generate_predictions_for_all_models,
            plot_time_series_with_physics,
            load_physics_data
        )
    except ImportError as e:
        print(f"Error importing plotting functions: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    
    print("=" * 70)
    print("Generating Prediction vs. Ground Truth Plot with Physical Context")
    print("=" * 70)
    print(f"Model type: {args.model}")
    print(f"Number of samples: {args.samples}")
    print(f"Save plot: {not args.no_save}")
    print(f"Show plot: {args.show}")
    print()
    
    # Handle 'all' model type
    if args.model.lower() == 'all':
        print("Generating plots for all model types...")
        print("Note: This may take several minutes.")
        print()
        
        # Load physics data for reference
        true_values_global, bdi_values_global, st_values_global = load_physics_data()
        
        # Generate predictions for all models
        predictions_dict = generate_predictions_for_all_models()
        
        if not predictions_dict:
            print("No predictions generated. Exiting.")
            sys.exit(1)
        
        # Create plots for each model type
        for model_type, data in predictions_dict.items():
            print(f"\nCreating plot for {model_type}...")
            
            true_values = data['true_values']
            predictions = data['predictions']
            bdi_values = data['bdi_values']
            
            # Limit to requested number of samples
            if len(true_values) > args.samples:
                true_values = true_values[:args.samples]
                predictions = predictions[:args.samples]
                bdi_values = bdi_values[:args.samples]
            
            # Create time-series plot
            import matplotlib.pyplot as plt
            fig, ax = plot_time_series_with_physics(
                true_values, predictions, bdi_values, model_type
            )
            
            # Adjust figure size if specified
            if args.figsize != [12, 6]:
                fig.set_size_inches(args.figsize[0], args.figsize[1])
            
            # Save the plot if requested
            if not args.no_save:
                if args.output:
                    # For 'all' mode with custom output, append model name
                    base, ext = os.path.splitext(args.output)
                    output_filename = f"{base}_{model_type}{ext}"
                else:
