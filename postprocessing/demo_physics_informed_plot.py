#!/usr/bin/env python3
"""
Demo script showing how to use the create_physics_informed_plot function.
This demonstrates the standalone function that creates time-series plots with
physical context (BDI regime coloring).
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the standalone function
from plot_prediction_time_series_with_physics import create_physics_informed_plot

def main():
    """Demo the standalone physics-informed plot function."""
    print("=== Demo: Physics-Informed Time Series Plot ===")
    print("This script demonstrates the standalone function that creates:")
    print("- Time-series plot of predicted vs ground truth surface roughness")
    print("- Color-coded background showing BDI regimes (blue=ductile, red=brittle)")
    print("- MAE calculation and insightful caption")
    print()
    
    # Example 1: Create plot for ae_features model
    print("Creating plot for 'ae_features' model...")
    fig, ax = create_physics_informed_plot(
        model_type="ae_features",
        save_plot=True,
        show_plot=False  # Set to True to display the plot
    )
    
    if fig is not None:
        print("✓ Plot created successfully!")
        print("✓ Plot saved as 'prediction_time_series_ae_features.png'")
    else:
        print("✗ Failed to create plot")
    
    print()
    print("=== Usage Examples ===")
    print("You can call the function with different parameters:")
    print()
    print("1. Basic usage:")
    print("   create_physics_informed_plot(model_type='ae_features')")
    print()
    print("2. For different model types:")
    print("   create_physics_informed_plot(model_type='vib_features')")
    print("   create_physics_informed_plot(model_type='ae_features+pp')")
    print()
    print("3. Without saving:")
    print("   create_physics_informed_plot(save_plot=False, show_plot=True)")
    print()
    print("4. Just get the figure objects:")
    print("   fig, ax = create_physics_informed_plot()")
    print("   # Then customize the plot as needed")
    print()
    print("The function returns matplotlib figure and axes objects for further customization.")

if __name__ == "__main__":
    main()
