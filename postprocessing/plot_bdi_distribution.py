import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# Add parent directory to path to import GrindingData
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GrindingData import GrindingData
from MyDataset import project_dir

# Set up plotting style
plt.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.dpi'] = 300

def load_physics_data():
    """
    Load physics data including surface roughness and BDI values.
    """
    # Create GrindingData instance
    grinding_data = GrindingData(project_dir)
    
    # Load only physics data (much more efficient)
    print("Loading physics data...")
    grinding_data._load_all_physics_data()
    
    # Extract the data we need
    true_values = grinding_data.sr * 1e3  # Convert to um
    bdi_values = grinding_data.bid
    st_values = grinding_data.st
    
    # Convert to numpy arrays and ensure proper shape
    true_values = np.array(true_values).flatten()
    bdi_values = np.array(bdi_values).flatten()
    st_values = np.array(st_values).flatten()
    
    print(f"Loaded physics data for {len(true_values)} samples")
    print(f"BDI range: {np.min(bdi_values):.3f} to {np.max(bdi_values):.3f}")
    print(f"Surface roughness range: {np.min(true_values):.3f} to {np.max(true_values):.3f} um")
    
    return true_values, bdi_values, st_values

def plot_bdi_distribution(bdi_values):
    """
    Create a distribution plot of BDI values.
    
    Parameters:
    - bdi_values: Array of BDI values
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate statistics
    mean_bdi = np.mean(bdi_values)
    median_bdi = np.median(bdi_values)
    std_bdi = np.std(bdi_values)
    
    # Create histogram
    n, bins, patches = ax.hist(bdi_values, bins=30, alpha=0.7, color='skyblue', 
                              edgecolor='black', linewidth=0.5, density=True)
    
    # Color bars based on BDI regime
    for i in range(len(patches)):
        if bins[i] < 1.0:
            patches[i].set_facecolor('lightcoral')  # Brittle regime
        else:
            patches[i].set_facecolor('lightblue')   # Ductile regime
    
    # Add vertical line at BDI = 1.0 (regime boundary)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, 
               label='BDI = 1.0 (Regime Boundary)')
    
    # Add mean and median lines
    ax.axvline(x=mean_bdi, color='green', linestyle='-', linewidth=2, 
               label=f'Mean BDI = {mean_bdi:.3f}')
    ax.axvline(x=median_bdi, color='orange', linestyle='-', linewidth=2, 
               label=f'Median BDI = {median_bdi:.3f}')
    
    # Customize plot
    ax.set_xlabel('Brittle-Ductile Index (BDI)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Brittle-Ductile Index (BDI) Values\n' + 
                f'(n={len(bdi_values)} samples)')
    
    # Add statistics text box
    stats_text = (f'Statistics:\n'
                 f'Mean: {mean_bdi:.3f}\n'
                 f'Median: {median_bdi:.3f}\n'
                 f'Std: {std_bdi:.3f}\n'
                 f'Min: {np.min(bdi_values):.3f}\n'
                 f'Max: {np.max(bdi_values):.3f}\n'
                 f'Ductile (BDI > 1): {np.sum(bdi_values > 1.0)} samples\n'
                 f'Brittle (BDI < 1): {np.sum(bdi_values < 1.0)} samples')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="white", alpha=0.8))
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def main():
    """Main function to generate BDI distribution plot."""
    print("=== BDI Value Distribution Analysis ===")
    
    # Load physics data
    true_values, bdi_values, st_values = load_physics_data()
    
    # Create BDI distribution plot
    print("Creating BDI distribution plot...")
    fig, ax = plot_bdi_distribution(bdi_values)
    
    # Save the plot
    output_filename = "bdi_distribution.png"
    output_path = os.path.join(os.path.dirname(__file__), output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"BDI distribution plot saved to: {output_path}")
    
    # Show the plot
    # plt.show() # Commented out for automated execution
    plt.close(fig)  # Close figure to free memory
    
    print("=== BDI distribution plot generated successfully ===")

if __name__ == "__main__":
    main()
