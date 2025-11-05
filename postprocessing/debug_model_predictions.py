#!/usr/bin/env python3
"""
Debugged standalone function for model loading and predictions.
This fixes the issues in post_processing_bid.ipynb and creates a working standalone function.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl

# Add parent directory to path - exactly like the notebook
sys.path.append('../.')

# Import exactly what's used in the notebook
from GrindingData import GrindingData
from MyDataset import project_dir, allowed_input_types
from MyCustomModels import MyCustomGrindingPredictor as GrindingPredictor

# Set up plotting style
plt.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.dpi'] = 300

def load_physics_data():
    """
    Load physics data including surface roughness and BDI values.
    Exactly the same as in post_processing_bid.ipynb
    """
    # Create GrindingData instance
    grinding_data = GrindingData(project_dir)
    
    # Load only physics data (much more efficient)
    print("Loading physics data...")
    grinding_data._load_all_physics_data()
    
    # Extract the data we need
    true_values = grinding_data.sr * 1e3  # Convert to μm
    bdi_values = grinding_data.bid
    st_values = grinding_data.st
    
    # Convert to numpy arrays and ensure proper shape
    true_values = np.array(true_values).flatten()
    bdi_values = np.array(bdi_values).flatten()
    st_values = np.array(st_values).flatten()
    
    print(f"Loaded physics data for {len(true_values)} samples")
    print(f"BDI range: {np.min(bdi_values):.3f} to {np.max(bdi_values):.3f}")
    print(f"Surface roughness range: {np.min(true_values):.3f} to {np.max(true_values):.3f} μm")
    
    return true_values, bdi_values, st_values

def debug_load_trained_model(model_type="ae_features", fold=0):
    """
    Debugged version of load_trained_model that handles architecture mismatches.
    """
    # Use MyCustomModels instead of MyModels (fixing the import issue)
    from MyCustomModels import MyCustomGrindingPredictor as GrindingPredictor
    
    # Construct model path relative to current working directory
    model_filename = f"{model_type}_fold{fold}_of_folds10.pt"
    model_path = os.path.join("../lfs/checkpoints", model_filename)
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        # Check what models are available
        checkpoints_dir = "../lfs/checkpoints"
        if os.path.exists(checkpoints_dir):
            available_models = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
            print(f"Available models: {available_models[:10]}...")  # Show first 10
        else:
            print(f"Checkpoints directory not found: {checkpoints_dir}")
        return None
    
    # Initialize model with correct input type
    model = GrindingPredictor(input_type=model_type)
    
    # Load model weights with better error handling
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Try different loading strategies
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try direct loading
            model.load_state_dict(checkpoint)
        
        print(f"✓ Successfully loaded model: {model_filename}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Trying partial loading...")
        
        # Try partial loading for architecture mismatches
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state' in checkpoint:
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['model_state'].items() 
                                 if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(f"✓ Partially loaded model: {len(pretrained_dict)}/{len(model_dict)} layers")
            else:
                print("✗ No compatible state dict found")
                return None
        except Exception as e2:
            print(f"✗ Partial loading also failed: {e2}")
            return None
    
    model.eval()
    return model

def debug_generate_predictions_for_model(model_type="ae_features", num_samples=50):
    """
    Debugged version that generates predictions for a single model type.
    """
    from MyDataset import get_dataset
    
    print(f"\n=== Debug: Generating predictions for {model_type} ===")
    
    # Load model
    model = debug_load_trained_model(model_type, fold=0)
    if model is None:
        print(f"✗ Failed to load model for {model_type}")
        return None
    
    # Load dataset based on model type
    try:
        dataset = get_dataset(input_type=model_type, dataset_mode="classical")
        print(f"✓ Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Error loading dataset for {model_type}: {e}")
        return None
    
    predictions = []
    true_values = []
    bdi_values = []
    
    # Generate predictions for each sample
    print(f"Generating predictions using {model_type} model...")
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):  # Use fewer samples for debugging
            try:
                item = dataset[i]
                
                # Debug: Print item structure
                if i == 0:
                    print(f"Sample 0 keys: {list(item.keys())}")
                    if 'features_pp' in item:
                        print(f"features_pp shape: {item['features_pp'].shape}")
                
                # Prepare input batch
                batch = {}
                for key in ['spec_ae', 'spec_vib', 'features_ae', 'features_vib', 'features_pp']:
                    if key in item:
                        batch[key] = item[key].unsqueeze(0)  # Add batch dimension
                
                # Add lengths if available
                if 'features_ae' in item:
                    batch['ae_lengths'] = torch.tensor([item['features_ae'].shape[0]])
                if 'features_vib' in item:
                    batch['vib_lengths'] = torch.tensor([item['features_vib'].shape[0]])
                
                # Generate prediction
                prediction = model(batch)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]  # Handle case where model returns (prediction, attention)
                
                predictions.append(prediction.item())
                true_values.append(item['label'].item())
                
                # Extract BDI (3rd element in features_pp: [ec, st, bid])
                if 'features_pp' in item:
                    bdi_values.append(item['features_pp'][2].item())
                else:
                    # If no features_pp, use default BDI value
                    bdi_values.append(1.0)
                    
            except Exception as e:
                print(f"✗ Error processing sample {i} for {model_type}: {e}")
                continue
    
    if len(predictions) > 0:
        result = {
            'true_values': np.array(true_values),
            'predictions': np.array(predictions),
            'bdi_values': np.array(bdi_values)
        }
        mae = np.mean(np.abs(result['true_values'] - result['predictions']))
        print(f"✓ Generated {len(predictions)} predictions for {model_type}")
        print(f"✓ MAE: {mae:.3f} μm")
        return result
    else:
        print(f"✗ No predictions generated for {model_type}")
        return None

def debug_generate_predictions_for_all_models():
    """
    Debugged version that generates predictions for all model types.
    """
    predictions_dict = {}
    
    # Test with a subset of model types first
    test_model_types = ['ae_features', 'vib_features']  # Start with simpler models
    
    for model_type in test_model_types:
        result = debug_generate_predictions_for_model(model_type, num_samples=20)
        if result is not None:
            predictions_dict[model_type] = result
    
    return predictions_dict

def create_debug_plot(predictions_data, model_type):
    """
    Create a debug plot showing predictions vs ground truth.
    """
    if predictions_data is None:
        print("No data to plot")
        return None
    
    true_values = predictions_data['true_values']
    predictions = predictions_data['predictions']
    bdi_values = predictions_data['bdi_values']
    
    sample_indices = np.arange(len(true_values))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate MAE
    mae = np.mean(np.abs(true_values - predictions))
    
    # Plot ground truth and predictions
    ax.plot(sample_indices, true_values, 'o-', label='Ground Truth', 
            color='black', alpha=0.8, markersize=4, linewidth=1.5)
    ax.plot(sample_indices, predictions, 's-', label='Prediction', 
            color='red', alpha=0.8, markersize=4, linewidth=1.5)
    
    # Color background based on BDI regime
    bdi_regime = bdi_values > 1.0  # True for ductile, False for brittle
    
    # Group consecutive samples with same regime
    regime_changes = np.where(np.diff(bdi_regime.astype(int)) != 0)[0] + 1
    regime_starts = np.concatenate(([0], regime_changes))
    regime_ends = np.concatenate((regime_changes, [len(bdi_regime)]))
    
    for start, end in zip(regime_starts, regime_ends):
        regime = bdi_regime[start]
        color = 'lightblue' if regime else 'lightcoral'
        alpha = 0.3 if regime else 0.2
        
        # Use integer indices for sample positions
        x_start = sample_indices[max(0, start)]
        x_end = sample_indices[min(len(sample_indices)-1, end)]
        
        ax.axvspan(x_start, x_end, ymin=0, ymax=1, alpha=alpha, color=color)
    
    # Customize plot
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Surface Roughness Ra (μm)')
    ax.set_title(f'Debug: Prediction vs Ground Truth\nModel: {model_type}')
    
    # Create legend with regime information
    legend_elements = [
        Line2D([0], [0], color='black', marker='o', linestyle='-', label='Ground Truth'),
        Line2D([0], [0], color='red', marker='s', linestyle='-', label='Prediction'),
        Patch(facecolor='lightblue', alpha=0.3, label='BDI > 1 (Ductile)'),
        Patch(facecolor='lightcoral', alpha=0.2, label='BDI < 1 (Brittle)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    # Add MAE annotation
    ax.text(0.02, 0.98, f'MAE = {mae:.2f} μm', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top', usetex=False)
    
    plt.tight_layout()
    return fig, ax

def main():
    """Main function to test the debugged model loading and predictions."""
    print("=== Debug: Model Loading and Predictions ===")
    print("This script debugs the model loading and prediction generation from post_processing_bid.ipynb")
    print()
    
    # Test 1: Load physics data
    print("1. Testing physics data loading...")
    try:
        true_values, bdi_values, st_values = load_physics_data()
        print("✓ Physics data loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load physics data: {e}")
        return
    
    # Test 2: Generate predictions for models
    print("\n2. Testing model loading and predictions...")
    predictions_dict = debug_generate_predictions_for_all_models()
    
    if predictions_dict and len(predictions_dict) > 0:
        print(f"\n✓ Successfully generated predictions for {len(predictions_dict)} models")
        
        # Test 3: Create debug plots
        print("\n3. Creating debug plots...")
        for model_type, data in predictions_dict.items():
            fig, ax = create_debug_plot(data, model_type)
            if fig is not None:
                output_filename = f"debug_prediction_{model_type}.png"
                output_path = os.path.join(os.path.dirname(__file__), output_filename)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved debug plot: {output_path}")
                plt.close(fig)
    else:
        print("✗ No predictions generated for any models")
        print("\nTroubleshooting steps:")
        print("1. Check if model files exist in lfs/checkpoints/")
        print("2. Verify model architecture compatibility")
        print("3. Check dataset loading")

if __name__ == "__main__":
    main()
