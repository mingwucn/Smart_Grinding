import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.patches import Patch

# Add parent directory to path to import GrindingData and MyModels
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_trained_model(model_type="ae_features", fold=0):
    """
    Load a trained model from lfs/checkpoints directory.
    """
    from MyModels import GrindingPredictor
    from MyDataset import project_dir
    
    # Construct model path
    model_filename = f"{model_type}_fold{fold}_of_folds10.pt"
    model_path = os.path.join("./lfs/checkpoints", model_filename)
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        available_models = [f for f in os.listdir(os.path.join(project_dir, "lfs", "checkpoints")) 
                           if f.endswith('.pt')]
        print(f"Available models: {available_models[:10]}...")  # Show first 10
        return None
    
    # Initialize model with correct input type
    model = GrindingPredictor(input_type=model_type)
    
    # Load model weights
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model: {model_filename}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    model.eval()
    return model

def load_physics_data_only():
    """
    Load only physics data using GrindingData's _load_all_physics_data method.
    This avoids loading the full dataset with spectrograms and other large data.
    """
    from GrindingData import GrindingData
    from MyDataset import project_dir
    
    # Create GrindingData instance
    grinding_data = GrindingData(project_dir)
    
    # Load only physics data (much more efficient)
    print("Loading physics data only...")
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

def generate_predictions_with_model(model, model_type):
    """
    Generate predictions using the trained model.
    For simplicity, we'll use a subset of the data that matches the model's input type.
    """
    from MyDataset import get_dataset
    
    # Load dataset based on model type
    dataset = get_dataset(input_type=model_type, dataset_mode="classical")
    
    predictions = []
    true_values = []
    bdi_values = []
    
    # Generate predictions for each sample
    print(f"Generating predictions using {model_type} model...")
    with torch.no_grad():
        for i in range(min(100, len(dataset))):  # Use first 100 samples for efficiency
            item = dataset[i]
            
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
    
    return np.array(true_values), np.array(predictions), np.array(bdi_values)

def plot_prediction_overlay_with_model(true_values, predictions, bdi_values, model_type):
    """
    Plot predictions overlaying measurements with background colored by BDI regime.
    Uses actual model predictions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort data by BDI for better visualization
    sort_idx = np.argsort(bdi_values)
    bdi_sorted = bdi_values[sort_idx]
    true_sorted = true_values[sort_idx]
    pred_sorted = predictions[sort_idx]
    
    # Plot measurements and predictions
    ax.plot(bdi_sorted, true_sorted, 'o-', label='Measurement', color='blue', alpha=0.7, markersize=4)
    ax.plot(bdi_sorted, pred_sorted, 's-', label='Prediction', color='red', alpha=0.7, markersize=4)
    
    # Add vertical line at BDI=1
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.8, linewidth=1)
    
    # Color background based on BDI regime
    bdi_min, bdi_max = np.min(bdi_sorted), np.max(bdi_sorted)
    
    # Fill area for BDI < 1
    if bdi_min < 1.0:
        ax.axvspan(bdi_min, 1.0, alpha=0.2, color='lightcoral', label='BDI < 1')
    
    # Fill area for BDI > 1  
    if bdi_max > 1.0:
        ax.axvspan(1.0, bdi_max, alpha=0.2, color='lightgreen', label='BDI > 1')
    
    # Customize plot
    ax.set_xlabel('BDI Value')
    ax.set_ylabel('Surface Roughness (μm)')
    ax.set_title(f'Prediction vs Measurement with BDI Regime Background\nUsing {model_type} Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotation for BDI=1 line
    ax.text(1.02, ax.get_ylim()[1] * 0.95, 'BDI = 1', rotation=90, 
            verticalalignment='top', fontsize=10)
    
    # Calculate and display prediction error
    mae = np.mean(np.abs(true_values - predictions))
    ax.text(0.05, 0.95, f'MAE: {mae:.2f} μm', 
            transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="white", alpha=0.8))
    
    return fig, ax

def main(model_type="ae_features", fold=0):
    """Main function using actual trained model predictions."""
    print(f"=== Prediction Overlay with Trained Model ===")
    print(f"Model type: {model_type}, Fold: {fold}")
    
    # Load trained model
    model = load_trained_model(model_type, fold)
    if model is None:
        print("Failed to load model. Using placeholder data.")
        return
    
    # Generate predictions using the model
    true_values, predictions, bdi_values = generate_predictions_with_model(model, model_type)
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction range: {np.min(predictions):.2f} to {np.max(predictions):.2f} μm")
    print(f"MAE: {np.mean(np.abs(true_values - predictions)):.2f} μm")
    
    # Create plot
    print("Creating plot with model predictions...")
    fig, ax = plot_prediction_overlay_with_model(true_values, predictions, bdi_values, model_type)
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), f'prediction_overlay_{model_type}_fold{fold}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Test with different model types
    model_types = ["ae_features", "vib_features", "ae_features+pp", "vib_features+pp"]
    
    for model_type in model_types:
        try:
            main(model_type=model_type, fold=0)
        except Exception as e:
            print(f"Error with model type {model_type}: {e}")
            continue
