import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

# Add project root and utils to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "utils"))

from MyModels import GrindingPredictor
from MyDataset import get_dataset, get_collate_fn

# Model types
ALLOWED_INPUT_TYPES = [
    'ae_spec',
    'ae_features',
    'ae_features+pp',
    'ae_spec+ae_features',
    'vib_spec',
    'vib_features',
    'vib_features+pp',
    'vib_spec+vib_features',
    'ae_features+vib_features',
    'ae_features+vib_features+pp',
    'ae_spec+vib_spec',
    'ae_spec+ae_features+vib_spec+vib_features',
    'all',
]

def load_model(model_type, fold, device="cpu"):
    """Load the trained model."""
    checkpoint_dirs = [
        os.path.join(project_root, "lfs", "checkpoints"),
        os.path.join(project_root, "checkpoints")
    ]
    
    model_path = None
    for d in checkpoint_dirs:
        path = os.path.join(d, f"{model_type}_fold{fold}_of_folds10.pt")
        if os.path.exists(path):
            model_path = path
            break
            
    if model_path is None:
        return None
    
    try:
        model = GrindingPredictor(input_type=model_type)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model {model_type} fold {fold}: {e}")
        return None

def get_predictions(model, dataset, device="cpu"):
    """Generate predictions for the dataset."""
    collate_fn = get_collate_fn(model.input_type)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    predictions = []
    ground_truth = []
    bdi_values = []
    st_values = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                else:
                    batch_on_device[k] = v
            
            pred = model(batch_on_device)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            predictions.extend(pred.cpu().flatten().numpy())
            ground_truth.extend(batch['label'].flatten().numpy())
            
            # Extract PP (assuming they are available in batch)
            # Check if features_pp exists and handle accordingly
            if 'features_pp' in batch:
                pp = batch['features_pp']
                if isinstance(pp, torch.Tensor):
                    pp = pp.cpu().numpy()
                st_values.extend(pp[:, 1])
                bdi_values.extend(pp[:, 2])
            else:
                # Fallback if no pp
                batch_size = len(predictions) - len(st_values)
                st_values.extend(np.zeros(batch_size))
                bdi_values.extend(np.zeros(batch_size))
            
    return np.array(predictions), np.array(ground_truth), np.array(bdi_values), np.array(st_values)

def main():
    parser = argparse.ArgumentParser(description="Generate and archive predictions for all models/folds.")
    parser.add_argument("--folds", type=int, default=100, help="Total number of folds (default: 100 for 10x10 CV)")
    parser.add_argument("--force", action="store_true", help="Force recalculation even if file exists")
    args = parser.parse_args()

    # Setup paths
    base_output_dir = os.path.join(project_root, "lfs", "predictions")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cache dataset to avoid reloading for every fold if input_type is same?
    # Actually, dataset depends on input_type. But 'all' dataset covers everything if loaded once?
    # MyDataset loads based on input_type. Optimization: Load 'all' dataset once and subset?
    # No, get_dataset returns a Dataset object that might have specific transforms.
    # Safe bet is to load fresh for each model type, but keep it in memory if iterating folds for SAME model type.
    # Actually, MyDataset loads ALL data into memory usually (unless 'chunked'/'ram'). 
    # Let's instantiate dataset once per model_type.
    
    for model_type in ALLOWED_INPUT_TYPES:
        print(f"\n=== Processing Model Type: {model_type} ===")
        
        # Create output dir for this model type
        model_output_dir = os.path.join(base_output_dir, model_type)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Load dataset once for this model type
        try:
            print(f"Loading dataset for {model_type}...")
            dataset = get_dataset(input_type=model_type, dataset_mode="classical")
        except Exception as e:
            print(f"Failed to load dataset for {model_type}: {e}")
            continue
            
        # Iterate folds
        for fold in tqdm(range(args.folds), desc=f"Folds ({model_type})"):
            output_file = os.path.join(model_output_dir, f"fold_{fold}.npz")
            
            if os.path.exists(output_file) and not args.force:
                # print(f"Skipping fold {fold} (already exists)")
                continue
            
            # Load model
            model = load_model(model_type, fold, device)
            if model is None:
                # print(f"Checkpoint not found for fold {fold}")
                continue
            
            # Predict
            try:
                preds, ground_truth, bdi, st = get_predictions(model, dataset, device)
                
                # Save compressed
                np.savez_compressed(
                    output_file,
                    predictions=preds,
                    ground_truth=ground_truth,
                    bdi=bdi,
                    st=st,
                    model_type=model_type,
                    fold=fold
                )
            except Exception as e:
                print(f"Error processing fold {fold}: {e}")
                continue
                
    print("\nProcessing complete. Data archived in lfs/predictions/")

if __name__ == "__main__":
    main()
