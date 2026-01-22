import sys
import os
import numpy as np
import torch
import gc
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

def get_predictions(model, dataset, device="cpu", batch_size=32):
    """Generate predictions for the dataset."""
    collate_fn = get_collate_fn(model.input_type)
    # Use smaller batch size for memory efficiency
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0, pin_memory=False)
    
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
                batch_size_current = len(predictions) - len(st_values)
                st_values.extend(np.zeros(batch_size_current))
                bdi_values.extend(np.zeros(batch_size_current))
            
    return np.array(predictions), np.array(ground_truth), np.array(bdi_values), np.array(st_values)

def main():
    parser = argparse.ArgumentParser(description="Generate and archive predictions for all models/folds.")
    parser.add_argument("--folds", type=int, default=100, help="Total number of folds (default: 100 for 10x10 CV)")
    parser.add_argument("--force", action="store_true", help="Force recalculation even if file exists")
    parser.add_argument("--lazy", action="store_true", help="Use lazy loading mode to reduce memory usage")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction (default: 32)")
    parser.add_argument("--data_fraction", type=float, default=0.5, 
                       help="Fraction of dataset to load (0.0 to 1.0). Default 0.5 (half data). Use 1.0 for all data.")
    parser.add_argument("--total_chunks", type=int, default=4,
                       help="Split dataset into N chunks for processing. Use with --chunk_index.")
    parser.add_argument("--chunk_index", type=int, default=0,
                       help="Index of chunk to process (0-based). Requires --total_chunks > 1.")
    args = parser.parse_args()

    # Validate arguments
    if args.data_fraction <= 0 or args.data_fraction > 1:
        print(f"Error: data_fraction must be between 0 and 1, got {args.data_fraction}")
        return
    
    if args.total_chunks < 1:
        print(f"Error: total_chunks must be >= 1, got {args.total_chunks}")
        return
    
    if args.chunk_index < 0 or args.chunk_index >= args.total_chunks:
        print(f"Error: chunk_index must be between 0 and {args.total_chunks-1}, got {args.chunk_index}")
        return
    
    if args.total_chunks > 1 and args.data_fraction < 1.0:
        print("Warning: Both total_chunks > 1 and data_fraction < 1.0 specified. Using chunking mode.")
    
    # Setup paths
    base_output_dir = os.path.join(project_root, "lfs", "predictions")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Choose dataset mode based on lazy flag
    dataset_mode = "lazy" if args.lazy else "classical"
    print(f"Using dataset mode: {dataset_mode}")
    print(f"Using data fraction: {args.data_fraction}")
    
    # Load the 'all' dataset once (contains all components)
    # This is more memory efficient than loading different datasets for each model type
    if args.total_chunks > 1:
        print(f"\n=== Loading 'all' dataset chunk {args.chunk_index+1}/{args.total_chunks} ===")
    else:
        print(f"\n=== Loading 'all' dataset with {args.data_fraction*100:.1f}% of data ===")
    
    try:
        print("Loading dataset with input_type='all'...")
        dataset = get_dataset(
            input_type='all', 
            dataset_mode=dataset_mode, 
            data_fraction=args.data_fraction,
            chunk_index=args.chunk_index,
            total_chunks=args.total_chunks
        )
        print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # Process one model type at a time to minimize memory usage
    for model_type in ALLOWED_INPUT_TYPES:
        print(f"\n=== Processing Model Type: {model_type} ===")
        
        # Create output dir for this model type
        model_output_dir = os.path.join(base_output_dir, model_type)
        os.makedirs(model_output_dir, exist_ok=True)
            
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
            
            # Predict with memory optimization
            try:
                preds, ground_truth, bdi, st = get_predictions(model, dataset, device, batch_size=args.batch_size)
                
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
                
                # Clear cache to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Error processing fold {fold}: {e}")
                continue
        
        # Clear cache between model types (but keep dataset)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Don't delete dataset here - it's used in the loop
    print("\nProcessing complete. Data archived in lfs/predictions/")

if __name__ == "__main__":
    main()
