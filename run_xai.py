import os
import torch
import argparse
import time
import sys
import gc
import matplotlib.pyplot as plt
from MyModels import GrindingPredictor
from utils.XAI import GradCAM, IntegratedGradients, get_conv_layer_names
from XAI_ModelWrapper import XAI_ModelWrapper
from MyDataset import get_dataset, get_collate_fn

# Configuration
REPORT_PATH = "report"
SNAPSHOT_DIR = "snapshots"
LFS_PATH = "/mnt/c/Users/Public/Documents/IntermediateData/SmartGrinding"

def resolve_model_path(input_type):
    """Get model path for given input combination"""
    # First check local snapshots directory
    local_path = os.path.join(SNAPSHOT_DIR, f"Res15_classification_input_{input_type}_output_regime_roi_time10_roi_radius3_fold0_of_folds10.pt")
    if os.path.exists(local_path):
        return local_path
    
    # Check LFS checkpoints directory
    checkpoints_dir = os.path.join(LFS_PATH, "checkpoints")
    if os.path.exists(checkpoints_dir):
        # Find the first model file matching the input_type pattern
        for filename in os.listdir(checkpoints_dir):
            if filename.startswith(f"{input_type}_") and filename.endswith(".pt"):
                return os.path.join(checkpoints_dir, filename)
    
    return None

def run_xai_analysis(args):
    """Run XAI analysis with the specified parameters"""
    # Load dataset with specified parameters
    cpus = [32, 16, 2, 1]
    percentage = [0.6, 0.8, 0.90, 1]
    _time = time.time()
    
    # Map dataset_mode to valid options
    valid_modes = ['classical', 'chunked', 'ram']
    dataset_mode = args.dataset_mode if args.dataset_mode in valid_modes else 'classical'
    
    dataset = get_dataset(input_type=args.input_type, dataset_mode=dataset_mode, cpus=cpus, percentage=percentage)
    collate_fn = get_collate_fn(input_type=args.input_type)
    gc.collect()
    print(f"Dataset loaded in {time.time()-_time:.2f} seconds")
    print(f"Dataset size: {sys.getsizeof(dataset)}")
    print("*" * 50)

    # Load model
    model_name = args.model_name
    base_model = GrindingPredictor(interp=False, input_type=args.input_type)
    
    # Load model weights if specified
    if model_name:
        model_path = resolve_model_path(args.input_type)
        if not model_path or not os.path.exists(model_path):
            print(f"Model not found for input type: {args.input_type}")
            return
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Try to load state dict with flexible key matching
        state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load state dict with strict=False to ignore mismatched keys
        base_model.load_state_dict(state_dict, strict=False)
        
        print("Model weights partially loaded (some layers may not match)")
    
    base_model.eval()
    model = XAI_ModelWrapper(base_model, args.input_type)
    # Get convolutional layers and select the correct one
    conv_layers = get_conv_layer_names(model)
    print(f"Found convolutional layers: {conv_layers}")
    if not conv_layers:
        print("No convolutional layers found.")
        return

    # Choose the first convolutional layer for Grad-CAM
    target_layer = conv_layers[0]
    print(f"Using target layer: {target_layer}")
    
    model.eval()
    
    from utils.xai_manager import XAIManager
    
    # Create XAI manager
    xai_manager = XAIManager(model, target_layer=target_layer)
    
    # Only run the requested XAI method
    methods_to_run = [args.xai_method]
    
    # Determine which spectrogram key to use
    if "ae" in args.input_type:
        spec_key = 'spec_ae'
    elif "vib" in args.input_type:
        spec_key = 'spec_vib'
    else:
        spec_key = 'spec_ae'  # default

    # Get sample indices to analyze
    if args.sample_indices:
        try:
            sample_indices = [int(idx.strip()) for idx in args.sample_indices.split(',')]
        except ValueError:
            print(f"Invalid sample indices: {args.sample_indices}. Using default [0]")
            sample_indices = [0]
    else:
        sample_indices = [0]  # Default to first sample
    print(f"Analyzing samples at indices: {sample_indices}")
    
    # Loop through selected samples
    for i, idx in enumerate(sample_indices):
        sample = dataset[idx]
        print(f"Analyzing sample {idx}:")
        print(f"Sample keys: {list(sample.keys())}")
        
        # Create a complete batch dictionary
        batch = {}
        # Always include features_pp with zeros as placeholder
        batch['features_pp'] = torch.zeros(1, 3)
        
        # Handle AE inputs
        if "ae" in args.input_type:
            # Extract the spectrogram and convert to float32
            spec_ae = sample['spec_ae'].float()
            print(f"Original spec_ae shape: {spec_ae.shape}")
            
            # Reduce dimensions if necessary
            if spec_ae.dim() > 4:
                spec_ae = spec_ae[0]
                print(f"After reducing dimensions: {spec_ae.shape}")
                
            # Take first time step
            if spec_ae.dim() == 4:
                spec_ae = spec_ae[0]  # Take first time step
                print(f"After taking first time step: {spec_ae.shape}")
                
            # Add batch and sequence dimensions
            batch['spec_ae'] = spec_ae.unsqueeze(0).unsqueeze(0)
            print(f"spec_ae shape: {batch['spec_ae'].shape}")
            
            # Add features_ae - take first time step
            features_ae = sample['features_ae'].float()
            # Take first time step and add batch dimension
            batch['features_ae'] = features_ae[:, 0].unsqueeze(0).unsqueeze(0)
            print(f"features_ae shape: {batch['features_ae'].shape}")
            
        # Handle VIB inputs
        if "vib" in args.input_type:
            # Similar processing for vibration data
            spec_vib = sample['spec_vib'].float()
            print(f"Original spec_vib shape: {spec_vib.shape}")
            
            if spec_vib.dim() > 4:
                spec_vib = spec_vib[0]
                print(f"After reducing dimensions: {spec_vib.shape}")
                
            if spec_vib.dim() == 4:
                spec_vib = spec_vib[0]
                print(f"After taking first time step: {spec_vib.shape}")
                
            batch['spec_vib'] = spec_vib.unsqueeze(0).unsqueeze(0)
            print(f"spec_vib shape: {batch['spec_vib'].shape}")
            
            # Add features_vib - take first time step
            features_vib = sample['features_vib'].float()
            # Take first time step and add batch dimension
            batch['features_vib'] = features_vib[:, 0].unsqueeze(0).unsqueeze(0)
            print(f"features_vib shape: {batch['features_vib'].shape}")
        
        print(f"Batch dictionary created with keys: {list(batch.keys())}")
        
        # Get convolutional layers and select the correct one
        conv_layers = get_conv_layer_names(model)
        print(f"Found convolutional layers: {conv_layers}")
        if not conv_layers:
            print("No convolutional layers found.")
            continue

        # Choose the first convolutional layer for Grad-CAM
        target_layer = conv_layers[0]
        print(f"Using target layer: {target_layer}")

        # Determine which spectrogram key to use
        if "ae" in args.input_type:
            spec_key = 'spec_ae'
        elif "vib" in args.input_type:
            spec_key = 'spec_vib'
        else:
            spec_key = 'spec_ae'  # default
        
        # Run implemented XAI methods
        results = xai_manager.run_analysis(
            input_dict=batch,
            methods=methods_to_run,
            sample_idx=idx,
            input_type=args.input_type,
            report_path=REPORT_PATH
        )
        
        print(f"Completed XAI analysis for sample {idx}")
        
        # Clear memory between samples
        del batch
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run XAI analysis for grinding prediction models')
    parser.add_argument('--input_type', type=str, required=True, 
                        help='Type of input data (e.g., ae_spec, vib_spec)')
    parser.add_argument('--dataset_mode', type=str, default="classical", 
                        help='Dataset mode (classical, chunked, ram)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name of model to load (optional)')
    parser.add_argument('--sample_indices', type=str, default=None,
                        help='Comma-separated indices of samples to analyze (e.g., "0,42,100")')
    parser.add_argument('--xai_method', type=str, default="gradcam",
                        choices=['gradcam', 'integrated_gradients'],
                        help='XAI method to use (gradcam or integrated_gradients)')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.join(REPORT_PATH, "xai_reports"), exist_ok=True)
    try:
        run_xai_analysis(args)
    except Exception as e:
        print(f"Error running XAI analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("XAI analysis completed for selected samples")
