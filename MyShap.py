import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For distinct colors

# Assume your model and dataloader are defined and loaded
# from your_model_file import GrindingPredictor, collate_fn # Or however you define them
# from torch.utils.data import DataLoader, Dataset

# Placeholder for your actual model and data loading
# model = GrindingPredictor()
# model.load_state_dict(torch.load('your_model.pth'))
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Create a dummy dataset and dataloader for demonstration
# class DummyDataset(Dataset):
#     def __init__(self, num_samples=100, seq_len_range=(10, 20)):
#         self.num_samples = num_samples
#         self.seq_len_range = seq_len_range
#         self.data = []
#         for _ in range(num_samples):
#             ae_len = np.random.randint(seq_len_range[0], seq_len_range[1] + 1)
#             vib_len = np.random.randint(seq_len_range[0], seq_len_range[1] + 1)
#             self.data.append({
#                 'spec_ae': torch.randn(1, 2, 64, 64), # seq_len=1 for simplicity here, actual model handles variable
#                 'spec_vib': torch.randn(1, 3, 64, 64),
#                 'features_ae': torch.randn(ae_len, 4), # seq_len, num_features
#                 'features_vib': torch.randn(vib_len, 4),
#                 'features_pp': torch.randn(3),
#                 'label': torch.randn(1),
#                 'ae_lengths': torch.tensor(ae_len), # Store as single value, collate_fn will make it a tensor
#                 'vib_lengths': torch.tensor(vib_len)
#             })
#     def __len__(self):
#         return self.num_samples
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         # Simulate how data might look before collate_fn for lengths
#         return {
#             'spec_ae': item['spec_ae'].squeeze(0), # Remove batch dim for individual sample
#             'spec_vib': item['spec_vib'].squeeze(0),
#             'features_ae': item['features_ae'],
#             'features_vib': item['features_vib'],
#             'features_pp': item['features_pp'],
#             'label': item['label'],
#             'ae_lengths': item['ae_lengths'].item(), # Pass as scalar
#             'vib_lengths': item['vib_lengths'].item()
#         }

# def demo_collate_fn(batch):
#     # Simplified collate_fn for the dummy data, focusing on SHAP needs
#     # In reality, use your actual collate_fn
#     collated = {}
#     collated['spec_ae'] = torch.stack([item['spec_ae'] for item in batch])
#     collated['spec_vib'] = torch.stack([item['spec_vib'] for item in batch])
#     collated['features_pp'] = torch.stack([item['features_pp'] for item in batch])

#     # For sequences, padding is needed
#     ae_features_list = [item['features_ae'] for item in batch]
#     collated['features_ae'] = torch.nn.utils.rnn.pad_sequence(ae_features_list, batch_first=True)
#     collated['ae_lengths'] = torch.tensor([len(x) for x in ae_features_list], dtype=torch.long)

#     vib_features_list = [item['features_vib'] for item in batch]
#     collated['features_vib'] = torch.nn.utils.rnn.pad_sequence(vib_features_list, batch_first=True)
#     collated['vib_lengths'] = torch.tensor([len(x) for x in vib_features_list], dtype=torch.long)

#     collated['label'] = torch.stack([item['label'] for item in batch])
#     return collated

# dummy_dataset = DummyDataset(num_samples=50) # Smaller for SHAP background
# dataloader = DataLoader(dummy_dataset, batch_size=10, collate_fn=demo_collate_fn)


# --- Helper Function to prepare inputs for SHAP ---
def prepare_shap_inputs(batch, device):
    """
    Converts a batch dictionary into a tuple/list of tensors suitable for SHAP.
    The order matters and must be consistent between background and test data.
    SHAP typically works best with tensors, so we'll pass lengths separately if needed
    or ensure the model handles them internally if they are part of the tensors.
    """
    # Order: spec_ae, spec_vib, features_ae, features_vib, features_pp, ae_lengths, vib_lengths
    # Note: SHAP GradientExplainer might not directly use non-tensor 'lengths' in its gradient path.
    # If lengths are critical for the *explained part* of the model, they need to be tensors.
    # For now, we assume the core model parts SHAP explains take tensor inputs.
    return (
        batch['spec_ae'].to(device),
        batch['spec_vib'].to(device),
        batch['features_ae'].to(device),
        batch['features_vib'].to(device),
        batch['features_pp'].to(device),
        batch['ae_lengths'].to(device), # Ensure lengths are tensors
        batch['vib_lengths'].to(device)
    )

# --- SHAP Interpretation ---
def explain_with_shap(model, dataloader, device, num_background_samples=20, num_test_samples=5):
    """
    Generates and plots SHAP explanations.
    """
    model.eval()
    background_data_list = []
    count = 0
    for batch in dataloader:
        if count >= num_background_samples:
            break
        prepared_inputs = prepare_shap_inputs(batch, device)
        # Add individual samples from the batch to the background_data_list
        # This assumes batch_size > 1 for dataloader
        num_in_batch = prepared_inputs[0].size(0)
        for i in range(num_in_batch):
            if count >= num_background_samples:
                break
            sample_inputs = tuple(inp[i:i+1] for inp in prepared_inputs) # Keep batch dim of 1
            background_data_list.append(sample_inputs)
            count += 1

    if not background_data_list:
        print("Not enough data to form a SHAP background set.")
        return

    # Stack background data for each input type
    # Each element in background_tensors will be a tensor stacking all background samples for that input
    background_tensors = tuple(torch.cat([sample[i] for sample in background_data_list], dim=0)
                               for i in range(len(background_data_list[0])))


    # Define a wrapper for the model's forward pass that SHAP can use
    # It must take the tuple of tensors and return a single tensor output (the prediction)
    def model_shap_wrapper(*args):
        # Reconstruct the batch dictionary
        # Order must match prepare_shap_inputs
        batch_dict = {
            'spec_ae': args[0],
            'spec_vib': args[1],
            'features_ae': args[2],
            'features_vib': args[3],
            'features_pp': args[4],
            'ae_lengths': args[5],
            'vib_lengths': args[6]
        }
        predictions, _ = model(batch_dict) # We only need predictions for SHAP values
        return predictions

    # Create SHAP explainer
    # GradientExplainer is suitable for PyTorch models
    explainer = shap.GradientExplainer(model_shap_wrapper, background_tensors)

    # Get a few test samples
    test_data_list = []
    count = 0
    for batch in dataloader: # Ideally use a separate test dataloader
        if count >= num_test_samples:
            break
        prepared_inputs = prepare_shap_inputs(batch, device)
        num_in_batch = prepared_inputs[0].size(0)
        for i in range(num_in_batch):
            if count >= num_test_samples:
                break
            sample_inputs = tuple(inp[i:i+1] for inp in prepared_inputs) # Keep batch dim of 1
            test_data_list.append(sample_inputs)
            count +=1

    if not test_data_list:
        print("Not enough data for SHAP test set.")
        return

    test_tensors = tuple(torch.cat([sample[i] for sample in test_data_list], dim=0)
                         for i in range(len(test_data_list[0])))

    # Calculate SHAP values for the test samples
    # shap_values will be a list of tensors, one for each input in test_tensors
    print("Calculating SHAP values...")
    try:
        shap_values_list = explainer.shap_values(test_tensors)
    except Exception as e:
        print(f"Error during SHAP value calculation: {e}")
        print("Ensure background_tensors and test_tensors have compatible shapes and types.")
        print(f"Background tensor shapes: {[t.shape for t in background_tensors]}")
        print(f"Test tensor shapes: {[t.shape for t in test_tensors]}")
        return


    # --- Visualize SHAP Values ---

    # Define feature names for tabular data
    # Adjust these based on your actual features
    ae_feature_names = [f'AE_feat_{i}' for i in range(test_tensors[2].shape[-1])] # features_ae is index 2
    vib_feature_names = [f'Vib_feat_{i}' for i in range(test_tensors[3].shape[-1])] # features_vib is index 3
    pp_feature_names = ['SurfaceSpeed', 'WorkpieceRotation', 'GrindingDepth'] # features_pp is index 4
    # BDI and ec' might be part of features_pp or separate. Adjust accordingly.
    # For this example, assuming 3 PPs. If BDI, ec' are included, expand this list.

    # 1. SHAP for Process Parameters (features_pp)
    # shap_values_list[4] corresponds to features_pp
    # test_tensors[4] is the actual data for features_pp
    if len(shap_values_list) > 4 and shap_values_list[4] is not None:
        print("\nSHAP Summary for Process Parameters:")
        shap.summary_plot(shap_values_list[4].cpu().numpy(),
                          features=test_tensors[4].cpu().numpy(),
                          feature_names=pp_feature_names,
                          show=False)
        plt.title("SHAP Feature Importance: Process Parameters")
        plt.show()

    # 2. SHAP for AE Time Features (features_ae)
    # shap_values_list[2] corresponds to features_ae
    # We need to handle the sequence length. SHAP values are (batch, seq_len, num_features)
    # For summary plot, we can average SHAP values over the sequence length.
    if len(shap_values_list) > 2 and shap_values_list[2] is not None:
        shap_ae_time_avg = np.mean(shap_values_list[2].cpu().numpy(), axis=1) # Avg over seq_len
        features_ae_time_avg = np.mean(test_tensors[2].cpu().numpy(), axis=1)
        print("\nSHAP Summary for AE Time Features (averaged over sequence):")
        shap.summary_plot(shap_ae_time_avg,
                          features=features_ae_time_avg,
                          feature_names=ae_feature_names,
                          show=False)
        plt.title("SHAP Feature Importance: AE Time Features (Avg)")
        plt.show()

    # 3. SHAP for Vibration Time Features (features_vib) - Similar to AE
    if len(shap_values_list) > 3 and shap_values_list[3] is not None:
        shap_vib_time_avg = np.mean(shap_values_list[3].cpu().numpy(), axis=1)
        features_vib_time_avg = np.mean(test_tensors[3].cpu().numpy(), axis=1)
        print("\nSHAP Summary for Vibration Time Features (averaged over sequence):")
        shap.summary_plot(shap_vib_time_avg,
                          features=features_vib_time_avg,
                          feature_names=vib_feature_names,
                          show=False)
        plt.title("SHAP Feature Importance: Vibration Time Features (Avg)")
        plt.show()

    # 4. SHAP for Spectrograms (spec_ae, spec_vib)
    # shap_values_list[0] for spec_ae, shap_values_list[1] for spec_vib
    # These are (batch, channels, height, width)
    # SHAP image_plot expects (batch, height, width, channels) or (batch, height, width)
    if len(shap_values_list) > 0 and shap_values_list[0] is not None:
        print("\nSHAP Image Plot for AE Spectrogram (first test sample, first channel):")
        # Select first sample, convert to (H, W, C)
        shap_spec_ae_sample = shap_values_list[0][0].cpu().numpy().transpose(1, 2, 0)
        spec_ae_sample_data = test_tensors[0][0].cpu().numpy().transpose(1, 2, 0)
        shap.image_plot(shap_values_sample_ae_sample,
                        spec_ae_sample_data, # Show original image pixels
                        show=False)
        plt.suptitle("SHAP for AE Spectrogram (Sample 0)")
        plt.show()

    if len(shap_values_list) > 1 and shap_values_list[1] is not None:
        print("\nSHAP Image Plot for Vibration Spectrogram (first test sample, first channel):")
        shap_spec_vib_sample = shap_values_list[1][0].cpu().numpy().transpose(1, 2, 0)
        spec_vib_sample_data = test_tensors[1][0].cpu().numpy().transpose(1, 2, 0)
        shap.image_plot(shap_spec_vib_sample,
                        spec_vib_sample_data,
                        show=False)
        plt.suptitle("SHAP for Vibration Spectrogram (Sample 0)")
        plt.show()

# --- Attention Visualization ---
def visualize_attentions(model, dataloader, device, num_samples_to_plot=1):
    """
    Visualizes attention weights if the model outputs them.
    """
    model.eval()
    plotted_count = 0
    for batch in dataloader:
        if plotted_count >= num_samples_to_plot:
            break

        # Move batch to device (do this for all items in batch)
        batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

        with torch.no_grad():
            predictions, attentions_dict = model(batch_on_device)

        if not attentions_dict:
            print("Model did not return an attentions_dict.")
            return

        for i in range(batch['spec_ae'].size(0)): # Iterate over samples in the batch
            if plotted_count >= num_samples_to_plot:
                break

            print(f"\n--- Attention Plots for Sample {plotted_count} ---")
            fig, axes = plt.subplots(len(attentions_dict), 1, figsize=(12, 4 * len(attentions_dict)))
            if len(attentions_dict) == 1: # Make axes iterable if only one attention type
                axes = [axes]

            ax_idx = 0
            for attn_name, attn_weights in attentions_dict.items():
                if attn_weights is None: continue

                # Detach and move to CPU
                attn_sample = attn_weights[i].cpu().numpy() # Get attention for the i-th sample

                current_ax = axes[ax_idx]
                if "time_attn" in attn_name: # Temporal attention (e.g., from GRU)
                    # attn_sample shape might be (seq_len,) or (seq_len, 1)
                    if attn_sample.ndim > 1:
                        attn_sample = attn_sample.squeeze()

                    # Get actual sequence length for this sample
                    actual_len = None
                    if "ae_time_attn" in attn_name and 'ae_lengths' in batch:
                        actual_len = batch['ae_lengths'][i].item()
                    elif "vib_time_attn" in attn_name and 'vib_lengths' in batch:
                        actual_len = batch['vib_lengths'][i].item()
                    
                    if actual_len:
                        current_ax.plot(attn_sample[:actual_len])
                        current_ax.set_xlim(0, actual_len -1 if actual_len > 1 else 1)
                    else:
                        current_ax.plot(attn_sample)

                    current_ax.set_title(f"Attention: {attn_name} (Sample {plotted_count})")
                    current_ax.set_xlabel("Time Step")
                    current_ax.set_ylabel("Attention Weight")

                elif "feature_attn" in attn_name: # Feature-wise attention
                    # E.g., if you have attention over concatenated features before final MLP
                    # attn_sample shape might be (num_features,)
                    # You'd need feature_names_for_fusion = [...]
                    # current_ax.bar(feature_names_for_fusion, attn_sample)
                    # current_ax.set_title(f"Attention: {attn_name} (Sample {plotted_count})")
                    # current_ax.set_ylabel("Attention Weight")
                    # current_ax.tick_params(axis='x', rotation=45)
                    print(f"Plotting for feature attention '{attn_name}' not fully implemented yet. Shape: {attn_sample.shape}")
                    current_ax.plot(attn_sample) # Generic plot for now
                    current_ax.set_title(f"Attention: {attn_name} (Sample {plotted_count})")


                else: # Generic plot for other attention types
                    # This could be for attention over spectrogram patches, etc.
                    # Needs specific handling based on shape
                    im = current_ax.imshow(attn_sample, cmap='viridis', aspect='auto')
                    current_ax.set_title(f"Attention: {attn_name} (Sample {plotted_count})")
                    plt.colorbar(im, ax=current_ax)

                ax_idx += 1
            plt.tight_layout()
            plt.show()
            plotted_count += 1
        
        if plotted_count >= num_samples_to_plot:
            break


if __name__ == '__main__':
    # This is a placeholder for your actual setup
    # 1. Define or import your GrindingPredictor model
    # 2. Define or import your Dataset and collate_fn
    # 3. Load your trained model weights
    # 4. Create your DataLoader instance(s)

    print("--- Running Interpretability Demo ---")
    print("NOTE: This demo uses placeholder model and data.")
    print("Replace with your actual model, data, and feature names.")

    # --- Dummy Model and Data Setup for Demonstration ---
    class DummyTemporalFeatureProcessor(torch.nn.Module):
        def __init__(self, feat_dim, hidden_dim=8):
            super().__init__()
            self.gru = torch.nn.GRU(feat_dim, hidden_dim, batch_first=True)
            self.attention_net = torch.nn.Linear(hidden_dim, 1)
        def forward(self, x, lengths):
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False) # lengths must be on cpu for pack_padded
            gru_out, _ = self.gru(packed)
            gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=x.size(1))
            
            attn_logits = self.attention_net(gru_out) # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_logits, dim=1)
            context = (gru_out * attn_weights).sum(dim=1)
            return context, attn_weights.squeeze(-1)

    class DummyGrindingPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ae_spec_cnn = torch.nn.Conv2d(2, 4, 3, padding=1) # C_in=2
            self.vib_spec_cnn = torch.nn.Conv2d(3, 4, 3, padding=1) # C_in=3
            self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
            self.ae_time_proc = DummyTemporalFeatureProcessor(4, 8) # 4 AE time features
            self.vib_time_proc = DummyTemporalFeatureProcessor(4, 8) # 4 Vib time features
            self.pp_mlp = torch.nn.Linear(3, 8) # 3 process parameters
            self.final_mlp = torch.nn.Linear(4+4+8+8+8, 1) # spec_ae_flat + spec_vib_flat + ae_time + vib_time + pp

        def forward(self, batch):
            ae_s = self.pool(self.ae_spec_cnn(batch['spec_ae'])).view(batch['spec_ae'].size(0), -1)
            vib_s = self.pool(self.vib_spec_cnn(batch['spec_vib'])).view(batch['spec_vib'].size(0), -1)
            
            ae_t, ae_attn = self.ae_time_proc(batch['features_ae'], batch['ae_lengths'])
            vib_t, vib_attn = self.vib_time_proc(batch['features_vib'], batch['vib_lengths'])
            
            pp_f = torch.relu(self.pp_mlp(batch['features_pp']))
            
            combined = torch.cat([ae_s, vib_s, ae_t, vib_t, pp_f], dim=1)
            pred = self.final_mlp(combined)
            
            attns = {'ae_time_attn': ae_attn, 'vib_time_attn': vib_attn}
            return pred, attns

    model = DummyGrindingPredictor() # Replace with your actual model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create dummy dataset and dataloader for demonstration
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100, max_seq_len=20):
            self.num_samples = num_samples
            self.max_seq_len = max_seq_len
        def __len__(self): return self.num_samples
        def __getitem__(self, idx):
            ae_len = np.random.randint(5, self.max_seq_len + 1)
            vib_len = np.random.randint(5, self.max_seq_len + 1)
            return {
                'spec_ae': torch.randn(2, 32, 32), # C, H, W
                'spec_vib': torch.randn(3, 32, 32),
                'features_ae': torch.randn(ae_len, 4), # seq_len, num_features
                'features_vib': torch.randn(vib_len, 4),
                'features_pp': torch.randn(3),
                'label': torch.randn(1),
                'ae_lengths': ae_len,
                'vib_lengths': vib_len
            }

    def demo_collate_fn(batch_list):
        batch = {}
        batch['spec_ae'] = torch.stack([item['spec_ae'] for item in batch_list])
        batch['spec_vib'] = torch.stack([item['spec_vib'] for item in batch_list])
        batch['features_pp'] = torch.stack([item['features_pp'] for item in batch_list])
        batch['label'] = torch.stack([item['label'] for item in batch_list])

        ae_feat_list = [item['features_ae'] for item in batch_list]
        batch['features_ae'] = torch.nn.utils.rnn.pad_sequence(ae_feat_list, batch_first=True)
        batch['ae_lengths'] = torch.tensor([item['ae_lengths'] for item in batch_list], dtype=torch.long)

        vib_feat_list = [item['features_vib'] for item in batch_list]
        batch['features_vib'] = torch.nn.utils.rnn.pad_sequence(vib_feat_list, batch_first=True)
        batch['vib_lengths'] = torch.tensor([item['vib_lengths'] for item in batch_list], dtype=torch.long)
        return batch

    dummy_train_dataset = DummyDataset(num_samples=50) # For SHAP background
    train_dataloader = torch.utils.data.DataLoader(dummy_train_dataset, batch_size=10, collate_fn=demo_collate_fn)
    
    dummy_test_dataset = DummyDataset(num_samples=20) # For SHAP explanation and attention
    test_dataloader = torch.utils.data.DataLoader(dummy_test_dataset, batch_size=5, collate_fn=demo_collate_fn)

    # --- Run SHAP ---
    print("\n--- Running SHAP Explanations ---")
    explain_with_shap(model, train_dataloader, device, num_background_samples=20, num_test_samples=5)
    # Note: For real data, num_background_samples might need to be larger (e.g., 100-200)
    # but ensure it's manageable for computation time.

    # --- Run Attention Visualization ---
    print("\n--- Running Attention Visualizations ---")
    visualize_attentions(model, test_dataloader, device, num_samples_to_plot=2)
