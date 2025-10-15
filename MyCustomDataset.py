import torch
import numpy as np
import os
import sys

# Add parent directory to path to import GrindingData and MyDataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MyDataset import GrindingDataset, project_dir, load_init_data, allowed_input_types
from GrindingData import GrindingData

class MyCustomDataset(GrindingDataset):
    def __init__(self, grinding_data, input_type: str = "all"):
        super().__init__(grinding_data, input_type)

    def _select_data_components(self, grinding_data):
        """Selectively store only needed data based on input_type"""
        data = {
            'fn_names': grinding_data.fn_names,
            'sr': self._normalize(grinding_data.sr),
            'ec': self._normalize(grinding_data.ec),
            'st': self._normalize(grinding_data.st),
            'bid': self._normalize(grinding_data.bid),
            'label': grinding_data.sr*1e3
        }
        # Always include spec_data and physical_data, as they are loaded by grinding_data._load_all_spec_data() and _load_all_physics_data()
        data['spec_data'] = grinding_data.spec_data
        data['physical_data'] = grinding_data.physical_data
        
        return data

    def __getitem__(self, idx):
        # Handle different index types
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        elif isinstance(idx, np.ndarray):
            idx = idx.item()
        elif isinstance(idx, slice):
            raise ValueError("Slice indexing not supported, use list of indices instead")
        idx = int(idx)
        if not isinstance(idx, int):
            raise TypeError(f"Index must be int, got {type(idx)}")

        item = {
            'label': torch.tensor(self.loaded_data['label'][idx], dtype=torch.long),
            'features_pp': torch.tensor([
                self.loaded_data['ec'][idx],
                self.loaded_data['st'][idx],
                self.loaded_data['bid'][idx]
            ], dtype=torch.float32)
        }

        # Conditionally include AE components
        if 'ae_spec' in self.required_components or 'all' in self.required_components:
            raw_spec_ae = self.loaded_data['spec_data'][self.loaded_data['fn_names'][idx]]['spec_ae']
            item['spec_ae'] = torch.tensor(raw_spec_ae) # Already has seq_len, C, H, W

        if 'ae_features' in self.required_components or 'all' in self.required_components:
            raw_ae_features_values = [
                self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['wavelet_energy_broad'],
                self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['wavelet_energy_narrow'],
                self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['burst_rate_narrow'],
                self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['burst_rate_broad'],
            ]
            
            # Determine seq_len for padding based on spec_ae if available, otherwise use max length of features
            seq_len_ae = item['spec_ae'].shape[0] if 'spec_ae' in item else max(len(f) for f in raw_ae_features_values)

            # Pad raw_ae_features_values to match seq_len_ae
            padded_ae_features = []
            for feature_array in raw_ae_features_values:
                normalized_feature = self._normalize(feature_array)
                padding = seq_len_ae - len(normalized_feature)
                if padding < 0: # If feature is longer than target seq_len, truncate
                    padded_ae_features.append(normalized_feature[:seq_len_ae])
                else:
                    padded_ae_features.append(np.pad(normalized_feature, (0, padding), 'constant'))
            
            item['features_ae'] = torch.tensor(np.stack(padded_ae_features), dtype=torch.float32).permute(1, 0) # [seq_len, num_features]

        # Conditionally include VIB components
        if 'vib_spec' in self.required_components or 'all' in self.required_components:
            raw_spec_vib = self.loaded_data['spec_data'][self.loaded_data['fn_names'][idx]]['spec_vib']
            item['spec_vib'] = torch.tensor(raw_spec_vib) # Already has seq_len, C, H, W

        if 'vib_features' in self.required_components or 'all' in self.required_components:
            raw_vib_features_values = [
                self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['env_kurtosis_x'],
                self._normalize(self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['env_kurtosis_y']),
                self._normalize(self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['env_kurtosis_z']),
                self._normalize(self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['mag']),
            ]
            
            # Determine seq_len for padding based on spec_vib if available, otherwise use max length of features
            seq_len_vib = item['spec_vib'].shape[0] if 'spec_vib' in item else max(len(f) for f in raw_vib_features_values)

            # Pad raw_vib_features_values to match seq_len_vib
            padded_vib_features = []
            for feature_array in raw_vib_features_values:
                normalized_feature = self._normalize(feature_array)
                padding = seq_len_vib - len(normalized_feature)
                if padding < 0: # If feature is longer than target seq_len, truncate
                    padded_vib_features.append(normalized_feature[:seq_len_vib])
                else:
                    padded_vib_features.append(np.pad(normalized_feature, (0, padding), 'constant'))
            
            item['features_vib'] = torch.tensor(np.stack(padded_vib_features), dtype=torch.float32).permute(1, 0) # [seq_len, num_features]

        return item


    def _normalize(self, data):
        # Min-max normalization to [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero

def get_custom_dataset(input_type: str = "all", dataset_mode: str = "classical"):
    data = load_init_data()
    grinding_data = data['grinding_data']
    if input_type not in allowed_input_types: # Use original allowed_input_types
        raise ValueError(f"input_type must be one of {allowed_input_types}")

    grinding_data._load_all_spec_data()  # Assuming this loads AE spectrograms
    grinding_data._load_all_physics_data()  
    
    dataset = MyCustomDataset(grinding_data, input_type)

    if dataset_mode == "classical":
        return dataset
    else:
        raise ValueError(f"dataset_mode must be 'classical' for custom dataset, but got {dataset_mode}")
