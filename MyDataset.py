allowed_input_types = ['ae_spec', 'vib_spec', 'ae_spec+ae_features', 'vib_spec+vib_features', 'ae_spec+ae_features+vib_spec+vib_features', 'pp', 'all']

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import sys
sys.path.append("../utils/")

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scipy
import os
import glob
import itertools
import gc
import time
import librosa
from nptdms import TdmsFile
from scipy import stats
from natsort import natsorted

import dill as pickle
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, BSpline

from utils.preprocessing import (
    centimeter,
    one_column,
    two_column,
    cm_std,
    cm_bright,
    cm_highCon,
    cm_mark,
)

np.random.seed(16)

# from pydub import AudioSegment
import itertools
import string
import glob
import subprocess
import seedir
from utils.fusion import (
    compute_bdi,
    compute_ec,
    compute_st,
    process_vibration,
    process_ae,
    process_triaxial_vib,
)
from utils.preprocessing import print_tdms_structure, check_identical_csv_lengths
from utils.preprocessing import (
    linearSpectrogram,
    logMelSpectrogram,
    melSpectrogram,
    logSpectrogram,
    standardize_array,
    slice_indices,
)
from GrindingData import GrindingData

class MemoryDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

class GrindingDataset(Dataset):
    def __init__(self, grinding_data, input_type: str = "all"):
        self.required_components = set(input_type.split('+')) if input_type != 'all' else {'all'}
        self.loaded_data = self._select_data_components(grinding_data)

    def _select_data_components(self, grinding_data):
        """Selectively store only needed data based on input_type"""
        data = {
            'fn_names': grinding_data.fn_names,
            'sr': self._normalize(grinding_data.sr),
            'ec': self._normalize(grinding_data.ec),
            'st': self._normalize(grinding_data.st),
            'bid': self._normalize(grinding_data.bid),
            'label': grinding_data.sr
        }
        if 'all' in self.required_components or '_spec' in self.required_components:
            data['spec_data'] = grinding_data.spec_data

        if '_features' in self.required_components or 'all' in self.required_components:
            data['physical_data'] = grinding_data.physical_data
        return data

    def __len__(self):
        return len(self.loaded_data['fn_names'])

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

        item = {'label': torch.tensor(self.loaded_data['label'][idx],dtype=torch.long)}

        # Process features_pp (always included)
        item['features_pp'] = torch.tensor([
            self.loaded_data['ec'][idx],
            self.loaded_data['st'][idx],
            self.loaded_data['bid'][idx]
        ], dtype=torch.float32) 

        # Conditionally include other components
        if 'ae' in self.required_components or 'all' in self.required_components:
            item['features_ae'] = torch.tensor([
                self._normalize(self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['wavelet_energy_broad']),
                self._normalize(self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['wavelet_energy_narrow']),
                self._normalize(self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['burst_rate_narrow']),
                self._normalize(self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['burst_rate_broad']),
            ], dtype=torch.float32)

            item['spec_ae'] = torch.tensor(
                self.loaded_data['spec_data'][self.loaded_data['fn_names'][idx]]['spec_ae']
            )

        if 'vib' in self.required_components or 'all' in self.required_components:
            item['features_vib'] = torch.tensor([
                self._normalize(self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['env_kurtosis_x']),
                self._normalize(self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['env_kurtosis_y']),
                self._normalize(self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['env_kurtosis_z']),
                self._normalize(self.loaded_data['physical_data'][self.loaded_data['fn_names'][idx]]['mag']),
            ], dtype=torch.float32)
            item['spec_vib'] = torch.tensor(
                self.loaded_data['spec_data'][self.loaded_data['fn_names'][idx]]['spec_vib']
            )


        return item


    def _normalize(self, data):
        # Min-max normalization to [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero

def collate_fn(batch):
    # Process parameters and labels
    pp = torch.stack([item['features_pp'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # AE features
    ae_features = [item['features_ae'].permute(1,0) for item in batch]
    # ae_lengths = [x.shape[0] for x in ae_features]
    ae_padded = torch.nn.utils.rnn.pad_sequence(ae_features, batch_first=True)
    
    # Vibration features
    vib_features = [item['features_vib'].permute(1,0) for item in batch]
    # vib_lengths = [x.shape[0] for x in vib_features]
    vib_padded = torch.nn.utils.rnn.pad_sequence(vib_features, batch_first=True)
    
    # Spectrograms
    ae_specs = pad_spectrograms([item['spec_ae'] for item in batch])
    vib_specs = pad_spectrograms([item['spec_vib'] for item in batch])
    
    return {
        'features_pp': pp,
        'features_ae': ae_padded,
        'features_vib': vib_padded,
        'spec_ae': ae_specs,
        'spec_vib': vib_specs,
        'label': labels,
    }

def pad_spectrograms(spectrograms):
    max_len = max(spec.shape[0] for spec in spectrograms)
    padded = []
    for spec in spectrograms:
        padding = max_len - spec.shape[0]
        padded.append(torch.cat([
            spec,
            torch.zeros((padding, *spec.shape[1:]), dtype=spec.dtype)
        ], dim=0))
    return torch.stack(padded)

def get_collate_fn(input_type='all'):
    # Parse input type into components
    required = set(input_type.split('+')) if input_type != 'all' else {'all'}
    print(f"Required components: {required}")
    
    def collate_fn(batch):
        def pad_spectrograms(spectrograms):
            max_len = max(spec.shape[0] for spec in spectrograms)
            return torch.stack([
                torch.cat([spec, torch.zeros((max_len - spec.shape[0], *spec.shape[1:]), 
                       dtype=spec.dtype)], dim=0)
                for spec in spectrograms
            ])

        # Always present components
        batch_dict = {
            'features_pp': torch.stack([item['features_pp'] for item in batch]),
            'label': torch.stack([item['label'] for item in batch])
        }

        # # Conditionally process AE components
        # if 'all' in required or 'ae' in required:
        # # if 'all' in required or 'ae_features' in required:
        #     ae_features = [item['features_ae'].permute(1,0) for item in batch]
        #     batch_dict['features_ae'] = torch.nn.utils.rnn.pad_sequence(ae_features, batch_first=True)
        #     batch_dict['spec_ae'] = pad_spectrograms([item['spec_ae'] for item in batch])

        # # Conditionally process VIB components
        # if 'all' in required or 'vib' in required:
        #     batch_dict['spec_vib'] = pad_spectrograms([item['spec_vib'] for item in batch])
        # # if 'all' in required or 'vib_features' in required:
        #     vib_features = [item['features_vib'].permute(1,0) for item in batch]
        #     batch_dict['features_vib'] = torch.nn.utils.rnn.pad_sequence(vib_features, batch_first=True)

        ae_features = [item['features_ae'].permute(1,0) for item in batch]
        vib_features = [item['features_vib'].permute(1,0) for item in batch]

        batch_dict['spec_ae'] = pad_spectrograms([item['spec_ae'] for item in batch])
        batch_dict['spec_vib'] = pad_spectrograms([item['spec_vib'] for item in batch])
        batch_dict['features_ae'] = torch.nn.utils.rnn.pad_sequence(ae_features, batch_first=True)
        batch_dict['features_vib'] = torch.nn.utils.rnn.pad_sequence(vib_features, batch_first=True)

        return batch_dict

    return collate_fn

def get_dataset(input_type: str = "all", dataset_mode: str = "classical"):
    data = load_init_data()
    grinding_data = data['grinding_data']
    if input_type not in allowed_input_types:
        raise ValueError(f"input_type must be one of {allowed_input_types}")

    # === Only need to modify the block line for new dataset ===
    # Only load necessary data based on input_type
    if 'spec' in input_type or input_type == 'all':
        print("Loading all spectrograms")
        grinding_data._load_all_spec_data()  # Assuming this loads AE spectrograms
 
    # if 'ae_features' in input_type or input_type == 'all':
        # grinding_data._load_all_physics_data() 
    # if 'vib_features' in input_type or input_type == 'all':
        # grinding_data._load_all_physics_data() 
    
    # grinding_data._load_all_physics_data()
    # grinding_data._load_all_spec_data()

    dataset = GrindingDataset(grinding_data)
    # === Only need to modify the block for new dataset ===

    if dataset_mode == "chunked":
        dataset = dataset
    elif dataset_mode == "ram":
        full_data = []
        size_bytes = 0
        for item in dataset:
            full_data.append(item)
            size_bytes += sys.getsizeof(item)
        size_bytes += sys.getsizeof(full_data)
        size_gb = size_bytes / (1024 ** 3)
        # full_data = [dataset[i] for i in range(len(dataset))]
        print("Load dataset into RAM")
        print(f"Estimated size of full_data: {size_gb:.2f} GB")
        dataset = MemoryDataset(full_data)

    elif dataset_mode == "classical":
        dataset = dataset
    else:
        raise ValueError(f"train_mode must be one of ['classical', 'chunked', 'ram'], but got {dataset_mode}")

    return dataset

def load_init_data():
    project_name = ["Grinding", "XiAnJiaoTong"]

    if os.name == "posix":
        project_dir: str = os.path.join(
            subprocess.getoutput("echo $DATADIR"),
            *project_name,
        )
    elif os.name == "nt":
        project_dir: str = os.path.join(
            subprocess.getoutput("echo %datadir%"), *project_name
        )

    dataDir_ae = os.path.join(project_dir, "AE")
    dataDir_vib = os.path.join(project_dir, "Vibration")
    grinding_data = GrindingData(project_dir)
    grinding_data._load_all_physics_data() 
    return {
        "dataDir_ae": dataDir_ae,
        "dataDir_vib": dataDir_vib,
        "grinding_data": grinding_data
    }