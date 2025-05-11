import torch
import psutil
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import torch.nn as nn
import numpy as np
import sys
import itertools
import string
import glob
import subprocess
from tqdm import tqdm
sys.path.append("../utils/")
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
from utils.MLUtils import getSubsetIdx

# Project settings
alphabet = list(string.ascii_lowercase)
sampling_rate_ae = 4*1e6
sampling_rate_vib = 51.2*1e3
project_name = ["Grinding","XiAnJiaoTong"]
if os.name == "posix":
    data_dir = subprocess.getoutput("echo $DATADIR")
elif os.name == "nt":
    data_dir = subprocess.getoutput("echo %datadir%")
project_dir = os.path.join(data_dir, *project_name)
if not os.path.exists(project_dir):
    project_name[0] = os.path.join("2024-MUSIC","Grinding")
project_dir = os.path.join(data_dir, *project_name)
dataDir_ae = os.path.join(project_dir,"AE")
dataDir_vib = os.path.join(project_dir,"Vibration")

allowed_input_types = ['ae_spec', 'vib_spec', 'ae_features','vib_features', 'ae_spec+ae_features', 'vib_spec+vib_features', 'ae_spec+ae_features+vib_spec+vib_features', 'ae_features+pp', 'vib_features+pp','pp', 'ae_spec+vib_spec', 'ae_features+vib_features', 'ae_features+vib_features+pp','all']

logical_threads = psutil.cpu_count(logical=True)
physical_threads = psutil.cpu_count(logical=False)
cpus = [logical_threads, physical_threads, 2, 1]
percentage=[0.6, 0.8, 0.90, 1]
# End project settings

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

        # Normalize the surface roughness (sr) values to [0, 1]
        # self._encoder()

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
        if 'all' in self.required_components or '_spec' in self.required_components:
            data['spec_data'] = grinding_data.spec_data

        if 'pp' in self.required_components or 'all' in self.required_components:
            data['physical_data'] = grinding_data.physical_data

        if '_features' in self.required_components or 'all' in self.required_components:
            data['physical_data'] = grinding_data.physical_data
        

        return data

    def _encoder(self):
        # Standardize the surface roughness to have mean 0 and variance 1
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # _d = scaler.fit_transform(self.loaded_data['label'].reshape(-1, 1)).squeeze()
        # self.loaded_data['label'] = _d
        self.loaded_data['label'] = self.loaded_data['label']*1e3

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
            spectrograms = [spec.squeeze() for spec in spectrograms]
            max_len = max(spec.shape[0] for spec in spectrograms)
            return torch.stack([
                torch.cat([spec, torch.zeros((max_len - spec.shape[0], *spec.shape[1:]), 
                       dtype=spec.dtype)], dim=0)
                for spec in spectrograms
            ])

        # Always present components
        batch_dict = {
            'features_pp': torch.stack([item['features_pp'].squeeze() for item in batch]),
            'label': torch.stack([item['label'].squeeze() for item in batch])
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

        ae_features = [item['features_ae'].squeeze().permute(1,0) for item in batch]
        vib_features = [item['features_vib'].squeeze().permute(1,0) for item in batch]

        batch_dict['spec_ae'] = pad_spectrograms([item['spec_ae'] for item in batch])
        batch_dict['spec_vib'] = pad_spectrograms([item['spec_vib'] for item in batch])
        batch_dict['features_ae'] = torch.nn.utils.rnn.pad_sequence(ae_features, batch_first=True)
        batch_dict['features_vib'] = torch.nn.utils.rnn.pad_sequence(vib_features, batch_first=True)

        return batch_dict

    return collate_fn

def get_dataset(input_type: str = "all", dataset_mode: str = "classical",cpus = [logical_threads,1], percentage = [0.6, 1.0]):
    data = load_init_data()
    grinding_data = data['grinding_data']
    if input_type not in allowed_input_types:
        raise ValueError(f"input_type must be one of {allowed_input_types}")

    grinding_data._load_all_spec_data()  # Assuming this loads AE spectrograms
    grinding_data._load_all_physics_data()  
    # === Only need to modify the block line for new dataset ===
    # Only load necessary data based on input_type
    # if 'spec' in input_type or input_type == 'all':
    #     print("Loading all spectrograms")
    #     grinding_data._load_all_spec_data()  # Assuming this loads AE spectrograms

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
        # size_bytes = 0

        lenDataset = len(dataset)
        idx = getSubsetIdx(lenDataset, percentage, cpus)
        keys = list(idx.keys())
        for _c,_k in zip(cpus[:-1], keys[:-1]):
            _idx = idx[_k]
            ramDataLoader = DataLoader(Subset(dataset, np.array(_idx)), batch_size=1, shuffle=False, num_workers=int(_c), pin_memory=False,prefetch_factor=1)
            for i,item in tqdm(enumerate(ramDataLoader),desc=f'Loading {_k} data for {_c} threads'):
                full_data.append(item)
                # size_bytes += sys.getsizeof(item)
            # print(f"Estimated size of data: {size_bytes:.2f} GB")
            print(f"Loading threads ({_c}) with remaining of data ({len(full_data)}/{len(dataset)})")
            del ramDataLoader
            gc.collect()

        for i in tqdm(idx[keys[-1]], desc=f'Loading {keys[-1]} data for single thread'):
            full_data.append(dataset[i])
            # size_bytes += sys.getsizeof(dataset[i])
        
        size_bytes = sys.getsizeof(full_data)
        size_gb = size_bytes / (1024 ** 3)
        dataset = MemoryDataset(full_data)
        print(f"Length of full_data: {len(full_data)}")
        print(f"Estimated size of full_data: {size_gb:.2f} GB")

    elif dataset_mode == "classical":
        dataset = dataset
    else:
        raise ValueError(f"train_mode must be one of ['classical', 'chunked', 'ram'], but got {dataset_mode}")

    return dataset

def load_init_data():
    grinding_data = GrindingData(project_dir)
    grinding_data._load_all_physics_data() 
    return {
        "dataDir_ae": dataDir_ae,
        "dataDir_vib": dataDir_vib,
        "grinding_data": grinding_data
    }