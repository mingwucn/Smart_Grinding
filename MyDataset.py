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
    def __init__(self, grinding_data):
        self.fn_names = grinding_data.fn_names
        self.physical_data = grinding_data.physical_data
        self.spec_data = grinding_data.spec_data
        self.sr  = self._normalize(grinding_data.sr)
        self.ec  = self._normalize(grinding_data.ec)
        self.st  = self._normalize(grinding_data.st)
        self.bid = self._normalize(grinding_data.bid)

    def __len__(self):
        return len(self.fn_names)

    def __getitem__(self, idx):
        fn = self.fn_names[idx]
        data = self.physical_data[fn]
        
        # Extract features
        features_ae = torch.tensor([
            self._normalize(data['wavelet_energy_broad']),
            self._normalize(data['wavelet_energy_narrow']),
            self._normalize(data['burst_rate_narrow']),
            self._normalize(data['burst_rate_broad']),
        ], dtype=torch.float32)

        features_vib = torch.tensor([
            self._normalize(data['env_kurtosis_x']),
            self._normalize(data['env_kurtosis_y']),
            self._normalize(data['env_kurtosis_z']),
            self._normalize(data['mag']),
        ], dtype=torch.float32)
        
        features_pp = torch.tensor([
            self.ec[idx],
            self.st[idx],
            self.bid[idx]
        ], dtype=torch.float32)

        # Extract sepc_data
        spec_ae = torch.tensor(self.spec_data[fn]['spec_ae'])
        spec_vib = torch.tensor(self.spec_data[fn]['spec_vib'])
       
        # Extract label
        label = self.sr[idx]
        label = torch.tensor(label, dtype=torch.long)
        
        # return spec_ae, spec_vib, features_pp, features_ae, features_vib, label
        return {
            'spec_ae': spec_ae,
            'spec_vib': spec_vib,
            'features_pp': features_pp,
            'features_ae': features_ae,
            'features_vib': features_vib,
            'label': label
        }


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

def get_dataset(input_type: str = "all"):

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
    grinding_data._load_all_spec_data()
    dataset = GrindingDataset(grinding_data)
    return dataset