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
sampling_rate_ae = 4 * 1e6
sampling_rate_vib = 51.2 * 1e3
project_name = ["Grinding", "XiAnJiaoTong"]
if os.name == "posix":
    data_dir = subprocess.getoutput("echo $DATADIR")
elif os.name == "nt":
    data_dir = subprocess.getoutput("echo %datadir%")
project_dir = os.path.join(data_dir, *project_name)
if not os.path.exists(project_dir):
    project_name[0] = os.path.join("2024-MUSIC", "Grinding")
project_dir = os.path.join(data_dir, *project_name)
dataDir_ae = os.path.join(project_dir, "AE")
dataDir_vib = os.path.join(project_dir, "Vibration")

allowed_input_types = [
    "ae_spec",
    "ae_features",
    "ae_features+pp",
    "ae_spec+ae_features",
    "vib_spec",
    "vib_features",
    "vib_features+pp",
    "vib_spec+vib_features",
    'ae_features+vib_features',
    "ae_features+vib_features+pp",
    "ae_spec+vib_spec",
    "ae_spec+ae_features+vib_spec+vib_features",
    "all",
]

logical_threads = psutil.cpu_count(logical=True)
physical_threads = psutil.cpu_count(logical=False)
cpus = [logical_threads, physical_threads, 2, 1]
percentage = [0.6, 0.8, 0.90, 1]
# End project settings

from GrindingData import GrindingData


class MemoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class LazyGrindingDataset(Dataset):
    """Memory-efficient dataset that loads data on-demand."""
    def __init__(self, grinding_data, input_type: str = "all"):
        self.required_components = (
            set(input_type.split("+")) if input_type != "all" else {"all"}
        )
        self.grinding_data = grinding_data
        self.fn_names = grinding_data.fn_names
        self.sr = grinding_data.sr
        self.ec = grinding_data.ec
        self.st = grinding_data.st
        self.bid = grinding_data.bid
        self.label = grinding_data.sr * 1e3
        
        # Store paths instead of loaded data
        self.project_dir = grinding_data.project_dir
        self.intermediate_dir = os.path.join(self.project_dir, "intermediate")
        
        # Cache for loaded data to avoid reloading same file multiple times
        self._spec_cache = {}
        self._physics_cache = {}
        
        # Pre-compute min-max for normalization
        self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute min-max stats for normalization."""
        # For simplicity, we'll compute stats from available data
        # In practice, you might want to pre-compute these from all data
        self._norm_stats = {
            'sr': (np.min(self.sr), np.max(self.sr)),
            'ec': (np.min(self.ec), np.max(self.ec)),
            'st': (np.min(self.st), np.max(self.st)),
            'bid': (np.min(self.bid), np.max(self.bid)),
        }

    def _load_physics_data(self, fn_name):
        """Load physics data for a specific file."""
        if fn_name in self._physics_cache:
            return self._physics_cache[fn_name]
        
        path = os.path.join(self.intermediate_dir, f"{fn_name}_physics.npz")
        if os.path.exists(path):
            try:
                data = np.load(path)
                # Convert to dictionary and cache
                physics_data = {key: data[key] for key in data.files}
                self._physics_cache[fn_name] = physics_data
                return physics_data
            except Exception as e:
                # If file exists but can't be loaded, return empty data
                print(f"Warning: Error loading physics data for {fn_name}: {e}")
                # Return empty physics data
                empty_physics_data = {
                    "wavelet_energy_broad": np.zeros(1, dtype=np.float32),
                    "wavelet_energy_narrow": np.zeros(1, dtype=np.float32),
                    "burst_rate_narrow": np.zeros(1, dtype=np.float32),
                    "burst_rate_broad": np.zeros(1, dtype=np.float32),
                    "env_kurtosis_x": np.zeros(1, dtype=np.float32),
                    "env_kurtosis_y": np.zeros(1, dtype=np.float32),
                    "env_kurtosis_z": np.zeros(1, dtype=np.float32),
                    "mag": np.zeros(1, dtype=np.float32),
                    "ec": np.zeros(1, dtype=np.float32),
                    "bid": np.zeros(1, dtype=np.float32),
                    "st": np.zeros(1, dtype=np.float32),
                }
                self._physics_cache[fn_name] = empty_physics_data
                return empty_physics_data
        else:
            # If file doesn't exist, return empty data
            print(f"Warning: Physics data file not found: {path}")
            empty_physics_data = {
                "wavelet_energy_broad": np.zeros(1, dtype=np.float32),
                "wavelet_energy_narrow": np.zeros(1, dtype=np.float32),
                "burst_rate_narrow": np.zeros(1, dtype=np.float32),
                "burst_rate_broad": np.zeros(1, dtype=np.float32),
                "env_kurtosis_x": np.zeros(1, dtype=np.float32),
                "env_kurtosis_y": np.zeros(1, dtype=np.float32),
                "env_kurtosis_z": np.zeros(1, dtype=np.float32),
                "mag": np.zeros(1, dtype=np.float32),
                "ec": np.zeros(1, dtype=np.float32),
                "bid": np.zeros(1, dtype=np.float32),
                "st": np.zeros(1, dtype=np.float32),
            }
            self._physics_cache[fn_name] = empty_physics_data
            return empty_physics_data

    def _load_spec_data(self, fn_name):
        """Load spectrogram data for a specific file."""
        if fn_name in self._spec_cache:
            return self._spec_cache[fn_name]
        
        path = os.path.join(self.intermediate_dir, f"{fn_name}_spec.npz")
        if os.path.exists(path):
            try:
                data = np.load(path)
                # Convert to dictionary and cache
                spec_data = {key: data[key] for key in data.files}
                self._spec_cache[fn_name] = spec_data
                return spec_data
            except Exception as e:
                # If file exists but can't be loaded, return empty data
                print(f"Warning: Error loading spec data for {fn_name}: {e}")
                # Return empty spec data
                empty_spec_data = {
                    "spec_ae": np.zeros((2, 300, 64), dtype=np.float32),
                    "spec_vib": np.zeros((3, 300, 64), dtype=np.float32)
                }
                self._spec_cache[fn_name] = empty_spec_data
                return empty_spec_data
        else:
            # If file doesn't exist, return empty data
            print(f"Warning: Spec data file not found: {path}")
            empty_spec_data = {
                "spec_ae": np.zeros((2, 300, 64), dtype=np.float32),
                "spec_vib": np.zeros((3, 300, 64), dtype=np.float32)
            }
            self._spec_cache[fn_name] = empty_spec_data
            return empty_spec_data

    def _normalize(self, data, data_type):
        """Normalize data using pre-computed stats."""
        if data_type in self._norm_stats:
            min_val, max_val = self._norm_stats[data_type]
            return (data - min_val) / (max_val - min_val + 1e-8)
        return data

    def __len__(self):
        return len(self.fn_names)

    def __getitem__(self, idx):
        idx = int(idx)
        fn_name = self.fn_names[idx]
        
        item = {
            "label": torch.tensor(self.label[idx], dtype=torch.float32),
            "features_pp": torch.tensor([
                self._normalize(self.ec[idx], 'ec'),
                self._normalize(self.st[idx], 'st'),
                self._normalize(self.bid[idx], 'bid')
            ], dtype=torch.float32)
        }

        # Load data on-demand based on required components
        # Check if AE components are needed
        needs_ae = False
        needs_ae_spec = False
        needs_ae_features = False
        
        for comp in self.required_components:
            if comp == "all" or "ae" in comp:
                needs_ae = True
                if "spec" in comp:
                    needs_ae_spec = True
                if "features" in comp:
                    needs_ae_features = True
        
        if needs_ae:
            # Always create features_ae (even if empty) because model expects it
            if needs_ae_features:
                physics_data = self._load_physics_data(fn_name)
                # AE features
                features_ae_list = [
                    self._normalize(physics_data["wavelet_energy_broad"], 'wavelet_energy_broad'),
                    self._normalize(physics_data["wavelet_energy_narrow"], 'wavelet_energy_narrow'),
                    self._normalize(physics_data["burst_rate_narrow"], 'burst_rate_narrow'),
                    self._normalize(physics_data["burst_rate_broad"], 'burst_rate_broad'),
                ]
                item["features_ae"] = torch.tensor(np.array(features_ae_list), dtype=torch.float32)
            else:
                # Add empty features_ae if model expects it
                item["features_ae"] = torch.zeros(4, dtype=torch.float32)
            
            if needs_ae_spec:
                spec_data = self._load_spec_data(fn_name)
                item["spec_ae"] = torch.tensor(spec_data["spec_ae"], dtype=torch.float32)
            else:
                # Add empty spec_ae if model expects it
                item["spec_ae"] = torch.zeros((2, 300, 64), dtype=torch.float32)

        # Check if VIB components are needed
        needs_vib = False
        needs_vib_spec = False
        needs_vib_features = False
        
        for comp in self.required_components:
            if comp == "all" or "vib" in comp:
                needs_vib = True
                if "spec" in comp:
                    needs_vib_spec = True
                if "features" in comp:
                    needs_vib_features = True
        
        if needs_vib:
            physics_data = self._load_physics_data(fn_name)
            
            if needs_vib_features:
                # Vibration features
                features_vib_list = [
                    self._normalize(physics_data["env_kurtosis_x"], 'env_kurtosis_x'),
                    self._normalize(physics_data["env_kurtosis_y"], 'env_kurtosis_y'),
                    self._normalize(physics_data["env_kurtosis_z"], 'env_kurtosis_z'),
                    self._normalize(physics_data["mag"], 'mag'),
                ]
                item["features_vib"] = torch.tensor(np.array(features_vib_list), dtype=torch.float32)
            else:
                # Add empty features_vib if model expects it
                item["features_vib"] = torch.zeros(4, dtype=torch.float32)
            
            if needs_vib_spec:
                spec_data = self._load_spec_data(fn_name)
                item["spec_vib"] = torch.tensor(spec_data["spec_vib"], dtype=torch.float32)
            else:
                # Add empty spec_vib if model expects it
                item["spec_vib"] = torch.zeros((3, 300, 64), dtype=torch.float32)

        return item


class GrindingDataset(Dataset):
    def __init__(self, grinding_data, input_type: str = "all"):
        self.required_components = (
            set(input_type.split("+")) if input_type != "all" else {"all"}
        )
        self.loaded_data = self._select_data_components(grinding_data)

        # Normalize the surface roughness (sr) values to [0, 1]
        # self._encoder()

    def _select_data_components(self, grinding_data):
        """Selectively store only needed data based on input_type"""
        data = {
            "fn_names": grinding_data.fn_names,
            "sr": self._normalize(grinding_data.sr),
            "ec": self._normalize(grinding_data.ec),
            "st": self._normalize(grinding_data.st),
            "bid": self._normalize(grinding_data.bid),
            "label": grinding_data.sr * 1e3,
        }
        if "all" in self.required_components or "_spec" in self.required_components:
            data["spec_data"] = grinding_data.spec_data

        if "pp" in self.required_components or "all" in self.required_components:
            data["physical_data"] = grinding_data.physical_data

        if "_features" in self.required_components or "all" in self.required_components:
            data["physical_data"] = grinding_data.physical_data

        return data

    def _encoder(self):
        # Standardize the surface roughness to have mean 0 and variance 1
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # _d = scaler.fit_transform(self.loaded_data['label'].reshape(-1, 1)).squeeze()
        # self.loaded_data['label'] = _d
        self.loaded_data["label"] = self.loaded_data["label"] * 1e3

    def __len__(self):
        return len(self.loaded_data["fn_names"])

    def __getitem__(self, idx):
        # Handle different index types
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        elif isinstance(idx, np.ndarray):
            idx = idx.item()
        elif isinstance(idx, slice):
            raise ValueError(
                "Slice indexing not supported, use list of indices instead"
            )
        idx = int(idx)
        if not isinstance(idx, int):
            raise TypeError(f"Index must be int, got {type(idx)}")

        item = {"label": torch.tensor(self.loaded_data["label"][idx], dtype=torch.long)}

        # Process features_pp (always included)
        item["features_pp"] = torch.tensor(
            [
                self.loaded_data["ec"][idx],
                self.loaded_data["st"][idx],
                self.loaded_data["bid"][idx],
            ],
            dtype=torch.float32,
        )

        # Conditionally include other components
        if "ae" in self.required_components or "all" in self.required_components:
            # Handle missing physical data
            if "physical_data" in self.loaded_data and self.loaded_data["fn_names"][idx] in self.loaded_data["physical_data"]:
                phys_data = self.loaded_data["physical_data"][self.loaded_data["fn_names"][idx]]
                features_ae_list = [
                    self._normalize(phys_data.get("wavelet_energy_broad", 0)),
                    self._normalize(phys_data.get("wavelet_energy_narrow", 0)),
                    self._normalize(phys_data.get("burst_rate_narrow", 0)),
                    self._normalize(phys_data.get("burst_rate_broad", 0)),
                ]
            else:
                features_ae_list = [0.0, 0.0, 0.0, 0.0]
            
            item["features_ae"] = torch.tensor(np.array(features_ae_list), dtype=torch.float32)

            # Handle missing spec data
            if "spec_data" in self.loaded_data and self.loaded_data["fn_names"][idx] in self.loaded_data["spec_data"]:
                spec_data = self.loaded_data["spec_data"][self.loaded_data["fn_names"][idx]]
                item["spec_ae"] = torch.tensor(spec_data.get("spec_ae", np.zeros((2, 300, 64))))
            else:
                item["spec_ae"] = torch.zeros((2, 300, 64), dtype=torch.float32)

        if "vib" in self.required_components or "all" in self.required_components:
            # Handle missing physical data
            if "physical_data" in self.loaded_data and self.loaded_data["fn_names"][idx] in self.loaded_data["physical_data"]:
                phys_data = self.loaded_data["physical_data"][self.loaded_data["fn_names"][idx]]
                features_vib_list = [
                    self._normalize(phys_data.get("env_kurtosis_x", 0)),
                    self._normalize(phys_data.get("env_kurtosis_y", 0)),
                    self._normalize(phys_data.get("env_kurtosis_z", 0)),
                    self._normalize(phys_data.get("mag", 0)),
                ]
            else:
                features_vib_list = [0.0, 0.0, 0.0, 0.0]
            
            item["features_vib"] = torch.tensor(np.array(features_vib_list), dtype=torch.float32)

            # Handle missing spec data
            if "spec_data" in self.loaded_data and self.loaded_data["fn_names"][idx] in self.loaded_data["spec_data"]:
                spec_data = self.loaded_data["spec_data"][self.loaded_data["fn_names"][idx]]
                item["spec_vib"] = torch.tensor(spec_data.get("spec_vib", np.zeros((3, 300, 64))))
            else:
                item["spec_vib"] = torch.zeros((3, 300, 64), dtype=torch.float32)

        return item

    def _normalize(self, data):
        # Min-max normalization to [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (
            max_val - min_val + 1e-8
        )  # Add small epsilon to avoid division by zero


def collate_fn(batch):
    # Process parameters and labels
    pp = torch.stack([item["features_pp"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    # AE features
    ae_features = [item["features_ae"].permute(1, 0) for item in batch]
    # ae_lengths = [x.shape[0] for x in ae_features]
    ae_padded = torch.nn.utils.rnn.pad_sequence(ae_features, batch_first=True)

    # Vibration features
    vib_features = [item["features_vib"].permute(1, 0) for item in batch]
    # vib_lengths = [x.shape[0] for x in vib_features]
    vib_padded = torch.nn.utils.rnn.pad_sequence(vib_features, batch_first=True)

    # Spectrograms
    ae_specs = pad_spectrograms([item["spec_ae"] for item in batch])
    vib_specs = pad_spectrograms([item["spec_vib"] for item in batch])

    return {
        "features_pp": pp,
        "features_ae": ae_padded,
        "features_vib": vib_padded,
        "spec_ae": ae_specs,
        "spec_vib": vib_specs,
        "label": labels,
    }


def pad_spectrograms(spectrograms):
    max_len = max(spec.shape[0] for spec in spectrograms)
    padded = []
    for spec in spectrograms:
        padding = max_len - spec.shape[0]
        padded.append(
            torch.cat(
                [spec, torch.zeros((padding, *spec.shape[1:]), dtype=spec.dtype)], dim=0
            )
        )
    return torch.stack(padded)


def get_collate_fn(input_type="all"):
    # Parse input type into components
    required = set(input_type.split("+")) if input_type != "all" else {"all"}
    # print(f"Required components: {required}")

    def collate_fn(batch):
        def pad_spectrograms(spectrograms):
            if not spectrograms:
                return torch.tensor([])
            spectrograms = [spec.squeeze() for spec in spectrograms]
            max_len = max(spec.shape[0] for spec in spectrograms)
            return torch.stack(
                [
                    torch.cat(
                        [
                            spec,
                            torch.zeros(
                                (max_len - spec.shape[0], *spec.shape[1:]), 
                                dtype=spec.dtype,
                            ),
                        ],
                        dim=0,
                    )
                    for spec in spectrograms
                ]
            )

        # Always present components
        batch_dict = {
            "features_pp": torch.stack(
                [item["features_pp"].squeeze() for item in batch]
            ),
            "label": torch.stack([item["label"].squeeze() for item in batch]),
        }

        # Conditionally process AE components
        if "features_ae" in batch[0]:
            ae_features = [item["features_ae"].squeeze().permute(1, 0) for item in batch]
            batch_dict["features_ae"] = torch.nn.utils.rnn.pad_sequence(
                ae_features, batch_first=True
            )
        
        if "spec_ae" in batch[0]:
            batch_dict["spec_ae"] = pad_spectrograms([item["spec_ae"] for item in batch])

        # Conditionally process VIB components
        if "features_vib" in batch[0]:
            vib_features = [item["features_vib"].squeeze().permute(1, 0) for item in batch]
            batch_dict["features_vib"] = torch.nn.utils.rnn.pad_sequence(
                vib_features, batch_first=True
            )
        
        if "spec_vib" in batch[0]:
            batch_dict["spec_vib"] = pad_spectrograms([item["spec_vib"] for item in batch])

        return batch_dict

    return collate_fn


def get_dataset(
    input_type: str = "all",
    dataset_mode: str = "classical",
    cpus=[logical_threads, 1],
    percentage=[0.6, 1.0],
    data_fraction: float = 1.0,
    chunk_index: int = 0,
    total_chunks: int = 1,
):
    """
    Get dataset with optional data fraction or chunking for memory efficiency.
    
    Args:
        input_type: Type of input data (e.g., 'all', 'ae_features', etc.)
        dataset_mode: Loading mode ('classical', 'chunked', 'ram', 'lazy')
        cpus: CPU configuration for parallel loading
        percentage: Percentage splits for loading
        data_fraction: Fraction of dataset to load (0.0 to 1.0). Default 1.0 (all data).
        chunk_index: Index of chunk to load (0-based). Used when total_chunks > 1.
        total_chunks: Total number of chunks to split dataset into. Default 1 (no chunking).
    """
    data = load_init_data()
    grinding_data = data["grinding_data"]
    if input_type not in allowed_input_types:
        raise ValueError(f"input_type must be one of {allowed_input_types}")
    
    if data_fraction <= 0 or data_fraction > 1:
        raise ValueError(f"data_fraction must be between 0 and 1, got {data_fraction}")
    
    if total_chunks < 1:
        raise ValueError(f"total_chunks must be >= 1, got {total_chunks}")
    
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise ValueError(f"chunk_index must be between 0 and {total_chunks-1}, got {chunk_index}")

    # For lazy mode, don't load all data at once
    if dataset_mode == "lazy":
        # Only load basic metadata, not the actual data
        dataset = LazyGrindingDataset(grinding_data, input_type)
        # Apply data fraction or chunking
        total_samples = len(dataset)
        
        if total_chunks > 1:
            # Use chunking
            chunk_size = total_samples // total_chunks
            start_idx = chunk_index * chunk_size
            end_idx = start_idx + chunk_size if chunk_index < total_chunks - 1 else total_samples
            indices = list(range(start_idx, end_idx))
            dataset = Subset(dataset, indices)
            print(f"Using chunk {chunk_index+1}/{total_chunks}: samples {start_idx}-{end_idx-1} ({len(indices)} samples)")
        elif data_fraction < 1.0:
            # Use data fraction
            subset_size = int(total_samples * data_fraction)
            indices = list(range(subset_size))
            dataset = Subset(dataset, indices)
            print(f"Using {subset_size}/{total_samples} samples ({data_fraction*100:.1f}% of data)")
        
        return dataset
    
    # For other modes, load data as before but more selectively
    # Only load what's needed based on input_type
    if "spec" in input_type or input_type == "all":
        print(f"Loading spectrogram data for {input_type}...")
        grinding_data._load_all_spec_data()
    else:
        # Don't load spec data if not needed
        grinding_data.spec_data = {}

    if "features" in input_type or "pp" in input_type or input_type == "all":
        print(f"Loading physics data for {input_type}...")
        grinding_data._load_all_physics_data()
    else:
        # Don't load physics data if not needed
        grinding_data.physical_data = {}

    dataset = GrindingDataset(grinding_data, input_type)
    
    # Apply data fraction or chunking
    total_samples = len(dataset)
    
    if total_chunks > 1:
        # Use chunking
        chunk_size = total_samples // total_chunks
        start_idx = chunk_index * chunk_size
        end_idx = start_idx + chunk_size if chunk_index < total_chunks - 1 else total_samples
        indices = list(range(start_idx, end_idx))
        dataset = Subset(dataset, indices)
        print(f"Using chunk {chunk_index+1}/{total_chunks}: samples {start_idx}-{end_idx-1} ({len(indices)} samples)")
    elif data_fraction < 1.0:
        # Use data fraction
        subset_size = int(total_samples * data_fraction)
        indices = list(range(subset_size))
        dataset = Subset(dataset, indices)
        print(f"Using {subset_size}/{total_samples} samples ({data_fraction*100:.1f}% of data)")

    if dataset_mode == "chunked":
        dataset = dataset
    elif dataset_mode == "ram":
        full_data = []
        # size_bytes = 0

        lenDataset = len(dataset)
        idx = getSubsetIdx(lenDataset, percentage, cpus)
        keys = list(idx.keys())
        for _c, _k in zip(cpus[:-1], keys[:-1]):
            _idx = idx[_k]
            ramDataLoader = DataLoader(
                Subset(dataset, np.array(_idx)),
                batch_size=1,
                shuffle=False,
                num_workers=int(_c),
                pin_memory=False,
                prefetch_factor=None if int(_c) == 0 else 1,
            )
            for i, item in tqdm(
                enumerate(ramDataLoader), desc=f"Loading {_k} data for {_c} threads"
            ):
                full_data.append(item)
                # size_bytes += sys.getsizeof(item)
            # print(f"Estimated size of data: {size_bytes:.2f} GB")
            print(
                f"Loading threads ({_c}) with remaining of data ({len(full_data)}/{len(dataset)})"
            )
            del ramDataLoader
            gc.collect()

        for i in tqdm(idx[keys[-1]], desc=f"Loading {keys[-1]} data for single thread"):
            full_data.append(dataset[i])
            # size_bytes += sys.getsizeof(dataset[i])

        size_bytes = sys.getsizeof(full_data)
        size_gb = size_bytes / (1024**3)
        dataset = MemoryDataset(full_data)
        print(f"Length of full_data: {len(full_data)}")
        print(f"Estimated size of full_data: {size_gb:.2f} GB")

    elif dataset_mode == "classical":
        dataset = dataset
    else:
        raise ValueError(
            f"dataset_mode must be one of ['classical', 'chunked', 'ram', 'lazy'], but got {dataset_mode}"
        )

    return dataset


def load_init_data():
    grinding_data = GrindingData(project_dir)
    grinding_data._load_all_physics_data()
    return {
        "dataDir_ae": dataDir_ae,
        "dataDir_vib": dataDir_vib,
        "grinding_data": grinding_data,
    }
