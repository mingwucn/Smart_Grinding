import sys
import os
import gc
from natsort import natsorted
# import pickle
import dill as pickle
import subprocess
import torch
import torchvision
import csv
import datetime
import time
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, StandardScaler,LabelEncoder

sys.path.append("./utils")
from utils.MLUtils import fade_in_out, standardize_tensor, CylinderDataset,LCVDataset, getKFoldCrossValidationIndexes, train_log, transform_ft, dataset_by_cross_validation, labels_by_classes, get_current_fold_and_hist_line_wised, LPBFDataset, TrainerBase, cv_trainer
from utils.InterfaceDeclaration import LPBFPointData,LPBFData
from utils.MLModels import SVMModel, CNN_Base_1D_Model, ResNet15_1D_Model

from MyModels import GrindingPredictor
from MyDataset import get_dataset

def collate_fn(batch):
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
    # Process parameters and labels
    pp = torch.stack([item['features_pp'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # AE features
    ae_features = [item['features_ae'].permute(1,0) for item in batch]
    ae_lengths = [x.shape[0] for x in ae_features]
    ae_padded = torch.nn.utils.rnn.pad_sequence(ae_features, batch_first=True)
    
    # Vibration features
    vib_features = [item['features_vib'].permute(1,0) for item in batch]
    vib_lengths = [x.shape[0] for x in vib_features]
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
        'lengths': (ae_lengths, vib_lengths)
    }

# Heritage from TrainerBase, only modify the _forward function
class Trainer(TrainerBase):
    def _forward(self, batch):
        # Example implementation:
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch['label'].to(self.device)
        
        outputs = self.model(inputs)
        
        return outputs, labels

if __name__ == "__main__":
    import argparse
    import json
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    num_epochs = 100
    batch_size = 2

    parser = argparse.ArgumentParser(description='Distributed training job')
    parser.add_argument('--epochs', type=int, default= 100, help='Total epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=batch_size, help=f'Input batch size on each device (default: {batch_size})')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help=f'Learning rate, default:1e-5')
    # parser.add_argument('--gpu', default="0", type=lambda a: json.loads('['+a.replace(" ",",")+']'), help="List of values") 
    parser.add_argument('--gpu', default="0", type=str, help="gpu id or 'cpu' ")
    parser.add_argument('--fold_i', default="0", type=lambda a: json.loads('['+a.replace(" ",",")+']'), help="fold_i") 
    parser.add_argument('--folds', default="10", type=int, help="folds number") 
    parser.add_argument('--save_every', type=int, default=1, help=f'Save every 1 steps')
    parser.add_argument('--model_name', type=str, default='SmartGrinding', help=f'The model name')
    parser.add_argument('--repeat', type=int, default=10, help=f'Repeat Times for Cross validation, default 1, no repeat')
    parser.add_argument('--test', type=bool, default=False, help=f'Go through the dataset without training, default False')
    parser.add_argument('--num_workers', type=int, default=4, help=f'Worker number in the dataloader, default:4')
    parser.add_argument('--verbose_interval', type=int, default=2, help=f'Verbose interval, default:2')
    
    args = parser.parse_args()

    print(f"\n============= Settings =============")
    print(f"Processed in GPU: {args.gpu}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save every {args.save_every} epoch(s)")
    print(f"Model name: {args.model_name}")
    print(f"Cross validation folds: {args.folds}")
    print(f"Repeat times: {args.repeat}")
    print(f"Test mode: {args.test}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Verbose interval: {args.verbose_interval}")
    print(f"============= Settings =============\n")

    dataset = get_dataset()
    model = GrindingPredictor(interp=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model_name = "SmartGrinding"
    cv_trainer(
        dataset = dataset,
        Trainer = Trainer, 
        folds=args.folds,
        repeat=args.repeat,
        model=model,
        optimizer=optimizer,
        loss_fn = None,
        collate_fn=collate_fn,
        model_name=model_name, 
        gpu=args.gpu,
        save_every=args.save_every, 
        epochs=args.epochs,
        batch_size = args.batch_size,
        fold_i = args.fold_i, 
        num_workers = args.num_workers, 
        test = args.test,
        task_type='classification'
        )

