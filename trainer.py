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
from utils.MLUtils import TrainerBase, cv_trainer
from MyModels import GrindingPredictor
from MyDataset import allowed_input_types, MemoryDataset, get_dataset, get_collate_fn

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
    import warnings
    warnings.filterwarnings('ignore')
    from MyDataset import  percentage, cpus

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"


    parser = argparse.ArgumentParser(description='Distributed training job')
    parser.add_argument('--epochs', type=int, default= 100, help='Total epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=4, help=f'Input batch size on each device (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help=f'Learning rate, default:1e-5')
    # parser.add_argument('--gpu', default="0", type=lambda a: json.loads('['+a.replace(" ",",")+']'), help="List of values") 
    parser.add_argument('--gpu', default="cuda:0", type=str, help="gpu id or 'cpu' ")
    parser.add_argument('--fold_i', default="0", type=lambda a: json.loads('['+a.replace(" ",",")+']'), help="fold_i") 
    parser.add_argument('--folds', default="10", type=int, help="folds number") 
    parser.add_argument('--save_every', type=int, default=1, help=f'Save every 1 steps')
    parser.add_argument('--model_name', type=str, default='SmartGrinding', help=f'The model name')
    parser.add_argument('--input_type', type=str, default='all', help=f'The input type')
    parser.add_argument('--repeat', type=int, default=10, help=f'Repeat Times for Cross validation, default 1, no repeat')
    parser.add_argument('--test', type=bool, default=False, help=f'Go through the dataset without training, default False')
    parser.add_argument('--num_workers', type=int, default=4, help=f'Worker number in the dataloader, default:4')
    parser.add_argument('--verbose_interval', type=int, default=2, help=f'Verbose interval, default:2')
    parser.add_argument('--ram_margin', type=int, default=0.2, help=f'RAM margin, default:0.2')
    parser.add_argument('--dataset_mode', type=str, default="classical", help=f'Dataset mode, options: [classical, chunked, ram], default:classical')
    
    args = parser.parse_args()
    if args.input_type not in allowed_input_types:
        raise ValueError(f"input_type must be one of {allowed_input_types}")

    print(f"\n============= Settings =============")
    print(f"Processed in GPU: {args.gpu}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Save every {args.save_every} epoch(s)")
    print(f"Model name: {args.model_name}")
    print(f"Input type: {args.input_type}")
    print(f"Cross validation folds: {args.folds}")
    print(f"Repeat times: {args.repeat}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Verbose interval: {args.verbose_interval}")
    print(f"Random access memory margin: {args.ram_margin}")
    print(f"Dataset model: {args.dataset_mode}")
    print(f"============= Settings =============\n")

    _time = time.time()
    cpus = [32, 16, 2, 1]
    percentage=[0.6, 0.8, 0.90, 1]
    dataset = get_dataset(input_type=args.input_type, dataset_mode=args.dataset_mode, cpus=cpus, percentage=percentage)
    collate_fn = get_collate_fn(input_type=args.input_type)
    gc.collect()
    print(f"Dataset loaded in {time.time()-_time:.2f} seconds")
    print(f"Dataset size: {sys.getsizeof(dataset)}")
    print("*" * 50)

    model_name = args.model_name

    model = GrindingPredictor(interp=False,input_type=args.input_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    cv_trainer(
        dataset = dataset,
        Trainer = Trainer, 
        folds=args.folds,
        repeat=args.repeat,
        model=model,
        input_type = args.input_type,
        optimizer=optimizer,
        criterion=None,
        collate_fn=collate_fn,
        model_name=model_name, 
        gpu=args.gpu,
        save_every=args.save_every, 
        epochs=args.epochs,
        batch_size = args.batch_size, 
        fold_i = args.fold_i, 
        num_workers = args.num_workers, 
        test = args.test, 
        task_type='regression', 
        verbose_interval=args.verbose_interval,
        safety_ram_margin=args.ram_margin,
        dataset_mode=args.dataset_mode
        )

####