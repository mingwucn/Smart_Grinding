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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, StandardScaler,LabelEncoder

sys.path.append("./utils")
from utils.MLUtils import fade_in_out, standardize_tensor, CylinderDataset,LCVDataset, getKFoldCrossValidationIndexes, train_log, transform_ft, dataset_by_cross_validation, labels_by_classes, get_current_fold_and_hist_line_wised, LPBFDataset
from utils.InterfaceDeclaration import LPBFPointData,LPBFData
from utils.MLModels import SVMModel, CNN_Base_1D_Model, ResNet15_1D_Model

def transform_pad(maximum_size,fad_in_out_length=16):
    t = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.nn.functional.pad(torch.tensor(data), (0, maximum_size - data.shape[-1])),
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        fade_in_out(data,fad_in_out_length),
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.tensor(data),
                    ),
                ])
    return t

def get_dataset(roi_radius=5, cube_i=2):
    project_name = ["MuSIC", "MaPS", "MuSIC_EXP1"]
    if os.name == "posix":
        data_dir = subprocess.getoutput("echo $DATADIR")
    elif os.name == "nt":
        data_dir = subprocess.getoutput("echo %datadir%")
    music_dir = os.path.join(data_dir, "MuSIC")
    if not os.path.exists(music_dir):
        project_name[0] = "2024-MUSIC"
    daq_dir = os.path.join(data_dir, *project_name, "Acoustic Monitoring")
    lmq_dir = os.path.join(data_dir, *project_name, "LMQ Monitoring")
    ct_img_dir = os.path.join(data_dir, *project_name,"CT_Cube3","Aligned_AN2")
    label_dir = os.path.join("../","lfs","point_wise_labels")
    del music_dir

    sampling_rate_daq: int = int(1.25 * 1e6)
    sampling_rate_lmq: int = int(0.1 * 1e6)
    tdms_daq_list = natsorted(
        [i for i in os.listdir(daq_dir) if i.split(".")[-1] == "tdms"]
    )
    bin_lmq_list = natsorted([i for i in os.listdir(lmq_dir) if i.split(".")[-1] == "bin"])
    lmq_channel_name = [
        "Vector ID",
        "meltpooldiode",
        "X Coordinate",
        "Y Coordinate",
        "Laser power",
        "Spare",
        "Laser diode",
        "Varioscan(focal length)",
    ]
    process_regime = [
        [0,60,     "Base"], # ignored
        [61, 130,  "GP"], # Gas Pore, Unstable keyhole, very hard to predict
        [131, 200, "NP"], # No Pore, Nominal Parameter
        [201, 270, "RLoF"], # Random lack of fusion
        [271, 340, "LoF"] # lack of Fusion, most easy to predict
    ]

    laser_power_setting = [
        [0,60,     "165"], # ignored
        [61, 130,  "110"], # Gas Pore, Unstable keyhole, very hard to predict
        [131, 200, "180"], # No Pore, Nominal Parameter
        [201, 270, "300"], # Random lack of fusion
        [271, 340, "165"] # lack of Fusion, most easy to predict
    ]

    scanning_speed_setting = [
        [0,60,     "900"], # ignored
        [61, 130,  "900"], # Gas Pore, Unstable keyhole, very hard to predict
        [131, 200, "900"], # No Pore, Nominal Parameter
        [201, 270, "900"], # Random lack of fusion
        [271, 340, "900"] # lack of Fusion, most easy to predict
]

    label_dir = os.path.join(data_dir, *project_name, "intermediate", f"roi_radius{roi_radius}")
    with open(os.path.join(label_dir, f"line_wised_cube{cube_i}.pkl"), "rb") as fp:
        lpbf = pickle.load(fp)

    lpbf_data = LPBFPointData(
    context_info={
        "cube_position": [lpbf.cube_i]*len(lpbf.laser_power),
        "laser_power": lpbf.laser_power,
        "start_coord": lpbf.start_coord,
        "end_coord":   lpbf.end_coord,
        "scanning_speed": lpbf.scanning_speed,
        "regime_info": lpbf.regime_info,
    },
    defect_labels=lpbf.line_labels,
    in_process_data={
        "microphone": lpbf.microphone,
        "AE": lpbf.ae,
        "photodiode": lpbf.photodiode,
        }
    )
    del lpbf
    gc.collect()
    
    sc_power = StandardScaler().fit(np.unique(lpbf_data.laser_power).astype(float).reshape(-1,1))
    sc_direction = StandardScaler().fit(np.unique(np.round(lpbf_data.print_vector[1])).astype(float).reshape(-1,1))
    le_speed = LabelEncoder().fit(np.asarray(lpbf_data.scanning_speed,dtype=str))
    le_region = LabelEncoder().fit(np.asarray(lpbf_data.regime_info,dtype=str))

    laser_power = sc_power.transform(np.asarray(lpbf_data.laser_power).astype(float).reshape(-1,1)).reshape(-1)
    print_direction = sc_direction.transform(np.asarray(np.round(lpbf_data.print_vector[1])).astype(float).reshape(-1,1)).reshape(-1)
    scanning_speed = le_speed.transform(np.asarray(lpbf_data.scanning_speed).astype(float))
    regime_info = le_region.transform(np.asarray(lpbf_data.regime_info,dtype=str))

    dataset = LPBFDataset(lpbf_data.cube_position,laser_power,lpbf_data.scanning_speed,regime_info,print_direction,lpbf_data.microphone, lpbf_data.AE, lpbf_data.defect_labels)
    return dataset

def setup(rank, gpu_ids):
    os.environ['NCCL_P2P_DISABLE'] = '1'
    # os.environ['NCCL_DEBUG'] = 'INFO' #DEBUG
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10086'
    dist.init_process_group(
        backend= "nccl", 
        init_method="env://",
        rank=rank, 
        world_size=len(gpu_ids),
        timeout=datetime.timedelta(seconds=5400),
        )
    torch.cuda.set_device(gpu_ids[rank])

def cleanup():
    dist.destroy_process_group()

class Trainer:
    def __init__(
        self,
        rank,
        gpu_ids,
        model: torch.nn.Module,
        num_epochs: int,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        checkpoint_name: str,
        save_every: int,
        test:bool=False,
        input_type='mic',
        output_type='regime',
    ) -> None:
        self.gpu_ids = gpu_ids
        self.rank = rank
        # self.local_rank = int(os.environ["LOCAL_RANK"])
        # self.global_rank = int(os.environ["RANK"])
        self.local_rank_gpu = gpu_ids[rank]
        self.model = model.to(self.local_rank_gpu)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = int(save_every)
        self.epochs_run = 0
        self.checkpoint_name = checkpoint_name
        self.max_epochs = num_epochs
        self.training_time = 0
        self.epoch_trained = 0
        self.test = test
        self.input_type = input_type
        self.output_type = output_type

        # if rank ==0:
        #     print(f"Save every {self.save_every} epoch(s)")

        if os.path.isfile(f"./lfs/weights/{checkpoint_name}.pt"):
            if rank == 0:
                print(f"Loading checkpoint from ./lfs/weights/{checkpoint_name}.pt...")
                print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
            self._load_snapshot()

        if len(gpu_ids)>1:
            # self.model = DDP(self.model, device_ids=[self.local_rank_gpu],find_unused_parameters=True) 
            # [!] Make sure there is no unused layer in the model
            self.model = DDP(self.model, device_ids=[self.local_rank_gpu])
        else:
            self.mode = model.to(self.local_rank_gpu)
        self.criterion_reg = nn.MSELoss()
        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_binary = nn.BCEWithLogitsLoss()
        self.transform = None

    def _save_snapshot(self, epoch):
        if len(self.gpu_ids)>1:
            state_key =  self.model.module.state_dict()
        else:
            state_key =  self.model.state_dict()
        snapshot = {
            "model_state_dict": state_key,
            'optimizer_state_dict': self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        # torch.save({
        #         'model_state_dict': self.model.eval().state_dict(),
        #         'optimizer_state_dict': self.optimizer.state_dict(),
        #         "EPOCHS_RUN": epoch,
        #         }, f"./lfs/weights/{checkpoint_name}.pt")
        torch.save(snapshot, f"./lfs/weights/{self.checkpoint_name}.pt")
        print(f"Epoch {epoch} | Training snapshot saved at ./lfs/weights/{self.checkpoint_name}.pt")

    def _load_snapshot(self):
        loc = f"cuda:{self.local_rank_gpu}"
        checkpoint_path = f"./lfs/weights/{self.checkpoint_name}.pt"
        snapshot = torch.load(checkpoint_path, map_location=loc, weights_only=True)

        _state_dict = snapshot["model_state_dict"]
        self.model.load_state_dict(_state_dict)
        self.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _load_snapshot_ddp(self):
        loc = f"cuda:{self.local_rank_gpu}"
        checkpoint_path = f"./lfs/weights/{self.checkpoint_name}.pt"
        snapshot = torch.load(checkpoint_path, map_location=loc, weights_only=True)

        # For multi GPUs
        if len(self.gpu_ids)>1:
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            _state_dict = OrderedDict()
            for k, v in snapshot["model_state_dict"].items():
                name = k[7:] # remove `module.`
                _state_dict[name] = v
        else:
            _state_dict = snapshot["model_state_dict"]
        self.model.load_state_dict(_state_dict)
        self.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_forward(self,data):
        """
        Only need to modify this for other datasets
        """
        _cube_position, _laser_power, _scanning_speed, _regime_info, _print_direction, _microphone, _ae, _defect_labels = data
        input_list = self.input_type.split("+")

        meata_list = []
        if 'mic' in input_list:
            _input = _microphone
        if 'ae' in input_list:
            _input = _ae
        if 'position' in input_list:
            meata_list.append(_cube_position.to(self.local_rank_gpu, dtype=torch.float64))
        if 'direction' in input_list:
            meata_list.append(_print_direction.to(self.local_rank_gpu))
        if 'laser_power' in input_list:
            meata_list.append(_laser_power.to(self.local_rank_gpu))
        if 'speed' in input_list:
            meata_list.append(_scanning_speed.to(self.local_rank_gpu))
        if 'regime' in input_list:
            meata_list.append(_regime_info.to(self.local_rank_gpu, dtype=torch.float64))
        if 'energy' in input_list:
            energy = torch.sum(_input ** 2,dim=1)
            meata_list.append(energy.to(self.local_rank_gpu))

        if self.output_type == 'position':
            labels = _cube_position.to(self.local_rank_gpu)
            criterion = self.criterion_class
        # if self.output_type == 'position':
        #     labels = _print_direction.to(self.local_rank_gpu)
        #     criterion = self.criterion_reg
        if self.output_type == 'laser_power':
            criterion = self.criterion_reg
            labels = _laser_power.to(self.local_rank_gpu)
        if self.output_type == 'speed':
            criterion = self.criterion_reg
            labels = _scanning_speed.to(self.local_rank_gpu)
        if self.output_type == 'regime':
            criterion = self.criterion_class
            labels = _regime_info.to(self.local_rank_gpu)
        if self.output_type == 'defect':
            criterion = self.criterion_binary
            labels = _defect_labels.to(self.local_rank_gpu)
        if self.output_type == 'direction':
            criterion = self.criterion_class
            labels = _print_direction.to(self.local_rank_gpu)

        time_series = (transform_ft()(standardize_tensor(_input)).to(self.local_rank_gpu))

        logits = self.model(time_series,meata_list)
        probs = torch.sigmoid(logits)
        predictions = torch.argmax(probs,axis=1).clone().int().detach().cpu()

        if self.output_type == "defect":
            loss = criterion(logits.squeeze(), labels.to(logits.dtype))
        else:
            loss = criterion(logits,labels)

        hit_number = torch.sum(predictions == labels.clone().detach().cpu())
        total_number = labels.numel()
        return hit_number, total_number, loss

    def _run_batch(self, i, data, epoch):
        self.test_hit_number = 0
        self.test_hit_number = 0
        self.optimizer.zero_grad()

        hit_number, total_number, loss = self._run_forward(data)

        self.hit_number += hit_number
        self.total_number += total_number

        loss.backward()
        self.optimizer.step()
        self.train_loss += loss.item()
        self.train_number += 1
        self.train_acc = self.hit_number/self.total_number

        # if i % 20 ==0:
            # print(f"[GPU{self.local_rank_gpu}] | Epoch [{epoch}/{self.max_epochs}:{i + 1}/{len(self.train_data)}] loss: {self.train_loss / self.train_number:.3f}, Acc: {self.train_acc:.2%}")

    def _run_batch_test(self, i, data, epoch):
        """
        Only need to modify this for other datasets
        """

        self.test_hit_number = 0
        self.test_hit_number = 0
        self.optimizer.zero_grad()

        mic, power, velocity, direction, if_defect_LoF, if_defect_KH = data
        energy = torch.sum(mic ** 2,dim=1)
        inputs = (transform_ft()(standardize_tensor(mic)).to(self.local_rank_gpu),
                  energy.to(self.local_rank_gpu),
                  power.to(self.local_rank_gpu),
                  velocity.to(self.local_rank_gpu),
                  direction.to(self.local_rank_gpu),
                  )

        logits = self.model(*inputs)
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).clone().int().detach().cpu()

        loss = self.criterion_binary(logits[:,0],if_defect_LoF.to(f"cuda:{self.local_rank_gpu}"))+ self.criterion_binary(logits[:,1],if_defect_KH.to(f"cuda:{self.local_rank_gpu}"))

        _labels = torch.stack([if_defect_LoF, if_defect_KH],dim=1).detach().cpu()
        self.hit_number += torch.sum(predictions == _labels)
        self.total_number += _labels.numel()

        self.train_loss += loss.item()
        self.train_number += 1
        self.train_acc = self.hit_number/self.total_number

        # if i % 20 ==0:
        #     print(f"[GPU{self.local_rank_gpu}] | Epoch [{epoch}/{self.max_epochs}:{i + 1}/{len(self.train_data)}] loss: {self.train_loss / self.train_number:.3f}, Acc: {self.train_acc:.2%}")

    def _run_epoch(self, epoch):
        print(f"[GPU{self.local_rank_gpu}] Epoch {epoch} | Steps: {len(self.train_data)}")
        if len(self.gpu_ids)>1:
            self.train_data.sampler.set_epoch(epoch)
            self.test_data.sampler.set_epoch(epoch)
        self.train_number = 0
        self.total_number = 0
        self.hit_number = 0
        for i, data in enumerate(self.train_data):
            # data = data.to(self.local_rank)
            if self.test==True:
                self._run_batch_test(i, data, epoch)
            else:
                self._run_batch(i, data, epoch)

    def _evaluate_model(self, dataloader):
        """
        Evaluates the model on the provided dataset.
        Args:
            model (nn.Module): The model to evaluate.
            dataloader (DataLoader): DataLoader for the test set.
            criterion_reg (nn.Module): Loss function for regression.
            criterion_class (nn.Module): Loss function for classification.
        Returns:
            float: Average loss over the test set.
        """
        print(f"[GPU{self.local_rank_gpu}] | Testing steps: {len(self.test_data)}")

        model = self.model
        rank = self.rank
        
        model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        total = 0

        with torch.no_grad():
            for data in dataloader:
                hit_number, total_number, loss = self._run_forward(data)

                self.test_hit_number += hit_number
                self.test_total_number += total_number
                total_loss += loss

        if len(self.gpu_ids)>1:
            # Aggregate results across all processes dist.reduce(torch.tensor(total_loss).cuda(self.local_rank_gpu), dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(torch.tensor(total).cuda(self.local_rank_gpu), dst=0, op=dist.ReduceOp.SUM)
        # Only the process with rank 0 will print the final results
        if rank == 0:
            avg_loss = total_loss / total
            return avg_loss, self.test_hit_number/self.test_total_number

    def train(self):
        for epoch in range(self.epochs_run, self.max_epochs):
            self.model.train()
            start_time = time.time()
            self.train_loss = 0.0
            self.train_acc = 0.0
            self.train_number = 0 
            self.total_number = 0
            self.hit_number = 0
            self.test_hit_number = 0
            self.test_total_number = 0
            self._run_epoch(epoch)

            # Save in the middle
            if self.rank == 0 and (epoch % self.save_every) == 0:
                if self.test==False:
                    self._save_snapshot(epoch)

            if len(self.gpu_ids)>1:
                # Calculate train_loss across different gpus
                dist.reduce(torch.tensor(self.train_loss).cuda(self.local_rank_gpu), dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(torch.tensor(self.train_number).cuda(self.local_rank_gpu), dst=0, op=dist.ReduceOp.SUM)

            test_res = self._evaluate_model(self.test_data)
            if self.rank == 0: 
                test_loss, test_acc = test_res
                print('Testing...')
                if self.test==False:
                    train_log(f"./lfs/weights/hist/{self.checkpoint_name}.csv",epoch=epoch,train_loss=self.train_loss/self.train_number,train_acc =self.train_acc,test_loss=test_loss,test_acc=test_acc)

                print(f"[GPU{self.local_rank_gpu}] | Epoch [{epoch}/{self.max_epochs}], Train loss: {self.train_loss/self.train_number}, Train ACC: {self.train_acc:.3%} Test Loss: {test_loss:.3f}, Test ACC: {test_acc:.3%}")

            self.train_loss = 0.0

            end_time = time.time()
            epoch_duration = end_time - start_time 
            self.training_time += epoch_duration
            self.epoch_trained += 1
            average_training_time = self.training_time/self.epoch_trained
            remaining_time = (self.max_epochs-epoch)*(average_training_time)

            print(f"=============")
            print(f"[GPU{self.local_rank_gpu}]| Epoch {epoch} completed in {epoch_duration:.2f} seconds")
            print(f"[GPU{self.local_rank_gpu}]| Average training time for each epoch: {average_training_time:.2f} seconds")
            print(f"[GPU{self.local_rank_gpu}]| Remaining time: {datetime.timedelta(seconds=remaining_time)} ")
            print(f"=============")

        if self.rank == 0 and (self.epochs_run!=self.max_epochs): 
            self._save_snapshot(epoch+1)
            print("Finished Training")

def main_folds(rank, gpu_ids, model_name, dataset, num_epochs, batch_size, learning_rate, num_workers, folds, checkpoint_name, save_every, test, input_type, output_type,time_series_length):
    world_size = len(gpu_ids)
    if world_size >1:
        setup(rank, gpu_ids)
    train_idx, test_idx = folds

    # Dataset
    # dataset = get_dataset(roi_time,roi_radius)
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    # Create DataLoaders
    if len(gpu_ids)>1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset)
        train_loader = DataLoader(train_subset, sampler=train_sampler, batch_size=batch_size,num_workers=num_workers, pin_memory=True,prefetch_factor=6,shuffle=False,persistent_workers=True)
        test_loader =  DataLoader(test_subset, sampler=test_sampler, batch_size=batch_size,num_workers=num_workers, pin_memory=True,prefetch_factor=6,shuffle=False,persistent_workers=True)
    else:
        train_loader = DataLoader(train_subset, batch_size=batch_size,num_workers=num_workers, pin_memory=True,prefetch_factor=6,shuffle=True, persistent_workers=True)
        test_loader =  DataLoader(test_subset, batch_size=batch_size,num_workers=num_workers, pin_memory=True,prefetch_factor=6,shuffle=True, persistent_workers=True)


    # ## Test
    # for data in train_loader:
    #     print(f"[{rank}] len {len(data)}")
    # ## Test

    meta_data_size = len(input_type.split("+"))-1
    # time_series_length = 5888
    print(f"Input type: {input_type}")
    print(f"Output type: {output_type}")
    print(f"Lenght of time series data: {time_series_length}")
    print(f"Lenght of context data: {meta_data_size}")

    num_classes = 1
    if output_type == "regime":
        num_classes = 4
    if output_type == "defect":
        num_classes = 1
    if output_type == "direction":
        num_classes = 5
    if output_type == "position":
        num_classes = 5

    if model_name == "SVM":
        if rank ==0:
            print("Using SVM")
        model = SVMModel(time_series_length+meta_data_size,num_classes=num_classes).double()
    if model_name == "CNN":
        if rank ==0:
            print("Using CNN")
        model = CNN_Base_1D_Model(time_series_length=time_series_length, meta_data_size=meta_data_size,num_classes=num_classes).double()
    if model_name == "Res15":
        if rank ==0:
            print("Using Res15")
        model = ResNet15_1D_Model(time_series_length=time_series_length, meta_data_size=meta_data_size,num_classes=num_classes).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(rank, gpu_ids, model, num_epochs, train_loader, test_loader, optimizer, checkpoint_name, save_every, test, input_type, output_type)
    trainer.train() 

    if world_size >1:
        cleanup()

if __name__ == "__main__":
    import argparse
    import json
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    num_epochs = 100
    batch_size = int(40*0.5)

    parser = argparse.ArgumentParser(description='Distributed training job')
    parser.add_argument('--epochs', type=int, default= num_epochs, help='Total epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=batch_size, help=f'Input batch size on each device (default: {batch_size})')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help=f'Learning rate, default:1e-5')
    parser.add_argument('--gpu', default="0", type=lambda a: json.loads('['+a.replace(" ",",")+']'), help="List of values") 
    parser.add_argument('--fold_i', default="0", type=lambda a: json.loads('['+a.replace(" ",",")+']'), help="fold_i") 
    parser.add_argument('--folds', default="5", type=int, help="folds number") 
    parser.add_argument('--save_every', type=int, default=1, help=f'Save every 1 steps')
    parser.add_argument('--model_name', type=str, default='CNN', help=f'The model name')
    # parser.add_argument('--roi_time', type=int, default=10, help=f'ROI time, unit(ms)')
    parser.add_argument('--roi_radius', type=int, default=5, help=f'ROI radius, unit(pixel)')
    parser.add_argument('--repeat', type=int, default=1, help=f'Repeat Times for Cross validation, default 1, no repeat')
    parser.add_argument('--test', type=bool, default=False, help=f'Go through the dataset without training, default False')
    parser.add_argument('--num_workers', type=int, default=4, help=f'Worker number in the dataloader, default:4')
    parser.add_argument('--input_type',  type=str, default='mic+energy', help=f'Input type')
    parser.add_argument('--output_type', type=str, default='regime', help=f'Output type')
    # parser.add_argument('--time_series_length', type=int, default='5888', help=f'The maximum length of time series inputs')
    
    args = parser.parse_args()

    print(f"\n============= Settings =============")
    print(f"Processed in GPU: {args.gpu}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save every {args.save_every} epoch(s)")
    print(f"Input type: {args.input_type}")
    print(f"Output type {args.output_type}")
    # print(f"ROI time: {args.roi_time}")
    print(f"ROI radius {args.roi_radius}")
    # print(f"Maximum time series length {args.time_series_length}")
    print(f"============= Settings =============\n")
    if int(args.fold_i[0]) == 0:
        folds_i = range(int(args.folds*args.repeat))
        print(f"Cross validation on all the {args.folds}*{args.repeat} folds")
    else:
        folds_i = np.array(args.fold_i)-1
        print(f"Cross validation on fold [{args.fold_i}]/{args.folds}*{args.repeat}")

    ## Restore form the snapshot
    current_fold,trained_epochs = get_current_fold_and_hist_line_wised(args.model_name,args.input_type,args.output_type,args.folds,args.roi_radius,args.epochs)
    
    if (trained_epochs == args.epochs) & (current_fold+1==int(args.folds))==True:
        print("Finished current folds\n")
    else:
        # print(f"ROI duration {args.roi_time}(ms)")
        # print(f"ROI radius {args.roi_radius}")

        # labels = np.load(os.path.join(f"lfs","intermediate",f"labels_roi_time{args.roi_time}_roi_radius{args.roi_radius}.npy"))
        # len_dataset = len(labels)

        # print(f"Len dataset {len_dataset}")
        # normal_labels, err_labels, scaler_power, scaler_speed, scaler_direction = labels_by_classes(args.roi_time,args.roi_radius)
        # ratio = normal_labels.shape[0]/err_labels.shape[0]
        # print(f"the abnomal ratio {1/ratio:.2%}, will create {int(ratio)-1} replicates")
        # total_labels = np.vstack([normal_labels,err_labels])
        # print(f"Total segments: {total_labels.shape[0]}")

        # err_k_folds = getKFoldCrossValidationIndexes(len(err_labels), args.folds, seed=10086)
        # nor_k_folds = getKFoldCrossValidationIndexes(len(normal_labels), args.folds, seed=10086)
        dataset = get_dataset(roi_radius=args.roi_radius)
        len_dataset = len(dataset)

        print(f"Len dataset {len_dataset}")
        all_k_folds = getKFoldCrossValidationIndexes(len_dataset, args.folds, seed=10086)

        for _i in range(args.repeat): 
            # err_k_folds += getKFoldCrossValidationIndexes(len(err_labels), args.folds, _i)
            all_k_folds += getKFoldCrossValidationIndexes(len_dataset, args.folds, _i)

        training_time = 0
        trained_folds = 0
        for fold_i in folds_i[current_fold:]:
            start_time = time.time()

            print(f"Fold [{fold_i}/{args.folds} * {args.repeat}]")

            # _err_labels_train = err_labels[err_k_folds[fold_i][0]]
            # _err_labels_train = np.repeat(_err_labels_train, int(ratio)-1, axis=0)
            # _nor_labels_train = normal_labels[nor_k_folds[fold_i][0]]
            # train_labels = np.vstack([_err_labels_train,_nor_labels_train])
            # all_train_labels = err_labels[err_k_folds[fold_i][0]]

            # _err_labels_test = err_labels[err_k_folds[fold_i][1]]
            # _err_labels_test = np.repeat(_err_labels_test, int(ratio)-1, axis=0)
            # _nor_labels_test = normal_labels[nor_k_folds[fold_i][1]]
            # test_labels = np.vstack([_err_labels_test,_nor_labels_test])

            # rng = np.random.default_rng(seed=fold_i)
            # rng.shuffle(train_labels,axis=0)
            # rng.shuffle(test_labels,axis=0)


            # checkpoint_name = f'{args.model_name}_classification_input_{args.input_type}_output_{args.output_type}_roi_time{args.roi_time}_roi_radius{args.roi_radius}_fold{fold_i}_of_folds{args.folds}'

            # checkpoint_name = f'{args.model_name}_classification_window{args.time_series_length}_stride{args.stride}_fold{fold_i}_of_folds{args.folds}'
            checkpoint_name = f'Line_wised_{args.model_name}_{args.input_type}_{args.output_type}_roi_radius{args.roi_radius}_fold{fold_i}_of_folds{args.folds}'

            # dataset, train_idx, test_idx = dataset_by_cross_validation(args.roi_time,train_labels,test_labels,total_labels=total_labels, ram=True)
            
            # print(f"Total dataset {len(dataset)}: Train {len(train_labels)} | Test {len(test_labels)}")
            train_idx, test_idx = all_k_folds[fold_i]
            # max_length = dataset.max_length
            max_length = transform_ft()(torch.ones(1,dataset.max_length)).shape[1]
            if len(args.gpu)>1:
                mp.spawn(main_folds,
                    args = (args.gpu, args.model_name, dataset, args.epochs, args.batch_size,args.learning_rate,args.num_workers,  (train_idx, test_idx), checkpoint_name, args.save_every,args.test,args.input_type,args.output_type,max_length),
                    nprocs=len(args.gpu),
                    join=True)
            else:
                main_folds(0, args.gpu, args.model_name, dataset, args.epochs, args.batch_size,args.learning_rate,args.num_workers,  (train_idx, test_idx), checkpoint_name, args.save_every,args.test,args.input_type,args.output_type,max_length)

            end_time = time.time()
            fold_duration = end_time - start_time 

            training_time += fold_duration
            trained_folds += 1
            average_training_time = training_time/trained_folds
            remaining_time = (len(folds_i)-fold_i)*(average_training_time)

            print(f"Fold [{fold_i}/{args.folds} * {args.repeat}]")
            print(f"\n=============")
            print(f"Fold [{fold_i}/{args.folds} * {args.repeat}] completed in {fold_duration:.2f} seconds")
            print(f"Average training time for each fold: {average_training_time:.2f} seconds")
            print(f"Remaining time: {datetime.timedelta(seconds=remaining_time)} ")
            print(f"=============\n")
