import datetime
import gc
import math
import numpy as np
import librosa
import os
import csv
import torch
import torch.nn as nn
import itertools
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torch.utils.data.dataset import Subset
import torchvision
import torchvision.transforms
from scipy.interpolate import griddata
from tqdm import tqdm
global device
global device_ids
import time
import psutil
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score

from sklearn.utils.class_weight import compute_sample_weight
from collections import defaultdict
import sys

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#     num_of_gpus = torch.cuda.device_count()
#     print(f"number of gpus:{num_of_gpus}")
#     device = torch.device(f"cuda:3")

def one_hot_encode(y, num_classes):
    """
    Convert labels to one-hot encoding.
    Args:
        y (torch.Tensor): Labels with shape (batch_size,)
        num_classes (int): Number of classes
    Returns:
        torch.Tensor: One-hot encoded labels with shape (batch_size, num_classes)
    """
    y_onehot = torch.zeros((y.size(0), num_classes), device=y.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot

def standardize_array(arr):
    """
    Standardize a NumPy array.

    Parameters:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Standardized array.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    standardized_arr = (arr - mean) / std
    return standardized_arr

def standardize_tensor(arr):
    """
    Standardize a torch tensor.

    Parameters:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Standardized array.
    """
    # arr = torch.tensor(arr)
    DType = arr.dtype
    mean = torch.mean(arr)
    std = torch.std(arr)
    standardized_arr = (arr - mean) / std
    standardize_array = torch.tensor(standardized_arr, dtype=DType)
    return standardize_array

def split_dataset( dataset_size, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=910420):
    """
    Splits a dataset into training, validation, and test sets using NumPy.

    Args:
        dataset_size: The total size of the dataset.
        train_ratio: Ratio of the dataset to be used for training (default: 0.7).
        val_ratio: Ratio of the dataset to be used for validation (default: 0.15).
        test_ratio: Ratio of the dataset to be used for testing (default: 0.15).

        seed: Random seed for reproducibility (default: 910420).

    Returns:
        train_idx: A list of indices for the training set.
        valid_idx: A list of indices for the validation set.
        test_idx: A list of indices for the test set.
    """

    np.random.seed(seed)  # Set the random seed for reproducibility

    # Calculate the number of samples in each set
    num_train = int(dataset_size * train_ratio)
    num_val = int(dataset_size * val_ratio)
    num_test = int(dataset_size * test_ratio)

    # Ensure all indices are used and no overlap occurs (adjust ratios if needed)
    # assert (
    #     num_train + num_val + num_test == dataset_size
    # ), "Ratios must sum to 1. Adjust to avoid exceeding dataset size."

    # Permute all indices for random shuffling
    all_indices = np.arange(dataset_size,dtype=int)
    np.random.shuffle(all_indices)

    # Split indices into training, validation, and test sets
    train_idx = all_indices[:num_train]
    valid_idx = all_indices[num_train : num_train + num_val]
    test_idx = all_indices[num_train + num_val :]

    return list(train_idx), list(valid_idx), list(test_idx)

def train_model(model, dataloader, optimizer, test_loader=None, transform=None, num_epochs=10, save_model=None,log_path=None):
    # Define Losses and Optimizer
    criterion_reg = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    model.to(device)
    for epoch in range(num_epochs):
        print("Start training...")
        model.train()
        train_loss = 0.0
        for i, data in tqdm(enumerate(dataloader)):
            data_mic, label_power, label_powder, label_carrier, label_shielding, label_defect, label_exhausted = data
            regression_targets = torch.stack((label_power, label_powder, label_carrier, label_shielding),dim=1).to(device,dtype=torch.float)
            inputs = transform(data_mic.numpy()).to(device)

            optimizer.zero_grad()

            regression_output, class_output1, class_output2 = model(inputs)

            # One-hot encode classification targets
            class_target1 = one_hot_encode(label_defect.to(device, dtype=torch.int64), 3)  # Defect type
            class_target2 = one_hot_encode(label_exhausted.to(device, dtype=torch.int64), 2)  # Exhaust on off, binary classification

            # Compute loss
            loss_reg = criterion_reg(regression_output, regression_targets)
            loss_class1 = criterion_class(class_output1, class_target1)
            loss_class2 = criterion_class(class_output2, class_target2)

            loss = loss_reg*1e4 + loss_class1 + loss_class2

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            train_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {train_loss / 10:.3f}")
                train_loss = 0.0
        if save_model:
            if (epoch) % 10 == 0:
                torch.save(model.state_dict(), f"./lfs/weights/{save_model}.pt") 

        if test_loader:
            # Evaluate on test set
            test_loss = evaluate_model(model, test_loader, criterion_reg, criterion_class,transform=transform)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.3f}")
        if log_path:
            train_log(log_path,epoch=epoch,train_loss=train_loss,test_loss=test_loss)
    print("Finished Training")

def evaluate_model(model, dataloader, criterion_reg, criterion_class,transform=None):
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
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:

            data_mic, label_power, label_powder, label_carrier, label_shielding, label_defect, label_exhausted = data
            regression_targets = torch.stack((label_power, label_powder, label_carrier, label_shielding),dim=1).to(device,dtype=torch.float)
            inputs = transform(data_mic.numpy()).to(device)
            
            # Forward pass
            regression_output, class_output1, class_output2 = model(inputs)
            
            # One-hot encode classification targets
            class_target1 = one_hot_encode(label_defect.to(device, dtype=torch.int64), 3)  # Defect type
            class_target2 = one_hot_encode(label_exhausted.to(device, dtype=torch.int64), 2)  # Exhaust on off, binary classification
            
            # Compute loss
            loss_reg = criterion_reg(regression_output, regression_targets)
            loss_class1 = criterion_class(class_output1, class_target1)
            loss_class2 = criterion_class(class_output2, class_target2)
            
            loss = loss_reg*1e4 + loss_class1 + loss_class2
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_log(save_path,epoch,train_loss,train_acc, test_loss, test_acc):
    # Check if the file exists to determine whether to write headers
    file_exists = os.path.isfile(save_path)
    # Open the CSV file in append mode
    with open(save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write headers if the file is new
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
        writer.writerow([epoch + 1, float(train_loss), float(train_acc), float(test_loss), float(test_acc)])

def getKFoldCrossValidationIndexes(n, k, seed=1):
    """
    Perform k-fold cross-validation.
    Returns:
        List of tuples: Each tuple contains the indices of the training and testing sets.
    """
    fold_size = n // k
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))
        folds.append((train_indices, test_indices))
    return folds

def print_confusion_matrix(cm, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    print("\nConfusion Matrix:")
    header = "True \\ Pred | " + " ".join(f"{name:>5}" for name in class_names)
    print(header)
    print("-" * len(header))
    
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<11} |", end="")
        for val in row:
            print(f"{val:5}", end="")
        print()

def logSpectrogram(
    sampling_rate,
    data=None,
    n_fft=1024,
    hop_length=320,
    window_type="hann",
    fmin=0,
    fmax=None,
    display=False,
    save=False,
    ):
    """
    n_fft=1024 # Size of the Fast Fourier Transform (FFT), which will also be used as the window length

    hop_length=320 # Step or stride between windows. If the step is smaller than the window length, the windows will overlap
    window_type ='hann' # Specify the window type for FFT/STFT
    """
    y = data
    # Check for non-finite values
    y = standardize_array(y)
    y = np.nan_to_num(y)
    # Check for non-finite values

    # Calculate the spectrogram as the square of the complex magnitude of the STFT
    spectrogram = (
        np.abs(
            librosa.stft(
                y,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window_type,
            )
        )
        ** 2
    )

    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # display
    return spectrogram

def load_train_data(data_dir):
    # load the dataset
    with open(os.path.join('intermediate',f"indices"), "rb") as fp:
        indices = pickle.load(fp)
    with open(os.path.join('intermediate',f"power_labels"), "rb") as fp:
        power_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"powder_labels"), "rb") as fp:
        powder_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"carrier_labels"), "rb") as fp:
        carrier_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"speed_labels"), "rb") as fp:
        speed_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"shielding_labels"), "rb") as fp:
        shielding_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"exhaust_labels"), "rb") as fp:
        exhaust_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"defect_labels"), "rb") as fp:
        defect_labels = pickle.load(fp)
    file_name_list = ['test2','test3','test4','test5','test6','test7','test1','carrier1','carrier2','carrier3','gas','gas2','exhaust','idle']
    dataset = LCVDataset(power_labels,powder_labels,carrier_labels,shielding_labels,defect_labels,exhaust_labels,indices,file_name_list,data_dir)
    return dataset

def load_model(model_name, model):
    checkpoint_path = f"./lfs/weights/{model_name}.pt"
    snapshot = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(snapshot["model_state_dict"])
    # optimizer.load_state_dict(snapshot['optimizer_state_dict'])
    epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Loading model from {checkpoint_path}")
    return model

def load_model_epochs(checkpoint_name,model):
    loc = f"cuda"
    checkpoint_path = f"./lfs/weights/{checkpoint_name}.pt"
    snapshot = torch.load(checkpoint_path, map_location=loc, weights_only=True)
    epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Resuming training from snapshot at Epoch {epochs_run}")
    return epochs_run

def fade_in_out(data, fade_length):
    # Convert to numpy array for processing
    # Apply fade-in
    audio = data.clone()
    fade_in = torch.linspace(0, 1, fade_length)
    audio[:fade_length] *= fade_in
    # Apply fade-out
    fade_out = torch.linspace(1, 0, fade_length)
    audio[-fade_length:] *= fade_out
    return audio

def transform_pad(maximum_size,fad_in_out_length=16):
    return torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.nn.functional.pad(data, (0, maximum_size - data.shape[-1])),
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        fade_in_out(data,fad_in_out_length),
                    ),
                ])

def transform_ft():
    return torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.abs(torch.fft.fft(data))
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        data[:,:data.shape[-1]//2]
                    ),
                ]
            )

def transform_spec():
    return torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.nn.functional.pad(torch.tensor(data), (0, mic_maximum_size - data.shape[-1])),
                    ),

                    torchvision.transforms.Lambda(
                        lambda data:
                        fade_in_out(data,16),
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.stft(data)
                    ),
                ]
            )

class CylinderDataset(Dataset):
    def __init__(
        self,
        dataDir,
        labelArray,
        roi_time,
        on_pad = True,
        maximum_size=3840,
        fade_in_out_length = 16,
        
        transformation1 = None,
        transformation2 = None,
        transformation3 = None,
    ):
        """
        For LCV dataset
        - Inputs:
            Pre-constructed labels
                - layer index
                - start index of segmentation
                - laser power
                - moving speed
                - direction
                - if keyhole defect 
                - if lack of fusion defect

            DataDir

        - Outputs:
            - Regression:
                - Laser power
                - Powder feed rate
                - Carrier
                - Shielding

            - Classification:
                LoF defect (Binary)
                Keyhole defect (Binary)
        Args:
            microphone_data (ndarray): Microphone data inputs.
            laser_power (ndarray): Laser power values (regression).
            powder_feed_rate (ndarray): Powder feed rate values (regression).
            carrier (ndarray): Carrier values (regression).
            shielding (ndarray): Shielding values (regression).
            defect (ndarray): Defect classification labels (3 classes).
            exhausted (ndarray): Exhausted classification labels (binary).
        """
        self.dataDir = dataDir
        self.labels = labelArray
        self.roi_time = roi_time
        self.on_pad = on_pad
        self.maximum_size = maximum_size
        self.fade_in_out_length = fade_in_out_length
        
        self.transformation1 = transformation1
        self.transformation2 = transformation2
        self.transformation3 = transformation3

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label_i = self.labels[idx,:]
        layer_i = int(label_i[0])
        i_start = int(label_i[1])
        power=label_i[2]
        velocity=label_i[3]
        direction=label_i[4]
        if_defect_KH=label_i[5]
        if_defect_LoF=label_i[6]
        mic_path = os.path.join(self.dataDir,'intermediate',f"roi{self.roi_time}ms_layer{layer_i}_{i_start}.npy")
        # np.load(os.path.join(data_dir))
        mic = np.load(mic_path)
        if self.on_pad == True:
            t = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.nn.functional.pad(torch.tensor(data), (0, self.maximum_size - data.shape[-1])),
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        fade_in_out(data,self.fade_in_out_length),
                    ),
                ])
            mic = t(mic)
        
        if self.transformation1 != None:
            mic = self.transformation1(mic)
        if self.transformation2 != None:
            mic = self.transformation2(mic)
        if self.transformation3 != None:
            mic = self.transformation3(mic)

        return (
            mic,
            power,
            velocity,
            direction,
            if_defect_LoF,
            if_defect_KH,
        )

class CylinderDataset_RAM(Dataset):
    def __init__(
        self,
        dataDir,
        labelArray,
        roi_time,
        mic_list,
        total_labels,
        on_pad = True,
        maximum_size=3840,
        fade_in_out_length = 16,
        
        transformation1 = None,
        transformation2 = None,
        transformation3 = None,
    ):
        """
        For LCV dataset
        - Inputs:
            Pre-constructed labels
                - layer index
                - start index of segmentation
                - laser power
                - moving speed
                - direction
                - if keyhole defect 
                - if lack of fusion defect

            DataDir

        - Outputs:
            - Regression:
                - Laser power
                - Powder feed rate
                - Carrier
                - Shielding

            - Classification:
                LoF defect (Binary)
                Keyhole defect (Binary)
        Args:
            microphone_data (ndarray): Microphone data inputs.
            laser_power (ndarray): Laser power values (regression).
            powder_feed_rate (ndarray): Powder feed rate values (regression).
            carrier (ndarray): Carrier values (regression).
            shielding (ndarray): Shielding values (regression).
            defect (ndarray): Defect classification labels (3 classes).
            exhausted (ndarray): Exhausted classification labels (binary).
        """
        self.dataDir = dataDir
        self.labels = labelArray
        self.roi_time = roi_time
        self.on_pad = on_pad
        self.maximum_size = maximum_size
        self.fade_in_out_length = fade_in_out_length
        self.total_labels = total_labels
        
        self.transformation1 = transformation1
        self.transformation2 = transformation2
        self.transformation3 = transformation3

        self.mic_list =mic_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label_i = self.labels[idx,:]
        layer_i = int(label_i[0])
        i_start = int(label_i[1])
        power=label_i[2]
        velocity=label_i[3]
        direction=label_i[4]
        if_defect_KH=label_i[5]
        if_defect_LoF=label_i[6]
        # np.load(os.path.join(data_dir))
        mic_idx = np.where((self.total_labels[:,0]==layer_i) & (self.total_labels[:,1]==i_start))[0][0]
        mic = self.mic_list[mic_idx]
        if self.on_pad == True:
            t = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.nn.functional.pad(torch.tensor(data), (0, self.maximum_size - data.shape[-1])),
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        fade_in_out(data,self.fade_in_out_length),
                    ),
                ])
            mic = t(mic)
        
        if self.transformation1 != None:
            mic = self.transformation1(mic)
        if self.transformation2 != None:
            mic = self.transformation2(mic)
        if self.transformation3 != None:
            mic = self.transformation3(mic)

        return (
            mic,
            power,
            velocity,
            direction,
            if_defect_LoF,
            if_defect_KH,
        )

class LCVDataset(Dataset):
    def __init__(
        self,
        laser_power,
        powder_feed_rate,
        carrier,
        shielding,
        defect,
        exhausted,
        indices,
        file_name_list,
        data_dir,
    ):
        """
        For LCV dataset
        - Inputs:
            Microphone data

        - Outputs:
            - Regression:
                - Laser power
                - Powder feed rate
                - Carrier
                - Shielding

            - Classification:
                Defect (3-classes)
                Exhausted (Binary)
        Args:
            microphone_data (ndarray): Microphone data inputs.
            laser_power (ndarray): Laser power values (regression).
            powder_feed_rate (ndarray): Powder feed rate values (regression).
            carrier (ndarray): Carrier values (regression).
            shielding (ndarray): Shielding values (regression).
            defect (ndarray): Defect classification labels (3 classes).
            exhausted (ndarray): Exhausted classification labels (binary).
        """
        self.laser_power = laser_power
        self.powder_feed_rate = powder_feed_rate
        self.carrier = carrier
        self.shielding = shielding
        self.defect = defect
        self.exhausted = exhausted
        self.data_dir = data_dir
        self.file_name_list = file_name_list
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        id_file, id_start_i = self.indices[idx]
        _mic = np.load(
            os.path.join(
                self.data_dir,
                "intermediate",
                f"{self.file_name_list[id_file]}-{id_start_i}.npy",
            )
        )

        return (
            _mic,
            self.laser_power[idx],
            self.powder_feed_rate[idx],
            self.carrier[idx],
            self.shielding[idx],
            self.defect[idx],
            self.exhausted[idx],
        )

def dataset_by_cross_validation(roi_time, train_labels, test_labels, total_labels, ram=False):
    import subprocess
    import pickle
    project_name = ["MuSIC", "FlandersMake","Cylinder14"]

    if os.name == "posix":
        data_dir = subprocess.getoutput("echo $DATADIR")
    elif os.name == "nt":
        data_dir = subprocess.getoutput("echo %datadir%")
    music_dir = os.path.join(data_dir, "MuSIC")
    if not os.path.exists(music_dir):
        project_name[0] = "2024-MUSIC"
    data_dir = os.path.join(data_dir, *project_name)
    del music_dir

    labels = np.vstack([train_labels,test_labels])
    train_idx = np.arange(0,len(train_labels),1,dtype=int)
    test_idx = np.arange(len(train_labels),len(train_labels)+len(test_labels),1,dtype=int)
    if ram == False:
        dataset = CylinderDataset(dataDir=data_dir,labelArray=labels,on_pad =True,  roi_time=roi_time)
    else:
        with open(os.path.join(data_dir,'intermediate',f"roi{roi_time}ms_mic_list"), 'rb') as handle:
            mic_list = pickle.load(handle)
        dataset = CylinderDataset_RAM(dataDir=data_dir,labelArray=labels,on_pad =True, mic_list=mic_list, total_labels=total_labels, roi_time=roi_time)
        
    return dataset, train_idx, test_idx

def labels_by_classes(roi_time, roi_radius):
    import subprocess
    import pickle
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    project_name = ["MuSIC", "FlandersMake","Cylinder14"]

    if os.name == "posix":
        data_dir = subprocess.getoutput("echo $DATADIR")
    elif os.name == "nt":
        data_dir = subprocess.getoutput("echo %datadir%")
    music_dir = os.path.join(data_dir, "MuSIC")
    if not os.path.exists(music_dir):
        project_name[0] = "2024-MUSIC"
    data_dir = os.path.join(data_dir, *project_name)
    del music_dir

    start_layer = 33
    end_layer = 481
    shell_exception_layers = [114, 208,423,476]
    errlayers = [60 , 61 , 62 , 90 , 120, 150, 180, 210, 240, 270, 300, 330 , 360 , 390, 420, 450 , 470, 471, 472]
    normal_layers = np.arange(end_layer+1)

    _mask = np.ones(len(normal_layers), dtype=bool)
    _mask[shell_exception_layers] = False
    _mask[:start_layer] = False

    _mask = np.ones(len(normal_layers), dtype=bool)
    _mask[errlayers] = False
    _mask[:start_layer] = False
    normal_layers = normal_layers[_mask,...]
    normal_labels = []
    for i in normal_layers:
        with open(os.path.join('lfs/labels',f"roi_{roi_time}ms",f'window_results_layer_{i}'), 'rb') as handle:
            _label = pickle.load(handle)
        normal_labels+=_label
    normal_labels = np.array(normal_labels)

    err_labels = []
    for i in errlayers:
        with open(os.path.join('lfs/labels',f"roi_{roi_time}ms",f'window_results_layer_{i}_roi_radius{roi_radius}'), 'rb') as handle:
            _label = pickle.load(handle)
        err_labels+=_label
    err_labels = np.array(err_labels)
    labels = np.vstack([normal_labels,err_labels])

    p_set = np.unique(labels[:,2]).reshape(-1, 1)
    v_set = np.unique(labels[:,3]).reshape(-1, 1)

    scaler_power = StandardScaler()
    laser_power_standardized = scaler_power.fit_transform(p_set)
    scaler_speed = StandardScaler()
    laser_speed_standardized = scaler_speed.fit_transform(v_set)
    scaler_direction = MinMaxScaler(feature_range=(0, 1))
    _ = scaler_direction.fit_transform([[-180],[180]])
    
    err_labels[:,2] =     scaler_power.transform(err_labels[:,2].reshape(-1,1)).squeeze()
    err_labels[:,3] =     scaler_speed.transform(err_labels[:,3].reshape(-1,1)).squeeze()
    err_labels[:,4] = scaler_direction.transform(err_labels[:,4].reshape(-1,1)).squeeze()

    normal_labels[:,2] =     scaler_power.transform(normal_labels[:,2].reshape(-1,1)).squeeze()
    normal_labels[:,3] =     scaler_speed.transform(normal_labels[:,3].reshape(-1,1)).squeeze()
    normal_labels[:,4] = scaler_direction.transform(normal_labels[:,4].reshape(-1,1)).squeeze()
    return normal_labels, err_labels, scaler_power, scaler_speed, scaler_direction

def textFinder(text,start,end):
    start = text.find(start) + len(start)
    end = text.find(end)
    result = text[start:end]
    return result

def get_current_fold_and_hist_old(model_name,input_type,output_type,folds,rou_time, roi_radius,max_epochs):
    from MLModels import SVMModel, CNN_Base_1D_Model, ResNet15_1D_Model
    from natsort import natsorted
    import pandas as pd
    checkpoint_name0 = f'{model_name}_classification_input_{input_type}_output_{output_type}_roi_radius{roi_radius}_fold'
    # trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.pt'])
    # trained_weights_hist =natsorted([i for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.csv'])
    trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and checkpoint_name0 in i])
    trained_weights_hist =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and checkpoint_name0 in i])

    if len(trained_weights) >= 1:
        print(f"Resuming from {trained_weights[-1]}")
        trained_path = os.path.join(os.getcwd(),"lfs","weights",trained_weights[-1])
        # trained_weights,trained_weights_hist

        if model_name == "SVM":
            # if rank ==0:
            print("Using SVM")
            trained_model = SVMModel(1920+4).double()
        if model_name == "CNN":
            # if rank ==0:
            print("Using CNN")
            trained_model = CNN_Base_1D_Model(time_series_length=1920, meta_data_size=4).double()
        if model_name == "Res15":
            # if rank ==0:
            print("Using Res15")
            trained_model = ResNet15_1D_Model(time_series_length=1920, meta_data_size=4).double()

        trained_epochs = load_model_epochs(trained_weights[-1],trained_model)
        trained_hist = pd.read_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"),index_col=0)
        train_hist_drop = trained_hist.drop(trained_hist[trained_hist.index>trained_epochs].index)
        train_hist_drop.to_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"))
        # current_fold = get_numbers_from_string(trained_weights[-1])[-2]
        current_fold = int(textFinder(trained_weights[-1],"_fold","_of_folds"))
        print(f"Roll back the hist .csv file from trained epochs: [{trained_epochs}] from fold [{current_fold+1}/{folds}]")
        print(f"====================\n")
    
        return int(current_fold), int(trained_epochs)
    else:
        print(f"{folds}-Folds Cross Validation for {model_name}")
        print(f"====================\n")
        return 0,0

def get_current_fold_and_hist_clip(model_name, fold_i, folds, clip_length, max_epochs):
    from MLModels import SVMModel, CNN_Base_1D_Model, ResNet15_1D_Model
    from natsort import natsorted
    import pandas as pd
    checkpoint_name0 = f'{model_name}_classification_clip_length_{clip_length}_fold{fold_i}_of_folds{folds}'
    # trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.pt'])
    # trained_weights_hist =natsorted([i for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.csv'])
    trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and checkpoint_name0 in i])
    trained_weights_hist =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and checkpoint_name0 in i])

    if len(trained_weights) >= 1:
        print(f"Resuming from {trained_weights[-1]}")
        trained_path = os.path.join(os.getcwd(),"lfs","weights",trained_weights[-1])
        # trained_weights,trained_weights_hist

        if model_name == "SVM":
            # if rank ==0:
            print("Using SVM")
            trained_model = SVMModel(1920+4).double()
        if model_name == "CNN":
            # if rank ==0:
            print("Using CNN")
            trained_model = CNN_Base_1D_Model(time_series_length=1920, meta_data_size=4).double()
        if model_name == "Res15":
            # if rank ==0:
            print("Using Res15")
            trained_model = ResNet15_1D_Model(time_series_length=1920, meta_data_size=4).double()

        trained_epochs = load_model_epochs(trained_weights[-1],trained_model)
        trained_hist = pd.read_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"),index_col=0)
        train_hist_drop = trained_hist.drop(trained_hist[trained_hist.index>trained_epochs].index)
        train_hist_drop.to_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"))
        # current_fold = get_numbers_from_string(trained_weights[-1])[-2]
        current_fold = int(textFinder(trained_weights[-1],"_fold","_of_folds"))
        print(f"Roll back the hist .csv file from trained epochs: [{trained_epochs}] from fold [{current_fold+1}/{folds}]")
        print(f"====================\n")
    
        return int(current_fold), int(trained_epochs)
    else:
        print(f"{folds}-Folds Cross Validation for {model_name}")
        print(f"====================\n")
        return 0,0

def get_current_fold_and_hist_window_stride(model_name, fold_i, folds, window_size, stride, max_epochs):
    from MLModels import SVMModel, CNN_Base_1D_Model, ResNet15_1D_Model
    from natsort import natsorted
    import pandas as pd
    checkpoint_name0 = f'{model_name}_classification_window{window_size}_stride{stride}_fold{fold_i}_of_folds{folds}'
    # trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.pt'])
    # trained_weights_hist =natsorted([i for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.csv'])
    trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and checkpoint_name0 in i])
    trained_weights_hist =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and checkpoint_name0 in i])

    if len(trained_weights) >= 1:
        print(f"Resuming from {trained_weights[-1]}")
        trained_path = os.path.join(os.getcwd(),"lfs","weights",trained_weights[-1])
        # trained_weights,trained_weights_hist

        if model_name == "SVM":
            # if rank ==0:
            print("Using SVM")
            trained_model = SVMModel(1920+4).double()
        if model_name == "CNN":
            # if rank ==0:
            print("Using CNN")
            trained_model = CNN_Base_1D_Model(time_series_length=1920, meta_data_size=4).double()
        if model_name == "Res15":
            # if rank ==0:
            print("Using Res15")
            trained_model = ResNet15_1D_Model(time_series_length=1920, meta_data_size=4).double()

        trained_epochs = load_model_epochs(trained_weights[-1],trained_model)
        trained_hist = pd.read_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"),index_col=0)
        train_hist_drop = trained_hist.drop(trained_hist[trained_hist.index>trained_epochs].index)
        train_hist_drop.to_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"))
        # current_fold = get_numbers_from_string(trained_weights[-1])[-2]
        current_fold = int(textFinder(trained_weights[-1],"_fold","_of_folds"))
        print(f"Roll back the hist .csv file from trained epochs: [{trained_epochs}] from fold [{current_fold+1}/{folds}]")
        print(f"====================\n")
    
        return int(current_fold), int(trained_epochs)
    else:
        print(f"{folds}-Folds Cross Validation for {model_name}")
        print(f"====================\n")
        return 0,0

def get_current_fold_and_hist_point_wised(model_name,input_type,output_type,folds, roi_radius,select_cube,select_power,select_direction,select_regime,max_epochs):
    from MLModels import CNN_Base_2D_Model, ResNet15_2D_Model
    from natsort import natsorted
    import pandas as pd
    checkpoint_name0 = f'Point_wised_{model_name}_classification_{input_type}_output_{output_type}_roi_radius{roi_radius}_select_cube{select_cube}_power{select_power}_direction{select_direction}_regime{select_regime}_fold'
    # trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.pt'])
    # trained_weights_hist =natsorted([i for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.csv'])
    trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and checkpoint_name0 in i])
    trained_weights_hist =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and checkpoint_name0 in i])

    if len(trained_weights) >= 1:
        print(f"Resuming from {trained_weights[-1]}")
        trained_path = os.path.join(os.getcwd(),"lfs","weights",trained_weights[-1])
        # trained_weights,trained_weights_hist

        if model_name == "CNN":
            # if rank ==0:
            print("Using CNN")
            trained_model = CNN_Base_2D_Model(input_shape=(2,100), meta_data_size=4).double()
        if model_name == "Res15":
            # if rank ==0:
            print("Using Res15")
            trained_model = ResNet15_2D_Model(input_shape=(2,100), meta_data_size=4).double()

        trained_epochs = load_model_epochs(trained_weights[-1],trained_model)
        trained_hist = pd.read_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"),index_col=0)
        train_hist_drop = trained_hist.drop(trained_hist[trained_hist.index>trained_epochs].index)
        train_hist_drop.to_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"))
        # current_fold = get_numbers_from_string(trained_weights[-1])[-2]
        current_fold = int(textFinder(trained_weights[-1],"_fold","_of_folds"))
        print(f"Roll back the hist .csv file from trained epochs: [{trained_epochs}] from fold [{current_fold+1}/{folds}]")
        print(f"====================\n")
    
        return int(current_fold), int(trained_epochs)
    else:
        print(f"{folds}-Folds Cross Validation for {model_name}")
        print(f"====================\n")
        return 0,0

def get_current_fold_and_hist_line_wised(model_name,input_type,output_type,folds, roi_radius,select_cube,select_power,select_direction,select_regime,max_epochs):
    from MLModels import SVMModel, CNN_Base_1D_Model, ResNet15_1D_Model
    from natsort import natsorted
    import pandas as pd
    checkpoint_name0 = f'Line_wised_{model_name}_classification_{input_type}_output_{output_type}_roi_radius{roi_radius}_select_cube{select_cube}_power{select_power}_direction{select_direction}_regime{select_regime}_fold'
    # trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.pt'])
    # trained_weights_hist =natsorted([i for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.csv'])
    trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and checkpoint_name0 in i])
    trained_weights_hist =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and checkpoint_name0 in i])

    if len(trained_weights) >= 1:
        print(f"Resuming from {trained_weights[-1]}")
        trained_path = os.path.join(os.getcwd(),"lfs","weights",trained_weights[-1])
        # trained_weights,trained_weights_hist

        if model_name == "SVM":
            # if rank ==0:
            print("Using SVM")
            trained_model = SVMModel(1920+4).double()
        if model_name == "CNN":
            # if rank ==0:
            print("Using CNN")
            trained_model = CNN_Base_1D_Model(time_series_length=1920, meta_data_size=4).double()
        if model_name == "Res15":
            # if rank ==0:
            print("Using Res15")
            trained_model = ResNet15_1D_Model(time_series_length=1920, meta_data_size=4).double()

        trained_epochs = load_model_epochs(trained_weights[-1],trained_model)
        trained_hist = pd.read_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"),index_col=0)
        train_hist_drop = trained_hist.drop(trained_hist[trained_hist.index>trained_epochs].index)
        train_hist_drop.to_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"))
        # current_fold = get_numbers_from_string(trained_weights[-1])[-2]
        current_fold = int(textFinder(trained_weights[-1],"_fold","_of_folds"))
        print(f"Roll back the hist .csv file from trained epochs: [{trained_epochs}] from fold [{current_fold+1}/{folds}]")
        print(f"====================\n")
    
        return int(current_fold), int(trained_epochs)
    else:
        print(f"{folds}-Folds Cross Validation for {model_name}")
        print(f"====================\n")
        return 0,0

def get_numbers_from_string(s):
    import re
    # Find all numbers in the string using regular expression
    numbers = re.findall(r'\d+', s)
    return numbers

def get_max_length(array_list):
    """
    Calculates the maximum length among all NumPy arrays in a given list.

    Args:
        array_list: A list of NumPy arrays.

    Returns:
        The maximum length found among the arrays.
    """

    if not array_list:
        return 0  # Handle empty list case

    max_length = max(len(array) for array in array_list)
    return max_length

def get_windowed_data(data, index, window_size=101, fade_length=16):
    """
    Gets a window of data around a given index, with zero padding and fade in/out.

    Args:
    data: A 1D numpy array or torch tensor.
    index: The index of the data point to center the window around.
    window_size: The desired size of the window.
    fade_length: The length of the fade in/out regions.

    Returns:
    A torch tensor of size (window_size,) containing the windowed data.
    """

    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    # Calculate start and end indices
    start_index = max(0, index - window_size // 2)
    end_index = min(len(data), start_index + window_size)

    # Calculate padding lengths
    left_pad = max(0, start_index - index + window_size // 2)
    right_pad = max(0, window_size - (end_index - start_index))

    # Create windowed data with padding
    windowed_data = torch.cat([
    torch.zeros(left_pad),
    data[start_index:end_index],
    torch.zeros(right_pad)
    ])

    # Apply fade in/out
    fade_in = torch.linspace(0, 1, fade_length)
    fade_out = torch.linspace(1, 0, fade_length)
    windowed_data[:fade_length] *= fade_in
    windowed_data[-fade_length:] *= fade_out

    return windowed_data[:window_size]

def augment_abnormal_data(data,gpu_id,ratio=4):
    """Augments abnormal data by random cropping and stretching.

    Args:
        data: The input data tensor.
        target_length_ratio: The target length ratio for cropping.
        stretch_compress_ratio: The stretch/compress ratio.

    Returns:
        A list of augmented data tensors.
    """
    
    target_length_ratio = torch.abs(0.9 + (torch.rand(1) * (0.95 - 0.9)))
    augmented_data = []
    stretch_compress_ratio = torch.abs(0.98 + (torch.rand(1) * (1.02 - 0.98)))

    for _ in range(ratio):
        # Randomly crop the data
        crop_start = int(torch.rand(1) * (data.shape[1] - data.shape[1] * target_length_ratio))
        crop_end = crop_start + int(data.shape[1] * target_length_ratio)
        cropped_data = data[:, crop_start:crop_end]

        # Stretch or compress the data
        stretched_data = torch.nn.functional.interpolate(cropped_data.unsqueeze(0), size=data.shape[1], mode='linear', align_corners=False).squeeze(0)
        stretched_data *= stretch_compress_ratio
        augmented_data.append(stretched_data)
    return augmented_data

def augment_dataset(label, time_series, meta_list,gpu_id, ratio=4):
    idx = torch.where(label==1)[0].to('cpu')
    data = time_series[idx]
    _m = augment_abnormal_data(data,gpu_id,ratio)
    augmented_data = torch.vstack([torch.vstack(_m),time_series])

    _aug_meta_list = []
    if len(meta_list)>0:
        for _meta in meta_list:
            _m = torch.vstack(list(_meta[idx])*ratio).squeeze()
            _aug_meta_list.append(torch.cat([_m,_meta]))
    _l = torch.cat([torch.vstack(list(label[idx])*ratio).squeeze(),label]) 
    return augmented_data, _aug_meta_list,_l

def fill_nan_scipy(arr):
    x, y = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
    points = np.vstack((x[~np.isnan(arr)].flatten(), y[~np.isnan(arr)].flatten())).T
    values = arr[~np.isnan(arr)].flatten()
    arr[np.isnan(arr)] = griddata(points, values, (x[np.isnan(arr)], y[np.isnan(arr)]), method='nearest')

def check_dataset_normality(dataloader):
    """
    Checks for normality in the dataset loaded by the provided dataloader.

    Args:
        dataloader: An iterable DataLoader object that yields batches of data.

    Prints:
        The indices of the datasets that do not appear to be normally distributed.
    """

    from scipy.stats import shapiro

    for idx, data in tqdm(enumerate(dataloader)):
        # Extract the relevant feature from the data (adjust this line as needed)
        feature = data[0]  # Assuming the first element in the data tuple is the feature

        # Perform Shapiro-Wilk test for normality
        stat, p = shapiro(feature)

        # Define a significance level (e.g., 0.05)
        alpha = 0.05

        # Check if the p-value is less than the significance level
        if p < alpha:
            print(f"Dataset at index {idx} does not appear to be normally distributed.") 

class FocalLoss(nn.Module):
    def __init__(self, 
                 alpha: torch.Tensor = None,
                 gamma: float = 2.0,
                 reduction: str = 'sum',
                 task: str = 'auto'):
        """
        PyTorch-native interface Focal Loss supporting:
        - Binary classification (BCEWithLogitsLoss compatible)
        - Multi-class classification (CrossEntropyLoss compatible)
        
        Args:
            alpha (Tensor, optional): Class weighting tensor (size [C]). Defaults to None
            gamma (float): Focusing parameter [0-5]. Higher values focus on hard examples
            reduction (str): 'none'|'mean'|'sum'
            task (str): 'auto'|'binary'|'multiclass' (auto-detects from input shape)
        """
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.task = task
        print(f"Class Weights:{alpha} with focusing parameter: {gamma}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Auto-detect task type from input dimensions
        device = inputs.device
        if self.task == 'auto':
            task = 'binary' if (inputs.ndim == 1 or inputs.shape[1] == 1) else 'multiclass'
        else:
            task = self.task
            
        if task == 'binary':
            return self._forward_bce(inputs, targets, device)
        return self._forward_ce(inputs, targets,device)

    def _forward_bce(self, inputs, targets, device):
        alpha = self.alpha.to(device) if self.alpha is not None else None
        # Binary classification path (BCEWithLogitsLoss compatible)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )

        # Focal modulation
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating = (1 - p_t) ** self.gamma
        
        # Class weighting
        if self.alpha is not None:
            alpha = self.alpha[targets.long()] if self.alpha.dim() > 0 else self.alpha
            alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
            loss *= alpha_factor

        loss = modulating * loss
        
        return self._reduce_loss(loss)

    def _forward_ce(self, inputs, targets, device):
        if self.alpha is not None:
            self.alpha = self.alpha.to(device)  # Ensure weights on correct device [13](@ref)
        # Multi-class path (CrossEntropyLoss compatible)
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, 
            weight=self.alpha, 
            reduction='none'
        )
        
        # Focal modulation
        p_t = torch.exp(-ce_loss)
        modulating = (1 - p_t) ** self.gamma
        
        loss = modulating * ce_loss
        
        return self._reduce_loss(loss)

    def _reduce_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss  # 'none'

class TrainerBase:
    """
    Trainer class for training and evaluating a machine learning model.
    Attributes:
        device (torch.device): The device to run the model on (CPU or GPU).
        model (torch.nn.Module): The machine learning model to be trained.
        checkpoint_dir (str): Directory to save checkpoints and training history.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        epochs (int): Number of epochs to train the model.
        save_interval (int): Interval (in epochs) to save checkpoints.
        verbose_interval (int): Interval (in iterations) to print training progress.
        checkpoint_name (str): Name of the checkpoint file.
        fold_n (int): Fold number extracted from the checkpoint name.
        train_his_path (str): Path to save the training history CSV file.
        checkpoint_path (str): Path to save the checkpoint file.
        start_epoch (int): Epoch to start/resume training from.
        loss_history (list): List to store the training and testing loss history.
        epoch_times (list): List to store the time taken for each epoch.
    Methods:
        _parse_fold_number(name): Extracts fold number from the checkpoint name.
        _try_resume(): Tries to resume training from the last checkpoint.
        _save_checkpoint(epoch): Saves the current state of the model and optimizer.
        _save_history(): Saves the training history to a CSV file.
        train(): Trains the model for the specified number of epochs.
        test(forward_fn): Evaluates the model on the test dataset.
    """
    def __init__(self, model, dataset, train_idx, test_idx, checkpoint_dir, checkpoint_name,
                optimizer,device_id='cpu', save_interval:int=2, 
                verbose_interval=10, 
                epochs=100, batch_size=64, locate_fn=None,num_workers=4,
                test=False,
                criterion = None,
                task_type='regression',  # 'regression', 'binary', 'multiclass',
                num_classes=None,  # Required for classification tasks
                input_type='all',
                safety_ram_margin=0.1, chunk_keep_in_memory=1,
                dataset_mode='classical', # 'ram', 'chunked','classical'
                augmentation=None, # a data augmentation function
                class_weights = None,
                sample_weights =None,
                gamma=4.0,
                reduction='sum'
                 ):
        # Device configuration
        self.device = f'{device_id}'
        self.model = model.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        self.test = test
        self.input_type = input_type
        self.batch_size = batch_size
        self.num_workers = num_workers

        # +++ Task-specific configuration +++
        self.task_type = task_type
        self.num_classes = num_classes
        print(f"Task type: {task_type}")
        if criterion is None:
            if self.task_type == 'classification':
                print(f"Number of classes: {num_classes}")
                if num_classes == 1:
                    # print(f"Using BCEWithLogitsLoss criterion, weights={class_weights[0]/class_weights[1]}")
                    # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[0]/class_weights[1])
                    self.criterion = FocalLoss(alpha=class_weights, gamma=gamma,task='binary', reduction=reduction).to(self.device)
                else:
                    print(f"Using CrossEntropyLoss criterion, weights={class_weights}")
                    # self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                    self.criterion = FocalLoss(alpha=class_weights, gamma=gamma,task='multiclass', reduction=reduction).to(self.device)
                self.n_classes = len(class_weights)
            else:
                self.criterion = torch.nn.MSELoss()
                print(f"Using MSELoss criterion")
        else:
            self.criterion = criterion
            print(f"Using custom criterion: {criterion}")

        # Validate task type
        if self.task_type == 'classification' and num_classes is None: 
            raise ValueError("num_classes must be specified for classification tasks")

        
        # Data configuration
        self.collate_fn = locate_fn if locate_fn is not None else None
        train_sampler=None
        test_sampler=None
        shuffle = True
        if self.task_type == 'classification':
            if sample_weights is not None:
                train_sampler = WeightedRandomSampler(
                    sample_weights['train'],
                    num_samples=len(train_idx))
                test_sampler  = WeightedRandomSampler(
                    sample_weights['test'],
                    len(test_idx))
                shuffle = False
        if dataset_mode == "classical" or "ram":
            self.train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=shuffle,collate_fn = self.collate_fn, num_workers=self.num_workers, pin_memory=True, sampler=train_sampler)
            self.train = self._train_classical
        else:
            self.train = self._train_chunked
        self.test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size,collate_fn = self.collate_fn, num_workers=self.num_workers, shuffle=shuffle,sampler=test_sampler)
        
        # Training configuration
        self.optimizer = optimizer
        self.epochs = epochs
        self.save_interval = save_interval
        self.verbose_interval = verbose_interval
        
        # Checkpoint configuration
        self.checkpoint_name = checkpoint_name
        self.train_his_path = os.path.join(checkpoint_dir, 'train_his', f'{checkpoint_name}.csv')
        self.checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints', f'{checkpoint_name}.pt')
        
        # Initialize training state
        self.start_epoch = 0
        self.loss_history = []
        self.epoch_times = []
        
        # Create directories if needed
        os.makedirs(os.path.join(checkpoint_dir,'train_his'), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir,'checkpoints'), exist_ok=True)
        
        # Try to resume from checkpoint
        self._try_resume()

        # Memory management parameters
        self.safety_margin = safety_ram_margin  # % of RAM to leave free
        self.chunk_keep_in_memory = chunk_keep_in_memory  

        self.current_chunk = None
        self.chunk_order = []
        self.chunk_generator = None
        
        
        # Estimate dataset memory footprint
        self.train_dataset = Subset(dataset, train_idx.cpu().numpy() if isinstance(train_idx, torch.Tensor) else train_idx)
        self.test_dataset = Subset(dataset, test_idx.cpu().numpy() if isinstance(test_idx, torch.Tensor) else test_idx)

        if dataset_mode == "chunked":
            self.train_chunks = self._create_memory_chunks(self.train_dataset)
            # self.test_chunks = self._create_memory_chunks(self.test_dataset)
            
            # Initialize data loaders
            self.current_train_loader = None
            # self.current_test_loader = None
            # Create chunk indices (not loaded yet)
            self.train_chunk_indices = self._create_chunk_indices()
            self.test_chunk_indices = self._create_chunk_indices(is_test=True)
            self.loaded_chunks = []     # Stores (chunk_data, chunk_idx)
   
    
    def _parse_fold_number(self, name):
        # Extract fold number from checkpoint name (e.g., 'fold3_of_5')
        if 'fold' in name:
            parts = name.split('_')
            return int(parts[0][4:])
        return 0

    def _create_memory_chunks(self, dataset):
        # First get the actual indices from the Subset
        original_indices = dataset.indices
        
        # Estimate sample size in bytes
        sample_size = self._estimate_sample_size(dataset[0])
        total_size = len(dataset) * sample_size
        
        # Get available memory (with safety margin)
        free_mem = psutil.virtual_memory().available * (1 - self.safety_margin)
        chunk_size = int(free_mem // (sample_size * self.chunk_keep_in_memory))

        print(f"Sample size (MB): {sample_size / 1e6:.2f}")
        print(f"Total dataset size (GB): {total_size / 1e9:.2f}")
        print(f"Available memory GB: {free_mem /1e9:.2f}")

        if chunk_size >= len(dataset):
            # Return list with single chunk containing all indices
            return [Subset(dataset.dataset, original_indices)]
        
        # Split indices into chunks
        num_chunks = math.ceil(len(original_indices) / chunk_size)
        chunks = [
            Subset(dataset.dataset, original_indices[i*chunk_size:(i+1)*chunk_size])
            for i in range(num_chunks)
        ]
        print(f"Number of chunks: {num_chunks}")
        print(f"Chunk size: {total_size /1e9 // num_chunks:.2f} GB")
        print(f"Creating {num_chunks} chunks for dataset of size {len(dataset)}")
        return chunks
    
    def _estimate_sample_size(self, sample):
        # Recursively calculate tensor sizes
        size = 0
        if isinstance(sample, torch.Tensor):
            return sample.element_size() * sample.nelement()
        elif isinstance(sample, (list, tuple)):
            for item in sample:
                size += self._estimate_sample_size(item)
        elif isinstance(sample, dict):
            for v in sample.values():
                size += self._estimate_sample_size(v)
        return size
    
    def _load_to_ram(self, dataset):
        if self.current_chunk is not None:
            del self.current_chunk
            self._manage_memory()

        loaded = []
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            # loaded.append(self._convert_to_device(sample))
            loaded.append(sample)
        self.current_chunk = loaded

    def _load_chunk(self, chunk_idx, is_test=False):
        """Load chunk using correct indices from original dataset"""
        # Get appropriate chunk indices
        chunk_indices = (
            self.test_chunk_indices[chunk_idx] 
            if is_test 
            else self.train_chunk_indices[chunk_idx]
        )
        
        # Verify chunk index validity
        if chunk_idx >= len(self.train_chunk_indices):
            raise ValueError(f"Invalid chunk index {chunk_idx} for {'test' if is_test else 'train'} dataset")
        
        # Load from original dataset using correct indices
        dataset = self.test_dataset if is_test else self.train_dataset
        chunk_subset = Subset(dataset.dataset, chunk_indices)
        
        return self._load_to_ram(chunk_subset)

    def _get_next_chunk(self):
        # Memory management and chunk loading
        while len(self.current_chunks) < self.chunk_keep_in_memory:
            next_chunk = next(self.chunk_generator)
            self.current_chunks.append(self._load_to_ram(next_chunk))
        
        current_chunk = self.current_chunks.pop(0)
        return DataLoader(current_chunk, batch_size=self.batch_size, shuffle=True, collate_fn = self.collate_fn)

    def _create_chunk_indices(self, is_test=False):
        """Create chunks using original dataset indices"""
        dataset = self.test_dataset if is_test else self.train_dataset
        
        # Get actual indices from the Subset
        original_indices = dataset.indices  # These are indices into the original dataset
        
        # Shuffle the original indices
        np.random.shuffle(original_indices)
        
        # Calculate chunk size based on available memory
        sample_size = self._estimate_sample_size(dataset[0])
        free_mem = psutil.virtual_memory().available * 0.8
        chunk_size = int(free_mem // sample_size)
        
        # Split original indices into chunks
        num_chunks = math.ceil(len(original_indices) / chunk_size)
        return [
            original_indices[i*chunk_size:(i+1)*chunk_size] 
            for i in range(num_chunks)
        ]

    def _manage_memory(self):
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _try_resume(self):
        """
        Attempts to resume training from a saved checkpoint if it exists.

        This method checks if a checkpoint file exists at the specified path.
        If it does, it loads the model state, optimizer state, starting epoch,
        and loss history from the checkpoint file. It then updates the model,
        optimizer, and other relevant attributes to continue training from where
        it left off.

        Attributes:
            checkpoint_path (str): The path to the checkpoint file.
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            start_epoch (int): The epoch to start training from.
            loss_history (list): The history of loss values during training.

        Prints:
            str: A message indicating the training has resumed from a specific epoch.
        """
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path,map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.loss_history = checkpoint['loss_history']
            print(f"Resumed training from epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
        }, self.checkpoint_path)

    def _save_history(self,current_epoch_number):
        if self.task_type == 'classification':
            header = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'epoch_time']
        elif self.task_type == 'regression':
            header = ['epoch', 'train_loss', 'train_mse', 'train_mae', 'test_loss', 'test_mse', 'test_mae', 'epoch_time']
            
        target_line = current_epoch_number  # Header is line 0, epoch 1 is line 1, etc.
        # Write or append to CSV
        # mode = 'a' if os.path.exists(self.train_his_path) else 'w'
        # with open(self.train_his_path, mode, newline='') as f:
        #     writer = csv.writer(f)
        #     if mode == 'w':
        #         writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'epoch_time'])
        #     writer.writerows(self.loss_history)

        # Find data for current epoch (assumes last entry is current)
        if not self.loss_history or int(self.loss_history[-1][0]) != current_epoch_number:
            raise ValueError(f"No valid data found for epoch {current_epoch_number}")
        current_data = self.loss_history[-1]

        # Read existing content
        rows = []
        if os.path.exists(self.train_his_path):
            with open(self.train_his_path, 'r', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)

        # Ensure header exists
        if not rows or rows[0] != header:
            rows = [header]

        # Expand rows to reach target line
        while len(rows) <= target_line:
            rows.append([])  # Add empty placeholder rows

        # Update target line with current data
        rows[target_line] = current_data

        # Write back all lines
        with open(self.train_his_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
    def _run_epoch(self, epoch):
        self.model.train()
        self._eval_matrix = {
            'train_loss': 0.0,
            'current_epoch_train_loss': 0.0,
            'test_loss': 0.0,
            'current_epoch_test_loss': 0.0,
            'current_epoch_train_number': 0,
            'current_epoch_test_number': 0,
            'start_iter_time': time.time(),
            'start_epoch_time': time.time(),
        }

        if self.task_type == 'regression':
            self._eval_matrix['train_mse'] = 0.0
            self._eval_matrix['train_mae'] = 0.0
            self._eval_matrix['test_mse'] = 0.0
            self._eval_matrix['test_mae'] = 0.0

        if self.task_type == 'classification':
            self._eval_matrix['train_correct'] = 0.0
            self._eval_matrix['test_correct'] = 0.0
        
        # Train
        for i, batch in enumerate(self.train_loader):
            self._run_batch(i=i, batch=batch, epoch=epoch, test=False)
        
        # verbose each epoch
        if self.task_type == 'classification':
            mae_or_correct = self._eval_matrix['train_correct']
        elif self.task_type == 'regression':
            mae_or_correct = self._eval_matrix['train_mae']
        print(f"Epoch {epoch:{3}} | Train:", end="")
        train_loss, train_acc_or_mse = self._verbose_epoch(
            epoch=epoch,
            loss=self._eval_matrix['train_loss'],
            current_epoch_sample_number=self._eval_matrix['current_epoch_train_number'],
            mae_or_correct=mae_or_correct
        )

        # Test after each epoch
        self.model.eval()
        for i, batch in enumerate(self.train_loader):
            self._run_batch(i=i, batch=batch, epoch=epoch, test=True)

        if self.task_type == 'classification':
            mae_or_correct = self._eval_matrix['test_correct']
        elif self.task_type == 'regression':
            mae_or_correct = self._eval_matrix['test_mae']
        print(f" | Test:", end="")
        test_loss, test_acc_or_mse = self._verbose_epoch(
            epoch=epoch,
            loss=self._eval_matrix['test_loss'],
            current_epoch_sample_number=self._eval_matrix['current_epoch_test_number'],
            mae_or_correct=mae_or_correct
        )
        
        # Calculate epoch metrics
        epoch_time = time.time() - self._eval_matrix['start_epoch_time']
        self.epoch_times.append(epoch_time)
        
        # Store history
        if self.task_type == 'classification':
            self.loss_history.append((
                epoch, train_loss, train_acc_or_mse,
                test_loss, test_acc_or_mse, epoch_time
            ))
        elif self.task_type == 'regression': 
            self.loss_history.append((
                epoch, 
                train_loss, 
                train_acc_or_mse, 
                self._eval_matrix['train_mae']/self._eval_matrix['current_epoch_train_number'],
                test_loss,
                test_acc_or_mse, 
                self._eval_matrix['test_mae']/self._eval_matrix['current_epoch_test_number'], 
                epoch_time
            ))
        
        # Save checkpoint periodically
        if (epoch % self.save_interval) == 0:
            self._save_checkpoint(epoch)
        
        # Save history and print status
        self._save_history(current_epoch_number=epoch)
        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining = (self.epochs - epoch - 1) * avg_time
        print(f' | Remaining: {remaining//3600:.0f}h {(remaining%3600)//60:.0f}m')

    def _run_batch(self, i, batch, epoch, test=False):
        # Forward pass
        self.optimizer.zero_grad()
        outputs, labels = self._forward(batch)
        if torch.isnan(outputs).any():
            print("NaN detected in outputs. Skipping batch.")
            raise ValueError(f"No valid data found for epoch [{epoch}], batch_number [{i}]")

        if self.task_type == 'regression':
            labels = labels.float()
        elif self.task_type == 'classification':
            if self.num_classes == 1:
                labels = labels.float() # BCEWithLogitsLoss needs float targets
                labels = labels.view(-1, 1)  # Ensure labels are 2D for BCEWithLogitsLoss
            else:
                labels = labels.long().to(self.device)  # CrossEntropyLoss needs long indices

        # Assert shape should be consistent
        if outputs.numel() != labels.numel():
            raise ValueError(f"{i}-th iteration: Shape mismatch: outputs {outputs.shape}, labels {labels.shape}")
        loss = self.criterion(outputs, labels)
        # print(f"output type: {outputs.dtype}, label type: {labels.dtype}")
        # print(f"loss: {loss}")
        if torch.isnan(loss):
            print(f"NaN in {i}th batch")
            print(f"output type: {outputs.dtype}, label type: {labels.dtype}")
            print(f"output: {outputs.squeeze()}, label: {labels.squeeze()}")

        # Backward pass
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        # Statistics
        running_loss = 0
        running_loss += loss.item()

        if test == False:
            total = self._eval_matrix['current_epoch_train_number']
        elif test ==True:
            total = self._eval_matrix['current_epoch_test_number']
        
        # Get current 'correct' number, or average 'mae', 'mse' only calculated when necessary (verbose)
        if self.task_type == 'classification':
            if self.num_classes == 1: # Binary classification
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                labels = labels.long().squeeze()
            else:  # Multi-class classification
                _, preds = torch.max(outputs, 1)
            correct = (preds == labels).sum().item()

            balanced_acc = self._balanced_acc(preds=preds,labels=labels)


        elif self.task_type == 'regression':
            # mse = running_loss / labels.size(0)
            mae = torch.nn.L1Loss()(outputs, labels).item()
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        total += labels.size(0)

        # Assign values to _evaluation_matrix
        if test == False:
            self._eval_matrix['current_epoch_train_number'] = total
            self._eval_matrix['train_loss'] += running_loss
            if self.task_type == 'classification':
                self._eval_matrix['train_correct'] += (balanced_acc*labels.size(0))
                # self._eval_matrix['train_correct'] += correct
            elif self.task_type == 'regression':
                self._eval_matrix['train_mae'] += mae

        elif test ==True:
            self._eval_matrix['current_epoch_test_number'] = total
            self._eval_matrix['test_loss'] += running_loss
            if self.task_type == 'classification':
                # self._eval_matrix['test_correct'] += correct
                self._eval_matrix['test_correct'] += (balanced_acc*labels.size(0))
            elif self.task_type == 'regression':
                self._eval_matrix['test_loss'] += running_loss
                self._eval_matrix['test_mae'] += mae


        # Verbose logging only for training
        if test == False:
            if self.verbose_interval != 0:
                if (i % self.verbose_interval) == (self.verbose_interval-1):
                    iter_time = (time.time() - self._eval_matrix['start_iter_time'])/labels.size(0)
                    avg_loss = running_loss / labels.size(0)
                    # +++ verbose output based on task type +++
                    log_str = f'Epoch {epoch}, iter {i+1}: Loss={avg_loss:.4f}'
                    if self.task_type == 'classification':
                        acc = 100 * correct / labels.size(0)
                        log_str += f', Acc={acc:.2f}%'
                    elif self.task_type == 'regression':
                        mse = running_loss / labels.size(0)
                        mae = torch.nn.L1Loss()(outputs, labels).item()
                        log_str += f', MSE={mse:.4f}, MAE={mae:.4f}'
                    log_str += f', Time={iter_time:.2f}s'
                    print(log_str)
                    self._eval_matrix['start_iter_time'] = time.time()
        # else:
            # train_loss = running_loss / len(self.train_loader)
            # if self.task_type == 'classification':
                # train_acc = 100 * correct / total
                # print(f'Epoch {epoch}, Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
                
                
            # else:
                # train_mse = running_loss / len(self.train_loader)
                # train_mae = torch.nn.L1Loss()(outputs, labels).item()
                # print(f'Epoch {epoch}, Loss={train_loss:.4f}, MSE={train_mse:.4f}, MAE={train_mae:.4f}')

        return 

    def _forward(self, batch):
        # Default forward function if not provided
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        return outputs, labels
    
    def _test_forward(self,batch):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model._test_forward(inputs)
        return outputs

    def _run_batch_chunked(self, batch):
        self.optimizer.zero_grad()
        outputs, labels = self._forward(batch)
        print(f"output type: {outputs.dtype}, label type: {labels.dtype}")
        print(f"output shape: {outputs.shape}, label shape: {labels.shape}")

        # +++ Add dtype conversion based on task type +++
        if self.task_type == 'regression':
            labels = labels.float()
        elif self.task_type == 'classification':
            if self.num_classes == 1:
                labels = labels.float() # BCEWithLogitsLoss needs float targets
                labels = labels.view(-1, 1)  # Ensure labels are 2D for BCEWithLogitsLoss
            else:
                labels = labels.long().to(self.device)  # CrossEntropyLoss needs long indices
        
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return outputs, labels, loss.item()

    def _train_classical(self):
        for epoch in range(self.start_epoch, self.epochs):
            self._run_epoch(epoch=epoch)

        self._save_checkpoint(self.epochs-1)

        # if self.task_type == 'classification':
        #     cm = self._confusion_matrix()
        #     # print(f"Confusion Matrix:\n{cm}")
        #     return cm
        # else:
        #     return None
            

    def _confusion_matrix(self): 
        # Go through all the data and calculate confusion matrix
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in self.test_loader:
                outputs, labels = self._forward(batch)
                if labels.numel() == 0:
                    continue
                if self.num_classes == 1:
                    preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                    labels = labels.long().squeeze()
                else:
                    _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            for batch in self.train_loader:
                outputs, labels = self._forward(batch)
                if labels.numel() == 0:
                    continue
                if self.num_classes == 1:
                    preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                    labels = labels.long().squeeze()
                else:
                    _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        if len(all_preds) == 0 or len(all_labels) == 0:
            return np.zeros((2,2))
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        # Calculate confusion matrix
        if self.num_classes == 1:
            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        else:
            cm = confusion_matrix(all_labels, all_preds, labels=np.arange(self.num_classes))
        return cm


    def _train_chunked(self):
        samples_seen = 0
        total_samples = len(self.train_dataset)

        for epoch in range(self.start_epoch, self.epochs):
            print(f'Epoch {self.start_epoch}/{self.epochs} started')

            # Shuffle chunk order each epoch
            chunk_order = np.random.permutation(len(self.train_chunks))
            epoch_samples = 0
            
            # # Get data loader for current chunk
            # train_loader = self._get_next_chunk()

            epoch_start = time.time()
            self.model.train()
            
            running_loss = 0.0
            correct = 0
            total = 0
            start_iter_time = time.time()

            if self.task_type == 'regression':
                train_mse = 0.0
                train_mae = 0.0

            for chunk_idx in chunk_order:
                # Verify chunk index before loading
                # if chunk_idx >= len(self.train_chunk_indices):
                #     continue  # or raise error
                # print(self.train_chunk_indices)
                print(f"Process chunk : {chunk_idx}/{len(self.train_chunk_indices)}")
                # print(f"Processing chunk {chunk_idx+1}/{len(self.train_chunk_indices)}")

                # Load current chunk
                chunk = self.train_chunks[chunk_idx]
                self._load_to_ram(chunk)
                
                # Create DataLoader
                train_loader = DataLoader(self.current_chunk, 
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        collate_fn=self.collate_fn,
                                        )
            
                for batch_idx, batch in enumerate(train_loader):
                    # Training step
                    if self.verbose_interval != 0:
                        print(f"running batch {batch_idx+1}/{len(train_loader)}")
                    outputs,labels,loss = self._run_batch_chunked(batch)
                    running_loss += loss
                    
                    samples_seen += len(batch['label'])
                    epoch_samples += len(batch['label'])
                    
                    # Handle intermediate logging
                    if self.verbose_interval != 0:
                        if (samples_seen % self.verbose_interval) == 0:
                            self._log_progress(samples_seen, total_samples, loss)
                    
                    # +++ Metrics calculation +++
                    if self.task_type == 'classification':
                        if self.num_classes == 1:
                            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                        else:
                            _, preds = torch.max(outputs, 1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                    
                    # Verbose logging
                    if self.verbose_interval != 0:
                        if (batch_idx % self.verbose_interval) == (self.verbose_interval-1):
                            iter_time = time.time() - start_iter_time
                            avg_loss = running_loss / self.verbose_interval
                            # +++ verbose output based on task type +++
                            log_str = f'Epoch {epoch}, Iter {batch_idx+1}: Loss={avg_loss:.4f}'
                            if self.task_type == 'classification':
                                acc = 100 * correct / total
                                log_str += f', Acc={acc:.2f}%'
                            else:
                                train_mse = running_loss / self.verbose_interval
                                train_mae = torch.nn.L1Loss()(outputs, labels).item()
                                log_str += f', MSE={train_mse:.4f}, MAE={train_mae:.4f}'
                            log_str += f', Time={iter_time:.2f}s'
                            print(log_str)
                        
                            running_loss = 0.0
                            correct = 0
                            total = 0
                            start_iter_time = time.time()

                self.current_chunk = None
                del train_loader
                self._manage_memory()

            # Ensure full dataset coverage
            # assert epoch_samples == len(self.train_dataset), \
            #     f"Epoch {epoch} only processed {epoch_samples}/{len(self.train_dataset)} samples"
            print(f"Epoch {epoch} processed {epoch_samples}/{len(self.train_dataset)} samples")

            # Test after each epoch
            if self.task_type == 'classification':
                test_loss, test_acc = self._test(epoch)
            else:
                test_mse, test_mae = self._test(epoch)
            
            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start
            train_loss = running_loss / len(self.train_dataset)
            train_acc = 100 * correct / total if self.task_type == 'classification' else 0.0

            self.epoch_times.append(epoch_time)

            
            # Store history
            if self.task_type == 'classification':
                self.loss_history.append((
                    epoch, train_loss, train_acc,
                    test_loss, test_acc, epoch_time
                ))
            else:
                self.loss_history.append((
                    epoch, train_mse, train_mae,
                    test_mse, test_mae, epoch_time
                ))
            
            # Save checkpoint periodically
            if (epoch % self.save_interval) == 0:
                self._save_checkpoint(epoch)
            
            # Save history and print status
            self._save_history(current_epoch_number=epoch)
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining = (self.epochs - epoch - 1) * avg_time
            print(f'Epoch {epoch} completed. Time: {epoch_time:.2f}s, '
                  f'Remaining: {remaining//3600:.0f}h {(remaining%3600)//60:.0f}m')


    def _balanced_acc(self,preds,labels):
        # Balanced acc
        tp = {cls: 0 for cls in range(self.n_classes)}
        fn = {cls: 0 for cls in range(self.n_classes)}
        for cls in range(self.n_classes):
            cls_mask = (labels == cls)
            tp[cls] += (preds[cls_mask] == cls).sum().item()
            fn[cls] += cls_mask.sum().item() - tp[cls]
        recall = {}
        for cls in range(self.n_classes):
            total = tp[cls] + fn[cls]
            recall[cls] = tp[cls] / total if total > 0 else 0  # Avoid division by zero
        balanced_accuracy = sum(recall.values()) / self.n_classes
        return balanced_accuracy


    def _verbose_epoch(self, epoch, loss, current_epoch_sample_number, mae_or_correct):
        iter_time = time.time() - self._eval_matrix['start_epoch_time']
        if current_epoch_sample_number == 0:
            current_epoch_sample_number = 1
        avg_loss = loss / current_epoch_sample_number
        # +++ verbose output based on task type +++
        log_str = f'Loss={avg_loss:.4f}'
        if self.task_type == 'classification':
            acc = 100 * mae_or_correct / current_epoch_sample_number
            acc = 100 * mae_or_correct / current_epoch_sample_number
            log_str += f', Acc={acc:.2f}%'
            acc_or_mse = acc
        elif self.task_type == 'regression':
            mae = mae_or_correct/current_epoch_sample_number
            mse = loss / current_epoch_sample_number
            log_str += f', MSE={mse:.4f}, MAE={mae:.4f}'
            acc_or_mse = mse
        log_str += f', Time={iter_time:.2f}s'
        self._eval_matrix['start_iter_time'] = time.time()                    
        print(log_str, end="")
        return avg_loss, acc_or_mse


    def _log_progress(self, seen, total, loss):
        progress = 100 * seen / total
        print(f"Progress: {progress:.1f}% | Loss: {loss:.4f}")


    def _test(self, epoch):
        """
        Evaluate the model on the test dataset.
        Args:
            forward_fn (function): A function that takes the model and a batch of data as input,
                                   and returns the model's outputs and the corresponding labels.
        Returns:
            tuple: A tuple containing:
                - float: The average loss over the test dataset.
                - float: The accuracy of the model on the test dataset, as a percentage.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        if self.task_type == 'regression':
            test_mse = 0.0
            test_mae = 0.0

        # +++ Add confusion matrix storage +++
        all_preds = []
        all_labels = []

        
        with torch.no_grad():
            for batch in self.test_loader:
                outputs, labels = self._forward(batch)
                if labels.numel() < 1:
                    continue
                if self.num_classes == 1:
                    labels = labels.float() # BCEWithLogitsLoss needs float targets
                    labels = labels.view(-1, 1)  # Ensure labels are 2D for BCEWithLogitsLoss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                if self.task_type == 'classification':
                    if self.num_classes == 1:
                        preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                        labels = labels.long().squeeze()
                        # print(f"Binary labels. \nPreds: {preds} \nlabels:{labels}")
                    else:
                        _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
                else:
                    test_mse += loss.item()
                    test_mae += torch.nn.L1Loss()(outputs, labels).item()

        if self.task_type == 'classification':
            test_loss = total_loss / len(self.test_loader)
            if total == 0:
                total = 1  # Avoid division by zero
            test_acc = 100 * correct / total 
        else:
            test_loss = test_mse / len(self.test_loader)
            test_mse /= len(self.test_loader)
            test_mae /= len(self.test_loader)

        if self.task_type == 'classification':
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            return test_loss, test_acc
        else:
            print(f'Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}')
            return test_mse, test_mae


class CrossValidationTrainer:
    def __init__(self, dataset, Trainer, folds, repeat, task_type, model, optimizer,
                 criterion, collate_fn, project_name, gpu, save_every, epochs,
                 batch_size, fold_i=0, num_workers=4, test=False, verbose_interval=10,
                 checkpoint_dir="lfs", num_classes=None, input_type="all",
                 safety_ram_margin=0.1, dataset_mode='classical', 
                 test_trainer=False, augmentation=False, augmentation_factor=1,gamma=4.0, reduction='sum'):
        # Initialize all parameters
        self.dataset = dataset
        self.Trainer = Trainer
        self.folds = folds
        self.repeat = repeat
        self.task_type = task_type
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.collate_fn = collate_fn
        self.project_name = project_name
        self.gpu = gpu
        self.save_every = int(save_every)
        self.epochs = epochs
        self.batch_size = batch_size
        self.fold_i = fold_i
        self.num_workers = num_workers
        self.test = test
        self.verbose_interval = verbose_interval
        self.checkpoint_dir = checkpoint_dir
        self.num_classes = num_classes
        self.input_type = input_type
        self.safety_ram_margin = safety_ram_margin
        self.dataset_mode = dataset_mode
        self.test_trainer = test_trainer
        self.augmentation = augmentation
        self.gamma = gamma
        self.reduction=reduction
        
        # Initialize state variables
        self.metrics = None
        self.all_k_folds = []
        self.training_time = 0
        self.trained_folds = 0
        self.average_training_time = 0
        self.augmentation_factor = augmentation_factor

        self._preprocess_fold_indices()
        # Initialize metrics
        if self.task_type == 'classification':
            if self.num_classes > 1:
                self.metrics = np.zeros((int(self.folds*self.repeat),self.num_classes, self.num_classes))
            elif self.num_classes == 1:
                self.metrics = np.zeros((int(self.folds*self.repeat),2, 2))

        if int(self.fold_i[0]) == 0:
            self.folds_i = range(int(self.folds * self.repeat))
            print(f"Cross validation on all the {self.folds}*{self.repeat} folds")
        else:
            self.folds_i = np.array(self.fold_i) - 1
            print(f"Cross validation on fold [{self.fold_i}]/{self.folds}*{self.repeat}")


        class_weights = []
        self.sample_weights = None
        if self.task_type == "classification":
            self._get_class_distribution()
            class_counts = {cls: len(indices) for cls, indices in self.class_indices.items()}
            total_samples = sum(class_counts.values())
            # self.sample_weights = np.zeros(total_samples)
            print(f"\n-------------")
            print(f"\nData augmentation")
            print(f"Total samples: {total_samples}")
            print("Class counts:")
            for cls, count in sorted(class_counts.items()):
                weight = total_samples / (len(class_counts) * count)  # Inverse frequency scaling
                class_weights.append(weight)
                print(f"  Class {cls}: {count} samples ({count/total_samples:.1%}), weight: {weight:.2f}")
            # for cls, indices in self.class_indices.items():
            #     self.sample_weights[indices] = class_weights[cls]
            self.class_weights = torch.tensor(class_weights)
            # self.sample_weights = [class_weights[int(d['label'].cpu().numpy())] for d in self.dataset]


            # if self.num_classes == 1:
            #     self.pos_weight = (np.sum(self._labels))/len(self._labels)
            #     print(f"Position weight = {self.pos_weight}")
            #     self.pos_weight = torch.tensor(self.pos_weight)

        if len(class_weights)==0:
            self.class_weights = [0]
        # Create trainer instance
        self.testTrainer = self.Trainer(
            model=self.model,
            dataset=self.dataset,
            train_idx=np.arange(len(self.dataset)),
            test_idx=[0],
            checkpoint_name="test",
            optimizer=self.optimizer,
            device_id=self.gpu,
            save_interval=99999,
            epochs=self.epochs,
            locate_fn=self.collate_fn,
            num_workers=self.num_workers,
            test=self.test,
            verbose_interval=self.verbose_interval,
            batch_size=self.batch_size,
            checkpoint_dir=self.checkpoint_dir,
            criterion=self.criterion,
            task_type=self.task_type,
            num_classes=self.num_classes,
            input_type=self.input_type,
            safety_ram_margin=self.safety_ram_margin,
            dataset_mode=self.dataset_mode,
            augmentation=self.augmentation,
            class_weights = self.class_weights,
            gamma=self.gamma
        )
        if self.test_trainer:
            return self.testTrainer
        
    def _get_class_distribution(self):
        self.class_indices = defaultdict(list)
        self._labels = np.zeros(len(self.dataset))
        for idx in range(len(self.dataset)):
            label = self.dataset[idx]['label']
            self.class_indices[int(label)].append(int(idx))
            self._labels[idx] = label

    def _augmentation(self, train_idx):
        # Compute duplication factors for each class in the training set
        train_labels = [self._labels[idx] for idx in train_idx]
        train_class_counts = defaultdict(int)
        train_class_indices = defaultdict(list)
        for idx, label in zip(train_idx, train_labels):
            train_class_counts[int(label)] += 1
            train_class_indices[int(label)].append(idx)

        max_count = max(train_class_counts.values())
        self.duplication_factors = {cls: (max_count // count) - 1 for cls, count in train_class_counts.items()}

        if hasattr(self, 'duplication_factors'):
            print("\nDuplication factors for balancing:")
            for cls, factor in sorted(self.duplication_factors.items()):
                print(f"  Class {cls}: {factor}x duplication")
        print(f"\n-------------")
        
        # Create augmented indices for each class in the training set
        augmented_indices = defaultdict(list)
        for cls in train_class_counts:
            count = train_class_counts[int(cls)]
            if count == 0:
                continue
            # Number of samples to add
            num_to_add = max_count - count
            num_to_add = int(self.augmentation_factor*num_to_add)
            if num_to_add <= 0:
                continue
            # Get the indices of this class in the original train_idx
            cls_indices = train_class_indices[cls]
            # Randomly select with replacement
            augmented_cls = np.random.choice(cls_indices, num_to_add, replace=True)
            augmented_indices[int(cls)] = augmented_cls.tolist()
        new_train_idx = np.concatenate([train_idx, np.concatenate([augmented_indices[cls] for cls in augmented_indices])])
        return new_train_idx

    def _preprocess_fold_indices(self):
        """Prepare cross-validation folds and initialize metrics"""
        # Generate K-fold indices
        len_dataset = len(self.dataset)
        print(f"Len dataset {len_dataset}")
        
        self.all_k_folds = getKFoldCrossValidationIndexes(len_dataset, self.folds, seed=10086)
        for _i in range(self.repeat):
            self.all_k_folds += getKFoldCrossValidationIndexes(len_dataset, self.folds, _i)

    def train(self):
        """Execute cross-validation training"""
        if not self.all_k_folds:
            self._preprocess_fold_indices()

        for fold_i in self.folds_i:
            self.start_time = time.time()
            print(f"Fold [{fold_i}/{self.folds} * {self.repeat}]")
            train_idx, test_idx = self.all_k_folds[fold_i]
            if self.augmentation == True:
                train_idx = self._augmentation(train_idx=train_idx)
            self._train_single_fold(fold_i, train_idx, test_idx)
            self._update_training_stats()

            if self.task_type == 'classification':
                fold_metrics = self._confusion_matrix()
                print(fold_metrics)
                self.metrics[fold_i] = fold_metrics
                self.verbose(fold_i, self.start_time, fold_metrics)
                self._save_metrics()
            
    def _train_single_fold(self, fold_i, train_idx=None, test_idx=None):
        """Train a single fold"""
        if train_idx is None:
            train_idx, test_idx = self.all_k_folds[fold_i]
        self.sample_weights = None
        if self.task_type=='classification':
            _y_train = [self._labels[i] for i in train_idx]
            _y_test = [self._labels[i] for i in test_idx]
            _sample_weights_train = compute_sample_weight( class_weight="balanced",  y=_y_train)
            _sample_weights_test = compute_sample_weight( class_weight="balanced",  y=_y_test)
            self.sample_weights = {
                'train':_sample_weights_train,
                'test':_sample_weights_test}

        # Initialize model and optimizer
        self.model.initialize_weights()
        self.optimizer.zero_grad()

        # Create trainer instance
        trainer = self.Trainer(
            model=self.model,
            dataset=self.dataset,
            train_idx=train_idx,
            test_idx=test_idx,
            checkpoint_name=f'{self.project_name}_fold{fold_i}_of_folds{self.folds}',
            optimizer=self.optimizer,
            device_id=self.gpu,
            save_interval=self.save_every,
            epochs=self.epochs,
            locate_fn=self.collate_fn,
            num_workers=self.num_workers,
            test=self.test,
            verbose_interval=self.verbose_interval,
            batch_size=self.batch_size,
            checkpoint_dir=self.checkpoint_dir,
            criterion=self.criterion,
            task_type=self.task_type,
            num_classes=self.num_classes,
            input_type=self.input_type,
            safety_ram_margin=self.safety_ram_margin,
            dataset_mode=self.dataset_mode,
            augmentation=self.augmentation,
            class_weights = self.class_weights,
            sample_weights = self.sample_weights,
            gamma = self.gamma,
            reduction=self.reduction
        )

        trainer.train()
        

        # return fold_metrics

    def _confusion_matrix(self): 
        # Go through all the data and calculate confusion matrix
        trainer = self.testTrainer
        trainer.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in trainer.train_loader:
                outputs, labels = trainer._forward(batch)
                if labels.numel() == 0:
                    continue
                if self.num_classes == 1:
                    preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                    labels = labels.long().squeeze()
                else:
                    _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if len(all_preds) == 0 or len(all_labels) == 0:
            return np.zeros((2,2))
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        # Calculate confusion matrix
        if self.num_classes == 1:
            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        else:
            cm = confusion_matrix(all_labels, all_preds, labels=np.arange(self.num_classes))
        return cm

    def _update_training_stats(self):
        """Update training statistics"""
        end_time = time.time()
        fold_duration = end_time - self.start_time
        self.training_time += fold_duration
        self.trained_folds += 1
        self.average_training_time = self.training_time / self.trained_folds

    def _save_metrics(self):
        """Save confusion matrix metrics"""
        os.makedirs(os.path.join(self.checkpoint_dir, "confusion_matrices"), exist_ok=True)
        np.save(
            os.path.join(self.checkpoint_dir, "confusion_matrices", f"{self.project_name}.npy"),
            self.metrics
        )

    def verbose(self, fold_i, start_time, fold_metrics):
        """Print training progress and metrics"""
        end_time = time.time()
        fold_duration = end_time - start_time
        remaining_time = (len(self.folds_i) - fold_i) * self.average_training_time

        print(f"\n=============")
        print(f"Fold [{fold_i}/{self.folds} * {self.repeat}] completed in {fold_duration:.2f}s")
        print(f"Average fold time: {self.average_training_time:.2f}s")
        print(f"Estimated remaining: {datetime.timedelta(seconds=remaining_time)}")
        
        if self.task_type == 'classification':
            print(f"Fold [{fold_i}/{self.folds} * {self.repeat}] Confusion metrics on fold:")
            print_confusion_matrix(fold_metrics)
            print(f"All confusion metrics:")
            print_confusion_matrix(self.metrics.astype(int).sum(0))
            
        print(f"=============\n")

def cv_trainer(dataset,Trainer, folds, repeat, task_type, model,optimizer,criterion, collate_fn,model_name, gpu,save_every:int, epochs,batch_size, fold_i=0, num_workers=4, test=False,verbose_interval=10,checkpoint_dir="lfs",num_classes=None, input_type="all",safety_ram_margin=0.1,dataset_mode='classical',test_trainer=False, augmentation=None):
    """
        Trains a model using cross-validation.
        Parameters:
        dataset (Dataset): The dataset to be used for training and validation.
        Trainer (class): The trainer class responsible for training the model.
        folds (int): The number of folds for cross-validation.
        repeat (int): The number of times to repeat the cross-validation process.
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer to be used for training.
        collate_fn (function): The function to collate data samples into batches.
        model_name (str): The name of the model, used for checkpoint naming.
        gpu (int): The GPU device ID to be used for training.
        save_every (int): The interval (in epochs) at which to save checkpoints.
        epochs (int): The number of epochs to train each fold.
        folds_i (int or list, optional): Specific fold indices to train on. If 0, trains on all folds. Default is 0.
        Returns:
        None
    """
    if int(fold_i[0]) == 0:
        folds_i = range(int(folds*repeat))
        print(f"Cross validation on all the {folds}*{repeat} folds")
    else:
        folds_i = np.array(fold_i)-1
        print(f"Cross validation on fold [{fold_i}]/{folds}*{repeat}")

    len_dataset = len(dataset)
    print(f"Len dataset {len_dataset}")
    all_k_folds = getKFoldCrossValidationIndexes(len_dataset, folds, seed=10086)

    for _i in range(repeat): 
        all_k_folds += getKFoldCrossValidationIndexes(len_dataset, folds, _i)

    training_time = 0
    trained_folds = 0
    # Initialize the confusion metrics 
    if task_type == 'classification':
        if num_classes > 1:
            metrics = np.zeros((num_classes, num_classes))
        elif num_classes == 1:
            metrics = np.zeros((2, 2))

        # Data augmentation
        if augmentation != None:
            augmentation = augmentation if augmentation is not None else lambda x: x

    for fold_i in folds_i:
        start_time = time.time()
        print(f"Fold [{fold_i}/{folds} * {repeat}]")

        checkpoint_name = f'{model_name}_fold{fold_i}_of_folds{folds}'
        
        train_idx, test_idx = all_k_folds[fold_i]
        # max_length = dataset.max_length
        # max_length = transform_ft()(torch.ones(1,dataset.max_length)).shape[1]

        model.initialize_weights()
        optimizer.zero_grad()
        

        trainer = Trainer(
            model = model,
            dataset=dataset,
            train_idx=train_idx,
            test_idx=test_idx,
            checkpoint_name=checkpoint_name,  # Example name
            optimizer=optimizer,
            device_id=gpu,
            save_interval=save_every,
            epochs=epochs,
            locate_fn=collate_fn,
            num_workers=num_workers,
            test=test,
            verbose_interval=verbose_interval,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            criterion=criterion,
            task_type=task_type,
            num_classes=num_classes,
            input_type = input_type,
            safety_ram_margin=safety_ram_margin,
            dataset_mode=dataset_mode,
            augmentation=augmentation
        )
        if test_trainer == True:
            return trainer

        _metrics = trainer.train()

        if task_type == 'classification':
            metrics += _metrics
            # save the metrics
            print(f"Confusion matrix: {metrics}")

        end_time = time.time()
        fold_duration = end_time - start_time 

        if task_type == 'classification':
            print(f"For all the folds, confusion matrix: {metrics}")

            os.makedirs(os.path.join(checkpoint_dir,"confusion_matrices"), exist_ok=True)
            np.save(os.path.join(checkpoint_dir, "confusion_matrices", f"{model_name}.npy"), metrics)

        training_time += fold_duration
        trained_folds += 1
        average_training_time = training_time/trained_folds
        remaining_time = (len(folds_i)-fold_i)*(average_training_time)

        print(f"Fold [{fold_i}/{folds} * {repeat}]")
        print(f"\n=============")
        print(f"Fold [{fold_i}/{folds} * {repeat}] completed in {fold_duration:.2f} seconds")
        print(f"Average training time for each fold: {average_training_time:.2f} seconds")
        print(f"Remaining time: {datetime.timedelta(seconds=remaining_time)} ")
        print(f"=============\n")

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

class TesterBase:
    """
    A base class for evaluating a trained PyTorch model on a dataset.

    This class provides core functionality to generate predictions for an entire
    dataset and compute standard evaluation metrics for regression and
    classification tasks. It is designed to be extended for more specific
    analysis or plotting.

    Attributes:
        model (torch.nn.Module): The trained model to evaluate.
        device (torch.device): The device (CPU or GPU) to run inference on.
        task_type (str): The type of task, either 'regression' or 'classification'.
        num_classes (int, optional): The number of classes for classification tasks.
        forward_fn (callable): The function used to perform a forward pass.
        predictions (np.ndarray): Stores the model's predictions after running predict().
        labels (np.ndarray): Stores the ground truth labels after running predict().
        inputs (list): Stores the raw input batches if requested.
    """
    def __init__(self, model, device, task_type='regression', num_classes=None, forward_fn=None):
        """
        Initializes the TesterBase.

        Args:
            model (torch.nn.Module): The trained PyTorch model.
            device (str or torch.device): The device to run the model on ('cuda:0', 'cpu', etc.).
            task_type (str): The task type, either 'regression' or 'classification'.
            num_classes (int, optional): Required for multi-class classification.
            forward_fn (callable, optional): A function that takes a batch and returns
                (outputs, labels). If None, a default _forward method is used.
        """
        self.model = model.to(device).eval() # Ensure model is on the correct device and in eval mode
        self.device = device
        self.task_type = task_type
        self.num_classes = num_classes
        
        # Use the provided forward function or the default placeholder
        self.forward_fn = forward_fn if forward_fn is not None else self._default_forward
        
        # Placeholders for results
        self.predictions = None
        self.labels = None
        self.inputs = None

        if self.task_type == 'classification' and self.num_classes is None:
            raise ValueError("`num_classes` must be specified for classification tasks.")

    def _default_forward(self, batch):
        """A default forward function placeholder."""
        # This implementation mirrors the one in your Trainer class.
        # It assumes the batch is a dictionary.
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch['label'].to(self.device)
        outputs = self.model(inputs)
        return outputs, labels

    def predict(self, dataset, batch_size=64, num_workers=4, collate_fn=None, return_inputs=False):
        """
        Generates predictions for a given dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to evaluate.
            batch_size (int): The batch size for the DataLoader.
            num_workers (int): Number of workers for the DataLoader.
            collate_fn (callable, optional): The collate function for the DataLoader.
            return_inputs (bool): If True, the raw input batches will also be stored and returned.

        Returns:
            tuple: A tuple containing (predictions, labels). If return_inputs is True,
                   it returns (predictions, labels, inputs).
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        
        all_preds = []
        all_labels = []
        all_inputs = []

        with torch.no_grad():
            for batch in dataloader:
                outputs, labels = self.forward_fn(batch)

                if self.task_type == 'regression':
                    preds = outputs
                elif self.task_type == 'classification':
                    if self.num_classes == 1: # Binary classification
                        preds = torch.sigmoid(outputs)
                    else: # Multi-class classification
                        preds = torch.softmax(outputs, dim=1)
                
                all_preds.append(preds.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                if return_inputs:
                    # Store input features necessary for custom plots (e.g., BDI)
                    # We move it to cpu to avoid storing too much on the GPU
                    cpu_batch = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    all_inputs.append(cpu_batch)

        self.predictions = np.concatenate(all_preds, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        
        if return_inputs:
            self.inputs = all_inputs
            return self.predictions, self.labels, self.inputs
            
        return self.predictions, self.labels

    def analyze(self):
        """
        Performs analysis based on the task type.
        
        You must run `predict()` before calling this method.

        Returns:
            dict: A dictionary containing relevant metrics.
        """
        if self.predictions is None or self.labels is None:
            raise RuntimeError("You must run the `predict()` method before calling `analyze()`.")

        if self.task_type == 'regression':
            return self._analyze_regression()
        elif self.task_type == 'classification':
            return self._analyze_classification()
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def _analyze_regression(self):
        """Computes and prints regression metrics."""
        mae = mean_absolute_error(self.labels, self.predictions)
        mse = mean_squared_error(self.labels, self.predictions)
        r2 = r2_score(self.labels, self.predictions)
        
        print("--- Regression Analysis ---")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE):  {mse:.4f}")
        print(f"R-squared (R²):            {r2:.4f}")
        
        return {'mae': mae, 'mse': mse, 'r2': r2}

    def _analyze_classification(self):
        """Computes and prints classification metrics."""
        if self.num_classes == 1: # Binary
            # Apply 0.5 threshold to sigmoid outputs
            class_preds = (self.predictions > 0.5).astype(int)
            labels = self.labels.astype(int)
            target_names = ['Class 0', 'Class 1']
        else: # Multiclass
            # Get the index of the max probability
            class_preds = np.argmax(self.predictions, axis=1)
            labels = self.labels
            target_names = [f'Class {i}' for i in range(self.num_classes)]

        cm = confusion_matrix(labels, class_preds)
        report = classification_report(labels, class_preds, target_names=target_names, output_dict=True)
        
        print("--- Classification Analysis ---")
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(labels, class_preds, target_names=target_names))
        
        return {'confusion_matrix': cm, 'classification_report': report}

class LPBFDataset(Dataset):
    def __init__(
        self,
        cube_position,
        laser_power,
        scanning_speed,
        regime_info,
        print_direction,
        microphone,
        ae,
        defect_labels,
        fad_in_out_length = 16
    ):
        self.cube_position = cube_position
        self.laser_power = laser_power
        self.scanning_speed = scanning_speed
        self.regime_info = regime_info
        self.print_direction = print_direction
        self.microphone = microphone
        self.ae = ae
        self.defect_labels = defect_labels
        self.max_length = get_max_length(self.microphone)
        self.fad_in_out_length = fad_in_out_length
        print(f"max_length = {self.max_length}")

    def __len__(self):
        return len(self.cube_position)

    def __getitem__(self, idx):
        mic = self.microphone[idx]
        ae = self.ae[idx]

        mic = torch.nn.functional.pad(torch.tensor(mic), (0, self.max_length - mic.shape[-1]))
        ae = torch.nn.functional.pad(torch.tensor(ae), (0, self.max_length - ae.shape[-1]))
        mic = fade_in_out(mic,self.fad_in_out_length)
        ae = fade_in_out(ae,self.fad_in_out_length)

        return (
            self.cube_position[idx],
            self.laser_power[idx],
            self.scanning_speed[idx],
            self.regime_info[idx],
            self.print_direction[idx],
            mic,
            ae,
            self.defect_labels[idx]
        )

class MemoryDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

class LPBFPointDataset(Dataset):
    def __init__(
        self,
        ids,
        cube_position,
        laser_power,
        scanning_speed,
        regime_info,
        print_direction,
        microphone,
        ae,
        window_size,
        fad_in_out_length = 16
    ):
        self.ids = ids
        self.cube_position = cube_position
        self.laser_power = laser_power
        self.scanning_speed = scanning_speed
        self.regime_info = regime_info
        self.print_direction = print_direction
        self.microphone = microphone
        self.ae = ae
        self.window_size = window_size
        self.fad_in_out_length = fad_in_out_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item_id, _layer_idx, _line_idx, daq_idx, _point_label = self.ids[idx]

        defect_label = _point_label
        _mic = self.microphone[item_id]
        _ae = self.ae[item_id]

        mic = get_windowed_data(_mic,daq_idx,window_size = self.window_size, fade_length=self.fad_in_out_length)
        ae = get_windowed_data(_ae,daq_idx,window_size = self.window_size, fade_length=self.fad_in_out_length)

        return (
            self.cube_position[item_id],
            self.laser_power[item_id],
            self.scanning_speed[item_id],
            self.regime_info[item_id],
            self.print_direction[item_id],
            mic,
            ae,
            defect_label
        )

class VideoClassificationDataset(Dataset):
    def __init__(self, data_dir, labels,data_ranges, clip_length=30, hotload=False):
        """
        Args:
            data_dir (str): Path to the directory containing video and label files.
            clip_length (int): Number of frames per clip.
            hotload (bool): If True, load videos on the fly. If False, load all videos into memory.
        """
        self.data_dir = data_dir
        self.clip_length = clip_length
        self.hotload = hotload

        self.labels = labels

        # Ensure each label set corresponds to a video file
        self.video_files = [os.path.join(data_dir, "intermediate", f"thermal_img_{i}.pt") for i in  data_ranges.keys()]
        assert len(self.video_files) == len(self.labels), "Mismatch between videos and labels"

        if not hotload:
            # Load all videos into memory
            self.videos = [torch.load(video_path,weights_only=True) for video_path in self.video_files]
        else:
            self.videos = None

        # Create a list of (video_index, start_frame) pairs for indexing clips
        self.indexes = []
        indexes = []
        # for video_idx, labels in enumerate(self.labels):
        #     num_clips = max(0, len(labels) - self.clip_length + 1)  # Number of clips based on labels
        #     self.indexes.extend([(video_idx, start_frame) for start_frame in range(num_clips)])
        for _ii,_k in enumerate(data_ranges.keys()):
            _indexes = []
            _idx_range = 0
            _ranges = data_ranges[_k]
            for _range in _ranges:
                _idx_range+=(_range[1]-_range[0])
            for _iii in range(_idx_range):
                # self.indexes.append((_ii, _iii))
                    _indexes.append((_ii, _iii))
            indexes.append((_indexes[-1200:-200]))
            self.indexes = list(list(itertools.chain(*indexes)))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        video_idx, start_frame = self.indexes[idx]
        label = self.labels[video_idx]

        if self.hotload:
            # Load the video on the fly and ensure it is a tensor
            video = torch.tensor(torch.load(self.video_files[video_idx]))  # Ensure it is a tensor
        else:
            # Access the preloaded video
            video = self.videos[video_idx]

        # Ensure clip length consistency
        video_length = video.shape[0]
        if start_frame + self.clip_length > video_length:

            clip = video[video_length-self.clip_length:]
            # label = torch.tensor(labels[video_length-self.clip_length:])  # Convert labels to tensor

        else:
            # Regular slicing
            clip = video[start_frame:start_frame + self.clip_length]
            # label = torch.tensor(labels[start_frame:start_frame + self.clip_length])  # Convert labels to tensor

        return clip, label

class SlidingWindowSqDataset(Dataset):
    def __init__(self, sequences, labels, window_size, stride):
        self.sequences = sequences
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.subsequences = []
        self.subsequence_labels = []

        for seq, label in zip(sequences, labels):
            for i in range(0, len(seq) - window_size + 1, stride):
                self.subsequences.append(seq[i:i+window_size])
                self.subsequence_labels.append(label)

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, index):
        subsequence = torch.tensor(self.subsequences[index], dtype=torch.float32)
        label = torch.tensor(self.subsequence_labels[index], dtype=torch.int16)
        return subsequence, label

def getSubsetIdx(lenDataset=320, percentages=[0.60, 0.80, 0.90, 1.00], cpus=[1,2,4,8]):
    lenDataset
    # Percentages defining the *end* points of the segments
    indices_segments = {}
    start_index = 0
    start_percentage_int = 0 # To keep track of the starting percentage for the key

    print(f"Generating segments for lenDataset = {lenDataset}")
    print("-" * 30)

    for _c, p in zip(cpus, percentages):
        # Calculate the integer end index for the current segment
        # This index is exclusive for range()
        end_index = int(lenDataset * p)
        # Get the integer percentage for the key string
        end_percentage_int = int(p * 100)

        # Create the key string, e.g., "0-60%", "60-80%"
        key = f"{start_percentage_int}-{end_percentage_int}%"

        # Generate the list of indices for this segment
        # range(start, end) goes from start up to (but not including) end
        indices = list(range(start_index, end_index))

        # Store in the dictionary
        indices_segments[key] = indices

        # Print segment info
        if indices: # Check if the list is not empty
            print(f"Cpus: {_c} | Segment {key}: Length = {len(indices)} (Indices {indices[0]}-{indices[-1]})")
        else:
            print(f"Segment {key}: Length = {len(indices)} (Empty segment)") # Handle cases like consecutive percentages

        # Update the starting point for the *next* segment
        start_index = end_index
        start_percentage_int = end_percentage_int

    print("-" * 30)
    # Verify total length
    total_indices = sum(len(v) for v in indices_segments.values())
    print(f"\nTotal number of indices across all segments: {total_indices} (Should be {lenDataset})")
    
    return indices_segments 

