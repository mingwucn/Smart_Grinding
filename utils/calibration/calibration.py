import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def _load_snapshot(snap_name, model, optimizer):
    loc = f"cuda"
    checkpoint_path = f"../lfs/weights/{snap_name}.pt"
    snapshot = torch.load(checkpoint_path, map_location=loc, weights_only=True)

    _state_dict = snapshot["model_state_dict"]
    model.load_state_dict(_state_dict)
    optimizer.load_state_dict(snapshot["optimizer_state_dict"])
    epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Resuming training from snapshot at Epoch {epochs_run}")

def inverse_min_max_scale(scaled_data, original_min, original_max):
    """Scales 1D data back to the original range."""
    return scaled_data * (original_max - original_min) + original_min

def min_max_scale(data,d_min=None,d_max=None):
    """Scales 1D data to the range of 0 to 1."""
    min_val = np.min(data) if d_min == None else d_min
    max_val = np.max(data) if d_max == None else d_max
    return (data - min_val) / (max_val - min_val)

def predict_in_chunks(model, large_inputs, chunk_size=1000):
    import gc
    """Predicts a large dataset in chunks to avoid memory issues.

    Args:
        model: The trained model.
        large_inputs: A large NumPy array of input data.
        chunk_size: The size of each chunk to predict.

    Returns:
        A NumPy array of predictions.
    """

    num_chunks = np.ceil(len(large_inputs) / chunk_size).astype(int)
    predictions = []

    for i in tqdm(range(num_chunks)):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(large_inputs))
        chunk = large_inputs[start_idx:end_idx]
        chunk_predictions = model.forward(chunk)
        predictions.extend(chunk_predictions.cpu().detach().numpy())

        del chunk
        del chunk_predictions
        gc.collect()

    return np.array(predictions)

def interpolate(x):
    # Known points
    x1_prime, x2_prime = 24.5, 236.1
    x1, x2 = 24.64407183, 236.1617513
    
    # Linear interpolation
    x_prime = x1_prime + ((x - x1) / (x2 - x1)) * (x2_prime - x1_prime)
    
    return x_prime

def get_temp_field(file_path, frame, cali_file = "./cali_0-250.csv"):
    reader = cv2.VideoCapture(file_path)
    reader.set(cv2.CAP_PROP_FORMAT, -1)

    if not reader.isOpened():
        print("Error opening video file")

    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    reader.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, data = reader.read()
    img = data
    input_view = img.view(np.int16).reshape(height, width)[1:, :]

    cali_df_0 = pd.read_csv(f"{cali_file}",index_col=0,header=None,delimiter='\s+').iloc[:,0]
    input_cali_0 = np.around(np.array([interpolate(cali_df_0.loc[f'{i}']) for i in input_view.flatten()]).reshape(height-1,width),1)
    return input_cali_0, num_frames

class IndividualLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 5000)
        # self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(5000, 1)

        self.softmax = nn.Softmax(dim=1)
        self.relu1 = nn.ReLU(100)
        self.relu2 = nn.ReLU(200)
        self.relu3 = nn.ReLU(1)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.fc2(x)
        # x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        # x = self.relu3(x)
        return x

# Define the gating network (decides which expert to use)
class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_size, num_experts)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        logits = self.fc(x)
        gates = self.softmax(logits)
        return gates

# Define the expert network (one expert is just a simple linear model here)
class Expert(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.fc(x)

# Define the overall Mixture of Experts model
class MixtureOfExperts(nn.Module):
    """"
    a Mixture of Experts approach, where the input space is partitioned, and different models (experts) handle different regions of the input space. The model consists of a gating network that selects which expert to use for a given input and a set of simple linear experts that approximate the function in each region.
    """
    def __init__(self, input_size, num_experts):
        super().__init__()
        self.gating_network = GatingNetwork(input_size, num_experts)
        self.experts = nn.ModuleList([Expert(input_size) for _ in range(num_experts)])
    
    def forward(self, x):
        # Get the gate values (which determine which expert is used)
        gates = self.gating_network(x)
        
        # Get expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Weighted sum of experts based on gate values
        output = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        return output

class MyDataset(Dataset):
    def __init__(self, input_data, output_data, transform=None):
        # self.data_path = data_path
        self.transform = transform
        # Load your data here
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.output_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

if __name__ == "__main__":
    min_i = -500
    max_i = 3976
    max_label = 236.2
    min_label = 24.8

    snap_name = "calibration_piecewise"
    model = IndividualLinear()
    model = MixtureOfExperts(input_size=1, num_experts=500)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model = model.to("cuda")
    _load_snapshot(snap_name, model, optimizer)

    file_path = os.path.join("../lfs", "test_data", "sample IR data.ravi")
    reader = cv2.VideoCapture(file_path)
    reader.set(cv2.CAP_PROP_FORMAT, -1)

    if not reader.isOpened():
        print("Error opening video file")

    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reader.set(cv2.CAP_PROP_POS_FRAMES, num_frames // 2)
    # print("Number of frames:", num_frames)
    # print("Size:", (width, height))

    ret, data = reader.read()
    img = data
    input_data = data.view(np.int16).reshape(height, width)[1:, :].flatten()
    # input_data = torch.tensor(min_max_scale(input_data,min_i,max_i))

    label = pd.read_csv(
        os.path.join("../lfs", "test_data", "frame_70.csv"),
        delimiter=";",
        index_col=None,
        header=None,
    ).to_numpy()[:, :-1]

    label = label.flatten()
    # label = torch.tensor(min_max_scale(label.flatten(),min_label,max_label))
    dataset = MyDataset(input_data, label)
    dataloader = DataLoader(dataset, batch_size=307200, shuffle=True)

    num_epochs = 100000
    for epoch in tqdm(range(num_epochs)):
        for i, _data in enumerate(dataloader):
            input_data, label = _data
            input_data = input_data.to("cuda").float().unsqueeze(1)
            label = label.to("cuda").float().unsqueeze(1)
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, label) * 100
            loss.backward()
            optimizer.step()
                # print(f"loss: {loss.item():.4f}")

        if epoch%100 ==0:
            state_key = model.state_dict()
            snapshot = {
                "model_state_dict": state_key,
                "optimizer_state_dict": optimizer.state_dict(),
                "EPOCHS_RUN": epoch,
            }
            torch.save(snapshot, f"../lfs/weights/{snap_name}.pt")
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.4f}")
    torch.save(snapshot, f"../lfs/weights/{snap_name}.pt")
    print(f"Epoch {epoch} | Training snapshot saved at ../lfs/weights/{snap_name}.pt")
