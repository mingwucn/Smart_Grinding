import sys
import os
import math
import numpy as np
import scipy
import subprocess
from natsort import natsorted
import string
sys.path.append("./../utils")
# sys.path.append("./cpp_utils")
sys.path.append("./..")
from utils.preprocessing import process_trajectory_with_dask_shared_memory,create_shared_memory_array,Sender, MaPS_LPBF_Construction,MaPS_LPBF_Point_Wise_Construction,normalize_array,pulse_signal_slicer_by_interval
from utils.preprocessing import cm_std
from utils.MLUtils import get_windowed_data
from utils.InterfaceDeclaration import LPBFPointData
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision
from torchvision.transforms import v2
import torch
import cv2
import dask
import dask.array as da
import multiprocessing
from multiprocessing import shared_memory, Lock
from tqdm import tqdm
import pickle
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset, DataLoader, DistributedSampler 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, StandardScaler,LabelEncoder
from scipy.interpolate import interp1d
# import cpp_utils

alphabet = list(string.ascii_lowercase)

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

# Step 1: Function to find neighboring points within the ROI radius
def find_neighbors_within_radius(lmq_map, x_center, y_center, radius):
    """
    return:
        line_index, point_index, distance between center xy and start point, angle between center xy and start point
    """
    neighbors = []
    for line_index, line in enumerate(lmq_map):
        for point_index, (x, y) in enumerate(line):
            distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
            if distance <= radius:
                neighbors.append((line_index, point_index, distance, math.atan2(y - y_center, x - x_center)))
    return neighbors

# Step 2: Function to compress information for neighboring points
def compress_neighbor_info(neighbors):
    """
    Compress the information for neighboring points.
    - Only consider neighbors that belong to the same line.
    - Sort them by point_index.
    - Compress the information by using the first and last points on the line as references.

    Returns:
        compressed information:
        - index_start: the line where the neighbors belong
        - start_index, end_index: indices of the first and last neighbor points on the line
        - relative_distance_start, relative_distance_end: distances of the first and last points
        - relative_direction_start, relative_direction_end: angles of the first and last points
    """
    if not neighbors:
        return []
    
    compressed_info = []
    neighbors.sort(key=lambda n: n[1])  # Sort by distance, therefore the first line will be the line of the point itself
    # Group neighbors by line_index
    neighbors_by_line = {}
    for neighbor in neighbors:
        line_index = neighbor[0]
        if line_index not in neighbors_by_line:
            neighbors_by_line[line_index] = []
        neighbors_by_line[line_index].append(neighbor)

    # Delete the item with only one point
    keys_to_remove = [key for key, value in neighbors_by_line.items() if len(value) == 1] 
    for key in keys_to_remove: 
        del neighbors_by_line[key]

    # Process each line separately
    for line_index, line_neighbors in neighbors_by_line.items():
        # Sort neighbors by point_index within the same line
        line_neighbors.sort(key=lambda n: n[1])  # Sort by point_index

        # First and last points in the sorted neighbors
        first_neighbor = line_neighbors[0]
        last_neighbor = line_neighbors[-1]

        # Compress relative distances and directions
        compressed_info.append([
            line_index,
            first_neighbor[2], # distance
            last_neighbor[2], # distance
            first_neighbor[3], # angle
            last_neighbor[3], # angle
            first_neighbor[1],
            last_neighbor[1]
        ])

    return compressed_info

# Step 3: Function to map coordinates from lmq_map to daq_map based on line length differences
def map_to_daq_map(lmq_map_info, daq_map_lengths, lmq_map_lengths):
    mapped_info = []
    
    for line_info in lmq_map_info:
        line_index, dist_start, dist_end, dir_start, dir_end, index_start, index_end = line_info
        # Mapping indices to daq_map
        daq_index_start = int(index_start * (daq_map_lengths[line_index] / lmq_map_lengths[line_index]))
        daq_index_end = int(index_end * (daq_map_lengths[line_index] / lmq_map_lengths[line_index]))
        
        # Append mapped result to list
        mapped_info.append([
            line_index, dist_start, dist_end, dir_start, dir_end, daq_index_start, daq_index_end
        ])
    
    return mapped_info

# Combine step 1-3
def signal_within_radius(xy_map,x_center, y_center, radius,daq_map_lengths,lmq_map_lengths):
    """
    Function to find the compressed info of points within an roi radius for a point

    Args:
        xy_map: Nested lists of xy coordinates, [(x1,y1),(x2,y2),...]
        x_center: the x coordinate of the center point
        y_center: the y coordinate of the center point
        radius: the roi radius
        daq_map_lengths: the list of all the lengths of the recorded data for each printing vector
        lmq_map_lengths: the list of all the lengths of the controller data for each printing vector
    
    Return:
        A list of compressed information of the recorded signal within a certain radius of RoI. 
        [Line_index, start_index, end_index, relative_direction_start, relative_direction_end, relative_distance_start, relative_distance_end]
        - Line_index: the index of the printing vector in a certain layer where the neighbors belong
        - start_index, end_index: indices of the first and last neighbor points on the line
        - relative_direction_start, relative_direction_end: angles of the first and last points
        - relative_distance_start, relative_distance_end: distances of the first and last points
    
    The [relative_direction_start, relative_direction_end, relative_distance_start, relative_distance_end] can be used to embedded the printing object's geometrical information. 
    """
    neighbors = find_neighbors_within_radius(xy_map,x_center, y_center, radius)
    compressed_info = compress_neighbor_info(neighbors)
    daq_mapped_info = map_to_daq_map(compressed_info,daq_map_lengths,lmq_map_lengths)
    daq_mapped_info.sort(key=lambda n: n[0])
    return daq_mapped_info

def geometrical_embedded_offset(daq_mapped_info,x_center,y_center):
    """
    Using the the longest printing vector as the reference, and calculate the relative offset for each line. Hence, the geometrical information can be embedded via these offsets.  

    Args:
        - daq_mapped_info: A list of compressed information of the recorded signal within a certain radius of RoI. 
        [Line_index, start_index, end_index, relative_direction_start, relative_direction_end, relative_distance_start, relative_distance_end]
    Return:
        - first_line_direction: 0: left to right; 1: right to left
        - start_offset: A list of offsets at the start point for each printing vector
    """
    ae_len = []
    _s0s =[]
    _s1s =[]
    point_pairs = []
    first_line_direction = 0

    for i, _info in enumerate(daq_mapped_info):
        _line_idx, _dist_0, _dist_t, _angle_0, _angle_t, _idx_0, _ind_t= _info
        _ae_length = np.ptp([_idx_0,_ind_t]) # Signal length of recorded data
        ae_len.append(_ae_length)
        _s0 = calculate_relative_point(x_center,y_center,_info[1],_info[3]) # Start point in the controller data
        _s1 = calculate_relative_point(x_center,y_center,_info[2],_info[4]) # End point in the controller data
        _dist = calculate_distance(_s0,_s1) # Distance, i.e. the length of this signal for the controller data
        # lmq_distance.append(calculate_distance(_s0,_s1))
        _sorted_p = sorted([_s0, _s1], key=lambda p: calculate_distance(p, (0, 0))) # For each the vector, sort their coordinate to ensure the one closer to Point (0,0) will be left. 
        if i==0:
                first_line_direction = 0 if _sorted_p[0] == _s0 else 1
        _s0s.append(_sorted_p[0])
        _s1s.append(_sorted_p[1])
        if _dist == 0:
            if i == 0:
                _ratio = 1
            else:
                _ratio = ae_len[i-1] / calculate_distance(_s0s[i-1], _s1s[i-1]) / sampling_rate_daq * 1e3
        else:
            _ratio = _ae_length / _dist / sampling_rate_daq * 1e3
        point_pairs.append([_sorted_p[0],_sorted_p[1]])

    # This block will center the longest line, and calculate the relative offset for each line. Hence, the geometrical information can be embedded via these offsets.  
    ## Using the longest printing vector as the reference line
    longest_pair_index = find_longest_pair(_s0s, _s1s)
    longest_pair = point_pairs[longest_pair_index]
    # reference_line_start = _s0s[longest_pair_index] 
    # reference_line_end = _s1s[longest_pair_index]
    ## Calculate the angle of the reference vector related to the "absolute coordinate", then rotate all the points to the global horizontal. 
    angle = math.atan2(longest_pair[1][1] - longest_pair[0][1], longest_pair[1][0] - longest_pair[0][0])
    # cos_theta = math.cos(angle)
    # sin_theta = math.sin(angle)
    rotated_point_pairs = rotate_points(point_pairs, -angle)
    ## Get the coordinate of the leftmost point of all the points. 
    leftmost_x = min(p[0] for p in np.array(rotated_point_pairs)[:,0,:])
    ## Calculate the offset added to the start point of the current printing vector based on the leftmost point, and scaled by the ratio of sampling rate difference
    start_offset = (np.array(rotated_point_pairs)[:,0,0]-leftmost_x)*_ratio
    # This block will center the longest line, and calculate the relative offset for each line. Hence, the geometrical information can be embedded via these offsets.  
    return first_line_direction, start_offset

def calculate_relative_point(x0, y0, distance, angle):
    """Calculates the position of a point based on a given point, relative distance, and relative angle.

    Args:
        x0: The x-coordinate of the starting point.
        y0: The y-coordinate of the starting point.
        distance: The distance from the starting point to the relative point.
        angle: The angle (in radians) between the x-axis and the line connecting the starting point and the relative point.

    Returns:
        A tuple representing the x and y coordinates of the relative point.
    """

    x = x0 + distance * math.cos(angle)
    y = y0 + distance * math.sin(angle)
    return [x, y]

def fourier_transform(signals, sampling_rate):
    """Performs 1D Fourier transform on a list of signals.

    Args:
    signals: A list of 1D signals.
    sampling_rate: The sampling rate of the signals.

    Returns:
        A list of Fourier transform results (real part only).
    """

    fourier_results = []
    for signal in signals:
        fft_result = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal), 1 / sampling_rate)
        fourier_results.append(np.real(fft_result[freq >= 0]))

    return fourier_results

def fourier_transform1d_interp(signals, sampling_rate, target_length=None):
    from scipy.interpolate import interp1d
    """
    Computes the real part of the Fourier transform for a list of 1D signals,
    and interpolates them onto a common frequency grid.

    Parameters:
    - signals: List of 1D numpy arrays, each representing a signal.
    - sampling_rate: Sampling rate of the signals (in Hz).
    - target_length: Optional integer specifying the length to which all Fourier
      transforms should be interpolated. If not provided, the maximum length
      of the Fourier transforms will be used.

    Returns:
    - interpolated_fts: List of interpolated real Fourier transform results.
    - common_freqs: Common frequency grid for all signals after interpolation.
    """
    real_fts = []
    freq_grids = []

    # Compute the Fourier Transform for each signal
    for signal in signals:
        if len(signal) != 0:
            # Compute FFT and keep only the real part
            fft_result = np.fft.fft(signal)
            real_fft = np.real(fft_result)
            # Compute the corresponding frequency grid
            n = len(signal)
            freqs = np.fft.fftfreq(n, d=1/sampling_rate)
        else:
            fft_result = np.fft.fft([np.sin(i) for i in np.linspace(0,3.14,314)])
            
            real_fft = np.real(fft_result)
            # Compute the corresponding frequency grid
            n = len(signal)
            freqs = np.fft.fftfreq(n, d=1/sampling_rate)
        
        # Store the real part of the FFT and the frequency grid
        real_fts.append(real_fft[:n//2])  # Take only the positive frequencies
        freq_grids.append(freqs[:n//2])
    
    # Determine the target length for interpolation
    if target_length is None:
        target_length = max(len(fft) for fft in real_fts)
    
    # Create a common frequency grid for interpolation
    common_freqs = np.linspace(0, min(freq_grid.max() for freq_grid in freq_grids), target_length)

    # Interpolate each Fourier transform onto the common frequency grid
    interpolated_fts = []
    for real_fft, freqs in zip(real_fts, freq_grids):
        # Interpolate using linear interpolation
        interpolator = interp1d(freqs, real_fft, kind='linear', fill_value="extrapolate")
        interpolated_fft = interpolator(common_freqs)
        interpolated_fts.append(interpolated_fft)

    return np.asarray(interpolated_fts), common_freqs

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_offsets(points1, points2, longest_pair):
    """Calculates the distance offsets for each pair."""
    longest_distance = calculate_distance(*longest_pair)
    offsets = []
    for p1, p2 in zip(points1, points2):
        distance = calculate_distance(p1, p2)
        offset = longest_distance - distance
        offsets.append(offset)
    return offsets

def find_longest_pair(point1_list, point2_list):
    """
    Finds index for the pair of points with the longest distance.

    Args:
        - List of Coordinate of start points
        - List of Coordinate of end points
    """
    if len(point1_list) != len(point2_list):
        raise ValueError("Lists must have the same length.")
    distances = [calculate_distance(p1, p2) for p1, p2 in zip(point1_list, point2_list)]
    longest_pair_index = distances.index(max(distances))
    return longest_pair_index

def calculate_projection(point, line_start, line_end):
    """Calculates the projection of a point onto a line segment.

    Args:
        point: The point to project.
        line_start: The starting point of the line segment.
        line_end: The ending point of the line segment.

    Returns:
        A tuple containing the projection point and the distance.
    """

    # Calculate the vector representing the line segment
    line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])

    # Calculate the vector from the line start to the point
    point_vector = (point[0] - line_start[0], point[1] - line_start[1])

    # Calculate the projection scalar
    projection_scalar = (point_vector[0] * line_vector[0] + point_vector[1] * line_vector[1]) / (line_vector[0]**2 + line_vector[1]**2)

    # Calculate the projection point
    projection_point = (line_start[0] + projection_scalar * line_vector[0], line_start[1] + projection_scalar * line_vector[1])

    return projection_point 

def rotate_points(points, angle):
    """
    Rotates a list of point pairs [[x1, y1], [x2, y2], ...] as a unified entity by a given angle.
    
    Parameters:
    - points: List of [x, y] coordinate pairs.
    - angle: Rotation angle in radians.
    
    Returns:
    - rotated_points: List of rotated [x', y'] coordinate pairs.
    """
    # Create the 2D rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    # Convert points to a NumPy array for matrix multiplication
    points_array = np.array(points)
    
    # Perform the rotation for all points
    rotated_points = np.dot(points_array, rotation_matrix.T)
    
    return rotated_points.tolist()

def place_signals_on_grid(signals,t_data, max_length, resolution,fill=0):
    """
    Place a list of signals on a uniform grid, interpolating each signal to a specified resolution.
    
    Args:
    - signals (list of np.ndarray): signals with different time ranges and lengths. 
    - max_length (int): The maximum possible length of any signal.
    - resolution (int): The number of points to which each signal should be interpolated.
    - fill: filled value, np.nan or 0
    
    Returns:
    - interpolated_signals (np.ndarray): A 2D numpy array where each row is an interpolated signal.
    """
    # Step 1: Define a uniform time grid
    time_grid = np.arange(0, max_length, resolution)

    # Step 2: Interpolate each signal onto the common time grid
    interpolated_signals = []
    for t, signal in zip(t_data, signals):
        if len(t) == 0 or len(signal) == 0:
            interpolated_signals.append(np.full_like(time_grid, fill))
            continue
        # Define interpolation function with extrapolation for values outside the range
        interp_func = interp1d(t, signal, kind='linear', bounds_error=False, fill_value=fill)
        # Interpolate signal onto the common time grid
        interpolated_signal = interp_func(time_grid)
        interpolated_signals.append(interpolated_signal)

    # Step 3: Stack interpolated signals into a 2D array
    if not interpolated_signals:
        signal_matrix = np.zeros((1, len(time_grid)))
    else:
        signal_matrix = np.vstack(interpolated_signals)
    return signal_matrix

def from_xy_get_signal_matrix(center_x, center_y,xy_map,daq_map_lengths,lmq_map_lengths, ae_collection,sampling_rate_daq,max_length=1.5, resolution=0.002,fill=0,roi_radius=100):
    """
    a over all function combine: 
        - signal_within_radius
        - geometrical_embedded_offset
        - place_signals_on_grid
    to get the signal_matrix
    """
    daq_mapped_info = signal_within_radius(xy_map,center_x,center_y,roi_radius,daq_map_lengths,lmq_map_lengths)
    if len(daq_mapped_info)==0:
        return np.zeros([5,int(max_length/resolution)])
    first_line_direction, start_offset = geometrical_embedded_offset(daq_mapped_info,center_x,center_y)
    ae_data = []
    t_data = []
    for i, _info  in enumerate(daq_mapped_info):
        _line_idx, _dist_0, _dist_t, _angle_0, _angle_t, _idx_0, _ind_t= _info
        if _ind_t-_idx_0<10:
            continue
        _ae = ae_collection[_line_idx][_idx_0:_ind_t]
        if i%2 != first_line_direction:
            _ae = _ae[::-1]
        _t = np.linspace(start_offset[i],len(_ae)/sampling_rate_daq*1e3+start_offset[i],len(_ae))
        ae_data.append(_ae)
        t_data.append(_t)
    signal_matrix = np.abs(place_signals_on_grid(ae_data,t_data,max_length,resolution,fill))
    return signal_matrix

# def from_xy_get_signal_matrix_pp(center_x, center_y,xy_map,daq_map_lengths,lmq_map_lengths, ae_collection,sampling_rate_daq,max_length=1.5, resolution=0.002,fill=0):
    """
    a over all function combine: 
        - signal_within_radius
        - geometrical_embedded_offset
        - place_signals_on_grid
    to get the signal_matrix
    """
    daq_mapped_info = cpp_utils.signal_within_radius(xy_map,center_x,center_y,100,daq_map_lengths,lmq_map_lengths)
    first_line_direction, start_offset = geometrical_embedded_offset(daq_mapped_info,center_x,center_y)
    ae_data = []
    t_data = []
    for i, _info  in enumerate(daq_mapped_info):
        _line_idx, _dist_0, _dist_t, _angle_0, _angle_t, _idx_0, _ind_t= _info
        _ae = ae_collection[int(_line_idx)][int(_idx_0):int(_ind_t)]
        if i%2 != first_line_direction:
            _ae = _ae[::-1]
        _t = np.linspace(start_offset[i],len(_ae)/sampling_rate_daq*1e3+start_offset[i],len(_ae))
        ae_data.append(_ae)
        t_data.append(_t)
    signal_matrix = np.abs(place_signals_on_grid(ae_data,t_data,max_length,resolution,fill))
    return signal_matrix

def center_pad_axis0(array, new_shape):
    if array.shape[0] >= new_shape[0]:
        raise ValueError("New shape must have more rows than the original array.")

    padding_amount = new_shape[0] - array.shape[0]
    top_padding = padding_amount // 2
    bottom_padding = padding_amount - top_padding
    return np.pad(array, ((top_padding, bottom_padding), (0, 0)), 'constant')

if __name__ =="__main__":
    import argparse
    import json
    import psutil
    import time
    num_physical_cores = psutil.cpu_count(logical=False)
    parser = argparse.ArgumentParser(description='Distributed training job')
    parser.add_argument('--layers', default="249", type=lambda a: json.loads('['+a.replace(" ",",")+']'), help="List of layer to be calculated") 
    parser.add_argument('--roi_radius', type=int, default=100, help=f'ROI radius (default: 100)')
    parser.add_argument('--multiprocessing', type=int, default=True, help=f'Multiprocessing (default: True)')
    parser.add_argument('--keep_core', type=int, default=1, help=f'leave n physical cores un used')
    parser.add_argument('--normal_layer', type=bool, default=False, help=f'Train the normal layers (default: False)')
    parser.add_argument('--cube_i', type=int, default=2, help=f'Cube index (default: 2)')
    args = parser.parse_args()

    print(f"Processing layers: {args.layers}")
    for layer_i in args.layers:
        print(f"Processing layers: {layer_i}")
        lpbf = MaPS_LPBF_Construction(daq_dir=daq_dir,lmq_dir=lmq_dir,sampling_rate_daq=sampling_rate_daq,sampling_rate_lmq=sampling_rate_lmq,process_regime=process_regime,laser_power_setting=laser_power_setting,scanning_speed_setting=scanning_speed_setting)
        _scaling= 149
        x_scaling = 1/3409.76*_scaling
        y_scaling = 1/3376*_scaling
        x_offset = -17049+980/x_scaling
        y_offset = 16883+905/y_scaling
        lpbf._construct_line_wise_labels_ram(layer_i,2)
        lpbf._xy_rescaler(x_offset,y_offset,x_scaling,y_scaling)

        xy = [(lpbf.current_x[i0:it],lpbf.current_y[i0:it]) for i0,it in lpbf.line_indices_lmq[:-1]]
        xy_all = np.array([(x,y) for i in xy for x,y in zip(i[0], i[1])])
        xy_map = []
        for _xy in xy:
            xy_map.append([(x,y) for x,y in zip(_xy[0],_xy[1])])
        daq_map_lengths = [_.ptp() for _ in lpbf.line_indices_daq[:-1]]
        lmq_map_lengths = [_.ptp() for _ in lpbf.line_indices_lmq[:-1]]

        ae_collection = lpbf.ae
        max_length = 1.5
        resolution = 0.002
        fill = 0 
        sig_list = []

        for i in tqdm(range(len(lpbf.current_x))[:]):
            center_x = lpbf.current_x[i]
            center_y = lpbf.current_y[i]
            signal_matrix = from_xy_get_signal_matrix(center_x, center_y,xy_map,daq_map_lengths,lmq_map_lengths, ae_collection,sampling_rate_daq,max_length,resolution,fill)
            signal_matrix = center_pad_axis0(np.abs(signal_matrix),(20,signal_matrix.shape[1]))
            sig_list.append(signal_matrix)
        sig_list = np.array(sig_list)

        layer_wise_sig_dir = os.path.join(data_dir, *project_name, "intermediate","Layer_wise_sig")
        with open(os.path.join(layer_wise_sig_dir,f'layer{layer_i}.npy'), 'wb') as f:
            np.save(f, sig_list)