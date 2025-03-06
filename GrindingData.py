sys.path.append("../utils/")
from utils.fusion import (
    compute_bdi,
    compute_ec,
    compute_st,
    process_vibration,
    process_ae,
    process_triaxial_vib,
)

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
import os
import glob
import seaborn as sns
import itertools
import gc
import time
import scienceplots
import sys
import librosa
from nptdms import TdmsFile
from scipy import stats
from natsort import natsorted

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import seaborn as sns
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
matplotlib.rcParams["text.usetex"] = True
plt.rcParams["text.usetex"] = True
matplotlib.rcParams["figure.dpi"] = 300
plt.style.use(["science", "nature"])
plt.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams.update(one_column)

import sys

sys.path.append("./utils")
# from pydub import AudioSegment
import itertools
import string
import glob
import subprocess
import seedir
from utils.preprocessing import print_tdms_structure, check_identical_csv_lengths
from utils.preprocessing import (
    linearSpectrogram,
    logMelSpectrogram,
    melSpectrogram,
    logSpectrogram,
    standardize_array,
    slice_indices,
)
from tqdm import tqdm


class GrindingData:
    def __init__(self, project_dir: str):
        self.dataDir_ae = os.path.join(project_dir, "AE")
        self.dataDir_vib = os.path.join(project_dir, "Vibration")
        print(f"AE data directory: {self.dataDir_ae}")
        print(f"Vibration data directory: {self.dataDir_vib}")
        self.sampling_rate_ae = 4 * 1e6
        self.sampling_rate_vib = 51.2 * 1e3
        self.project_dir = project_dir

        self.n_fft = 1024
        self.n_fft_vib = 512
        self.hop_length = self.n_fft // 2
        self.hop_length_vib = self.n_fft_vib // 2
        self.window_type = "hann"
        self.mel_bins = 256
        self._load_ae_names()
        self._load_parameters()
        self._load_surface_roughness()

    def _load_parameters(self):
        df = pd.read_excel(os.path.join(self.project_dir, "parameters.xlsx"), index_col=0)
        df.columns = ["Surface speed", "Workpiece rotation speed", "Grinding depth"]
        df["Surface speed"] = (
            df["Surface speed"].str.extract(r"(\d+)").astype(float)
        )  # mm/s
        df["Workpiece rotation speed"] = (
            df["Workpiece rotation speed"].str.extract(r"(\d+)").astype(float)
        )  # rpm
        df["Grinding depth"] = (
            df["Grinding depth"].str.extract(r"(\d+)").astype(float)
        )  # um
        self.parameters = df.copy()
        del df

    def _load_surface_roughness(self):
        df = pd.read_csv(
            os.path.join(self.project_dir, "surface roughness.csv"), index_col=None
        )
        df.columns = ["Surface roughness"]
        self.sr = df.iloc[:, 0].to_numpy()

    def _load_ae_names(self):
        self.ae_names = natsorted(
            [
                os.path.join(self.dataDir_ae, i)
                for i in os.listdir(self.dataDir_ae)
                if os.path.splitext(i)[1] == ".txt"
            ]
        )

    def _construct_data(self, process_type="physics"):
        for ae_name in tqdm(self.ae_names):
            if process_type == "physics":
                self._process_file_physic(ae_name)
            elif process_type == "spec":
                self._process_file_spec(ae_name)

    def _construct_data_mp(self, num_threads=None,process_type="physics"):
        # Use the number of CPU cores if num_threads is not specified
        if num_threads is None:
            num_threads = os.cpu_count()

        # Use ThreadPoolExecutor to run _process_file in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks to the executor
            if process_type == "physics":
                futures = [
                    executor.submit(self._process_file_physic, ae_name)
                    for ae_name in self.ae_names
                ]
            elif process_type == "spec":
                futures = [
                    executor.submit(self._process_file_spec, ae_name)
                    for ae_name in self.ae_names
                ]

            # # Wait for all tasks to complete
            # for future in futures:
            #     future.result()  # Ensures all tasks are completed

            # Add tqdm progress bar
            with tqdm(
                total=len(futures),
                desc="Total Progress",
                unit="file",
                dynamic_ncols=True,
            ) as pbar:
                # Update progress as each future completes
                for future in as_completed(futures):
                    future.result()  # Ensure completion
                    pbar.update(1)  # Increment progress bar

            print("All files processed")

    def _process_file_physic(self, ae_name: str):
        _fn = os.path.split(ae_name)[1].split(".")[0]

        # Check if the file already exists and has a non-zero size
        save_dir = os.path.join(self.project_dir, "intermediate")
        save_path = os.path.join(save_dir, f"{_fn}_physics.npz")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            print(f"Skipping {_fn}_physics.npz as it already exists and is not empty.")
            return

        print(f"Processing :{_fn}")
        # spec_ae_list = []
        # spec_vib_list = []
        wavelet_energy_narrow_list = []
        wavelet_energy_broad_list = []
        burst_rate_narrow_list = []
        burst_rate_broad_list = []
        ec_list = []
        bid_list = []
        st_list = []
        env_kurtosis_list_x = []
        env_kurtosis_list_y = []
        env_kurtosis_list_z = []
        mag_list = []
        # print(f"Processing {ae_name}")
        _fn = os.path.split(ae_name)[1].split(".")[0]
        p_n = int(_fn.split("-")[0]) - 1

        v_s = self.parameters.iloc[p_n]["Surface speed"]
        v_w = self.parameters.iloc[p_n]["Workpiece rotation speed"]
        a_p = self.parameters.iloc[p_n]["Grinding depth"]

        ec = compute_ec(v_s, v_w, a_p)
        bdi = compute_bdi(a_p)
        st = compute_st(v_s, a_p, 10)

        vib_data = TdmsFile.read(
            os.path.join(dataDir_vib, f"{_fn[:-2]}", "Test_Data.tdms")
        ).groups()
        vib_x = vib_data[0].channels()[0].data[:]
        vib_y = vib_data[0].channels()[1].data[:]
        vib_z = vib_data[0].channels()[2].data[:]

        ae_df = pd.read_csv(ae_name, sep="\s", header=None, engine="python")
        print(f"[P] AE data shape: {ae_df.shape}")
        # ae_narrow = np.loadtxt(ae_name, usecols=0, dtype=np.float32)  
        # ae_broad = np.loadtxt(ae_name, usecols=1, dtype=np.float32)

        ae_narrow = ae_df[0]
        ae_broad = ae_df[1]
        ae_indices = slice_indices(len(ae_narrow), int(self.sampling_rate_ae * 0.01), 0.5)
        vib_indices = slice_indices(len(vib_x), int(self.sampling_rate_vib * 0.1), 0.5)
        window_n = min(len(vib_indices) * 10, len(ae_indices))
        print(f"window number:{window_n} for {ae_name}")
        del vib_data
        del ae_df
        gc.collect()

        for idx in tqdm(range(window_n)):
            # print(f"Processing {idx}/{window_n} for {ae_name}")
            _i0, _it = ae_indices[idx]
            _i0_vib, _it_vib = vib_indices[int(idx // 10)]

            _data_vib_x = np.array(vib_x[_i0_vib:_it_vib])
            _data_vib_y = np.array(vib_y[_i0_vib:_it_vib])
            _data_vib_z = np.array(vib_z[_i0_vib:_it_vib])

            _data_narrow = np.array(ae_narrow[_i0:_it])
            _data_broad = np.array(ae_broad[_i0:_it])
            ae_info_narrow = process_ae(_data_narrow, self.sampling_rate_ae, mask_energy=(50e3,  400e3), burst_threshold=4, time_spacing=1e-4)
            ae_info_broad  = process_ae(_data_broad,  self.sampling_rate_ae, mask_energy=(150e3, 250e4), burst_threshold=4, time_spacing=1e-4)
            spec_narrow = logSpectrogram(
                data=_data_narrow,
                sampling_rate=self.sampling_rate_ae,
                display=False,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window_type=self.window_type,
            )
            spec_broad = logSpectrogram(
                data=_data_broad,
                sampling_rate=self.sampling_rate_ae,
                display=False,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window_type=self.window_type,
            )
            spec_ae = np.stack([spec_narrow[:300,:], spec_broad[:300,:]], axis=0)

            spec_vib_x = logSpectrogram(
                data=_data_vib_x,
                sampling_rate=self.sampling_rate_vib,
                display=False,
                n_fft=self.n_fft_vib,
                hop_length=self.hop_length_vib,
                window_type=self.window_type,
            )
            spec_vib_y = logSpectrogram(
                data=_data_vib_y,
                sampling_rate=self.sampling_rate_vib,
                display=False,
                n_fft=self.n_fft_vib,
                hop_length=self.hop_length_vib,
                window_type=self.window_type,
            )
            spec_vib_z = logSpectrogram(
                data=_data_vib_z,
                sampling_rate=self.sampling_rate_vib,
                display=False,
                n_fft=self.n_fft_vib,
                hop_length=self.hop_length_vib,
                window_type=self.window_type,
            )

            vib_info = process_triaxial_vib(
                _data_vib_x, _data_vib_y, _data_vib_z, self.sampling_rate_vib
            )
            vib_spec = np.stack(
                [
                    spec_vib_x,
                    spec_vib_y,
                    spec_vib_z,
                ],
                axis=0,
            )

            wavelet_energy_narrow_list.append(np.abs(ae_info_narrow["wavelet_energy"]))
            wavelet_energy_broad_list.append(np.abs(ae_info_broad["wavelet_energy"]))
            burst_rate_narrow_list.append(ae_info_narrow["burst_rate"])
            burst_rate_broad_list.append(ae_info_broad["burst_rate"])
            ec_list.append(ec)
            bid_list.append(bdi)
            st_list.append(st)
            # spec_ae_list.append(spec_ae)
            # spec_vib_list.append(vib_spec)

            env_kurtosis_list_x.append(vib_info["x_env_kurtosis"])
            env_kurtosis_list_y.append(vib_info["y_env_kurtosis"])
            env_kurtosis_list_z.append(vib_info["z_env_kurtosis"])
            mag_list.append(vib_info["mag_mesh_amp"])
            # del _data_narrow, _data_broad, _data_vib_x, _data_vib_y, _data_vib_z
            # gc.collect()

        # spec_ae = np.stack(spec_ae_list, axis=0)
        # spec_vib = np.stack(spec_vib_list, axis=0)
        wavelet_energy_narrow = np.stack(wavelet_energy_narrow_list, axis=0)
        wavelet_energy_broad = np.stack(wavelet_energy_broad_list, axis=0)
        burst_rate_narrow = np.stack(burst_rate_narrow_list, axis=0)
        burst_rate_broad = np.stack(burst_rate_broad_list, axis=0)
        ec = np.stack(ec_list, axis=0)
        bid = np.stack(bid_list, axis=0)
        st = np.stack(st_list, axis=0)
        env_kurtosis_x = np.stack(env_kurtosis_list_x, axis=0)
        env_kurtosis_y = np.stack(env_kurtosis_list_y, axis=0)
        env_kurtosis_z = np.stack(env_kurtosis_list_z, axis=0)
        mag = np.stack(mag_list, axis=0)

        # make these np arrays as a dictionary
        data = {
            # "spec_ae": spec_ae,
            # "spec_vib": spec_vib,
            "wavelet_energy_narrow": wavelet_energy_narrow,
            "wavelet_energy_broad": wavelet_energy_broad,
            "burst_rate_narrow": burst_rate_narrow,
            "burst_rate_broad": burst_rate_broad,
            "ec": ec,
            "bid": bid,
            "st": st,
            "env_kurtosis_x": env_kurtosis_x,
            "env_kurtosis_y": env_kurtosis_y,
            "env_kurtosis_z": env_kurtosis_z,
            "mag": mag,
        }
        self._save_data(data, f"{_fn}_physics")

    def _process_file_spec(self, ae_name: str):
        _fn = os.path.split(ae_name)[1].split(".")[0]

        # Check if the file already exists and has a non-zero size
        save_dir = os.path.join(self.project_dir, "intermediate")
        save_path = os.path.join(save_dir, f"{_fn}_spec.npz")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            print(f"Skipping {_fn}_spec.npz as it already exists and is not empty.")
            return
        print(f"Processing :{_fn}")

        spec_ae_list = []
        spec_vib_list = []
        # wavelet_energy_narrow_list = []
        # wavelet_energy_broad_list = []
        # burst_rate_narrow_list = []
        # burst_rate_broad_list = []
        # ec_list = []
        # bid_list = []
        # st_list = []
        # env_kurtosis_list_x = []
        # env_kurtosis_list_y = []
        # env_kurtosis_list_z = []
        # mag_list = []
        # print(f"Processing {ae_name}")
        p_n = int(_fn.split("-")[0]) - 1

        v_s = self.parameters.iloc[p_n]["Surface speed"]
        v_w = self.parameters.iloc[p_n]["Workpiece rotation speed"]
        a_p = self.parameters.iloc[p_n]["Grinding depth"]

        vib_data = TdmsFile.read(
            os.path.join(dataDir_vib, f"{_fn[:-2]}", "Test_Data.tdms")
        ).groups()
        vib_x = vib_data[0].channels()[0].data[:]
        vib_y = vib_data[0].channels()[1].data[:]
        vib_z = vib_data[0].channels()[2].data[:]

        ae_df = pd.read_csv(ae_name, sep="\s", header=None, engine="python")
        print(f"[S] AE data shape: {ae_df.shape}")
        # ae_narrow = np.loadtxt(ae_name, usecols=0, dtype=np.float32)
        # ae_broad = np.loadtxt(ae_name, usecols=1, dtype=np.float32)
        ae_narrow = ae_df[0]
        ae_broad = ae_df[1]
        ae_indices = slice_indices(len(ae_narrow), int(self.sampling_rate_ae * 0.01), 0.5)
        vib_indices = slice_indices(len(vib_x), int(self.sampling_rate_vib * 0.1), 0.5)
        window_n = min(len(vib_indices) * 10, len(ae_indices))
        print(f"window number:{window_n} for {ae_name}")
        del vib_data
        del ae_df
        gc.collect()

        for idx in tqdm(range(window_n)):
            # print(f"Processing {idx}/{window_n} for {ae_name}")
            _i0, _it = ae_indices[idx]
            _i0_vib, _it_vib = vib_indices[int(idx // 10)]

            _data_vib_x = np.array(vib_x[_i0_vib:_it_vib])
            _data_vib_y = np.array(vib_y[_i0_vib:_it_vib])
            _data_vib_z = np.array(vib_z[_i0_vib:_it_vib])

            _data_narrow = np.array(ae_narrow[_i0:_it])
            _data_broad = np.array(ae_broad[_i0:_it])
            spec_narrow = logSpectrogram(
                data=_data_narrow,
                sampling_rate=self.sampling_rate_ae,
                display=False,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window_type=self.window_type,
            )
            spec_broad = logSpectrogram(
                data=_data_broad,
                sampling_rate=self.sampling_rate_ae,
                display=False,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window_type=self.window_type,
            )
            spec_ae = np.stack([spec_narrow[:300,:], spec_broad[:300,:]], axis=0)

            spec_vib_x = logSpectrogram(
                data=_data_vib_x,
                sampling_rate=self.sampling_rate_vib,
                display=False,
                n_fft=self.n_fft_vib,
                hop_length=self.hop_length_vib,
                window_type=self.window_type,
            )
            spec_vib_y = logSpectrogram(
                data=_data_vib_y,
                sampling_rate=self.sampling_rate_vib,
                display=False,
                n_fft=self.n_fft_vib,
                hop_length=self.hop_length_vib,
                window_type=self.window_type,
            )
            spec_vib_z = logSpectrogram(
                data=_data_vib_z,
                sampling_rate=self.sampling_rate_vib,
                display=False,
                n_fft=self.n_fft_vib,
                hop_length=self.hop_length_vib,
                window_type=self.window_type,
            )

            vib_spec = np.stack(
                [
                    spec_vib_x,
                    spec_vib_y,
                    spec_vib_z,
                ],
                axis=0,
            )

            spec_ae_list.append(spec_ae)
            spec_vib_list.append(vib_spec)
            # del _data_narrow, _data_broad, _data_vib_x, _data_vib_y, _data_vib_z
            # gc.collect()

        spec_ae = np.stack(spec_ae_list, axis=0)
        spec_vib = np.stack(spec_vib_list, axis=0)
        # wavelet_energy_narrow = np.stack(wavelet_energy_narrow_list, axis=0)
        # wavelet_energy_broad = np.stack(wavelet_energy_broad_list, axis=0)
        # burst_rate_narrow = np.stack(burst_rate_narrow_list, axis=0)
        # burst_rate_broad = np.stack(burst_rate_broad_list, axis=0)
        # ec = np.stack(ec_list, axis=0)
        # bid = np.stack(bid_list, axis=0)
        # st = np.stack(st_list, axis=0)
        # env_kurtosis_x = np.stack(env_kurtosis_list_x, axis=0)
        # env_kurtosis_y = np.stack(env_kurtosis_list_y, axis=0)
        # env_kurtosis_z = np.stack(env_kurtosis_list_z, axis=0)
        # mag = np.stack(mag_list, axis=0)

        # make these np arrays as a dictionary
        data = {
            "spec_ae": spec_ae,
            "spec_vib": spec_vib,
            # "wavelet_energy_narrow": wavelet_energy_narrow,
            # "wavelet_energy_broad": wavelet_energy_broad,
            # "burst_rate_narrow": burst_rate_narrow,
            # "burst_rate_broad": burst_rate_broad,
            # "ec": ec,
            # "bid": bid,
            # "st": st,
            # "env_kurtosis_x": env_kurtosis_x,
            # "env_kurtosis_y": env_kurtosis_y,
            # "env_kurtosis_z": env_kurtosis_z,
            # "mag": mag,
        }
        # self._save_data(data, _fn)
        self._save_data(data, f"{_fn}_spec")

    def _save_data_pkl(self, data, filename: str, save_dir: str = None):
        if save_dir is None:
            save_dir = os.path.join(self.project_dir, "intermediate")
            # save_dir = os.path.join(os.getcwd(), "intermediate")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("Saving data to", os.path.join(save_dir, f"{filename}.pkl"))
        pickle.dump(data, open(os.path.join(save_dir, f"{filename}.pkl"), "wb"))
        print(
            "Data disk size:",
            os.path.getsize(os.path.join(save_dir, f"{filename}.pkl")) / 1e6,
            "MB",
        )

    def _save_data(self, data, filename: str, save_dir: str = None):
        if save_dir is None:
            save_dir = os.path.join(self.project_dir, "intermediate")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{filename}.npz")
        
        # Convert all numpy arrays to float32 and other optimizations
        compressed_data = {}
        for key in data:
            if isinstance(data[key], np.ndarray):
                # Preserve integer types for count-based data
                # if 'burst_rate' in key or 'bid' in key:
                #     compressed_data[key] = data[key].astype(np.uint16)
                # else:
                compressed_data[key] = data[key].astype(np.float32)
            else:
                compressed_data[key] = data[key]
        
        np.savez_compressed(save_path, **compressed_data)
        print(f"Saved {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

if __name__ == "__main__":
    import time

    # add arguments to the script
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Grinding data processing")
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="Number of threads to use for parallel processing",
    )
    parser.add_argument(
        "--process_type",
        type=str,
        default='physics',
        help="Process type. physics or spec",
    )
    args = parser.parse_args()
    print(f"Number of threads: {args.threads}")
    print(f"Process type: {args.process_type}")

    start_time = time.time()
    alphabet = list(string.ascii_lowercase)
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
    if args.threads == 1:
        grinding_data._construct_data(process_type=args.process_type)
    else:
        grinding_data._construct_data_mp(num_threads=args.threads,process_type=args.process_type)
    # intermediate_dir = os.path.join(project_dir, "intermediate")
    # print(f"Saving data to {intermediate_dir}")
    # if not os.path.exists(intermediate_dir):
    #     os.makedirs(intermediate_dir)
    # pickle.dump(
    #     grinding_data, open(os.path.join(intermediate_dir, "grinding_data.pkl"), "wb")
    # )
    print(f"Time taken: {time.time()-start_time:.2f} seconds")
