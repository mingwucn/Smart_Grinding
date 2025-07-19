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
import pickle
from nptdms import TdmsFile
from scipy import stats
from natsort import natsorted
from tqdm import tqdm

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import scienceplots

matplotlib.rcParams['text.usetex'] = True
plt.rcParams["text.usetex"] = True
# import seaborn as sns
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, BSpline

np.random.seed(16)
matplotlib.rcParams['figure.dpi'] = 300

plt.style.use(['science','nature'])
plt.rcParams['figure.constrained_layout.use'] = True

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(["Open","Arc","Short","Normal"])

import sys
sys.path.append('../.')
# from pydub import AudioSegment
import itertools
import string
import glob
import subprocess
import seedir
from utils.preprocessing import one_column,two_column,cm_std,cm_bright,cm_mark,cm_highCon,centimeter
from utils.preprocessing import print_tdms_structure,split_pulses_by_interval,getShiftMean
mpl.rcParams.update(two_column)
from MyDataset import sampling_rate_ae,sampling_rate_vib, project_dir, project_name, data_dir, dataDir_ae, dataDir_vib, alphabet

allowed_input_types=['ae_spec','vib_spec','ae_spec+ae_features','vib_spec+vib_features','ae_spec+ae_features+vib_spec+vib_features','all']

model_name = []
folds = 10
repeat = 1
max_epochs = 9

def get_hist_data(hist_dir, model_name, folds, max_epochs=100):
    train_acc = []
    test_acc = []
    for i in range(folds):
        file_path = f"{model_name}_fold{i}_of_folds{folds}.csv"
        file_path = os.path.join(hist_dir, file_path)
        # print(f"{file_path}")
        # print(f"{file_path},{os.listdir(file_path)}")
        df = pd.read_csv(file_path, index_col=0)
        df = df[~df.index.get_level_values(0).duplicated(keep="last")]
        train_acc.append((1-df["train_loss"][max_epochs]))
        test_acc.append((1-df["test_loss"][max_epochs]))
    return train_acc, test_acc

def generate_hist_df(
    hist_dir, model_name, folds, max_epochs
):
    train_acc_list = []
    test_acc_list = []
    fold_i_list = []
    model_list = []

    for _model_name in model_name:
        train_acc, test_acc = get_hist_data(
            hist_dir, _model_name, folds, max_epochs
        )
        for i in range(folds):
            fold_i_list.append(i)
            train_acc_list.append(train_acc[i])
            test_acc_list.append(test_acc[i])
            model_list.append(_model_name)

    df = pd.DataFrame()
    df["Input Type"] = model_list
    df["Train Acc"] = train_acc_list
    df["Test Acc"] = test_acc_list
    df["Fold Index"] = fold_i_list

    new_df = pd.concat([df, df])
    new_df["Acc"] = pd.concat([df["Train Acc"], df["Test Acc"]])
    new_df["Acc type"] = ["Train"] * len(df) + ["Test"] * len(df)
    new_df.index = range(len(new_df))
    return new_df


his_dir = os.path.join(os.pardir,"lfs","train_his")
df = generate_hist_df(his_dir,allowed_input_types,folds=folds,max_epochs=max_epochs)

plt.rcParams.update(one_column)
plt.rcParams['figure.constrained_layout.use'] = True
fig,ax = plt.subplots(figsize=(6.0,2))
snsdf = sns.load_dataset("tips")
ax = sns.violinplot(
    # data=df[df["Model"]=="CNN"],
    data=df,
    x="Input Type",
    y="Acc",
    hue="Acc type",
    split=True,
    saturation=0.75,
    density_norm="width",
    hue_order=["Train", "Test"],
    dodge=True,
    inner='quartile',
    ax=ax,
    linewidth=0.3,
    width = 0.5,
    palette = ['#0C5DA5', '#00B945', 'yellow'],
)
# ax.set_yticklabels([f'{x*100:.0f}\%' for x in ax.get_yticks()[:-2]]) 
xTickLabels = allowed_input_types
ax.set_xticklabels([('\n').join(x.split('_')) for x in xTickLabels]) 
ax.set_ylabel("Accuracy")
plt.savefig(os.path.join("raw_acc_vs_model.png"),dpi=300)
plt.close()