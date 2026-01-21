import sys
import numpy as np
import scipy
import subprocess
from natsort import natsorted
import itertools
import pandas as pd
import pickle

sys.path.append("../utils")
sys.path.append("../utils")
sys.path.append("../.")
import os, re
from .preprocessing import *
from .InterfaceDeclaration import LPBFInterface
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    StandardScaler,
    LabelEncoder,
)
from .MLUtils import LPBFDataset
from .MLModels import SVMModel, CNN_Base_1D_Model, ResNet15_1D_Model

def generate_hist_name(model_name, acoustic_type, context_type, output_type):
    context_combinations = []
    for r in range(len(context_type) + 1):
        context_combinations.extend(itertools.combinations(context_type, r))

    # Generate all combinations of acoustic_type with context_combinations
    all_combinations = []
    for _output in output_type:
        for model in model_name:
            for acoustic in acoustic_type:
                for context in context_combinations:
                    if len(context) > 0:
                        inputs = {acoustic} + {"+".join(list(context))}
                        all_combinations.append(
                            f"{model}_classification_input_{acoustic}+{'+'.join(list(context))}_output_{_output}"
                        )
                    else:
                        inputs = {acoustic}
                        all_combinations.append(
                            f"{model}_classification_input_{acoustic}_output_{_output}"
                        )
                    #     all_combinations.append(f"{model}_classification_input_{acoustic}")
    return all_combinations


def find_non_float_index(lst):
    for i, item in enumerate(lst):
        try:
            float(item)
        except TypeError:
            return i
    return -1


def get_hist_data(hist_dir, model_name, inputs, output_type, folds, max_epochs):
    train_acc = []
    test_acc = []
    for i in range(folds):
        file_path = f"{model_name}_classification_input_{inputs}_output_{output_type}_roi_time10_roi_radius3_fold{i}_of_folds10.csv"
        file_path = os.path.join(hist_dir, file_path)
        df = pd.read_csv(file_path, index_col=0)
        df = df[~df.index.get_level_values(0).duplicated(keep="last")]
        train_acc.append((df["Train Accuracy"][max_epochs]))
        test_acc.append((df["Test Accuracy"][max_epochs]))
    return train_acc, test_acc


def get_hist_data_path(hist_dir, model_name, inputs, output_type, fold=0):
    file_path = f"{model_name}_classification_input_{inputs}_output_{output_type}_roi_time10_roi_radius3_fold{fold}_of_folds10.csv"
    file_path = os.path.join(hist_dir, file_path)
    return file_path


def generate_hist_df(
    hist_dir, model_name, acoustic_type, context_type, output_type, folds, max_epochs
):
    context_combinations = []
    for r in range(len(context_type) + 1):
        context_combinations.extend(itertools.combinations(context_type, r))
    train_acc_list = []
    test_acc_list = []
    fold_i_list = []
    inputs_list = []
    outputs_list = []
    model_list = []

    for _output in output_type:
        for model in model_name:
            for acoustic in acoustic_type:
                for context in context_combinations:
                    if len(context) > 0:
                        inputs = f"{acoustic}+{'+'.join(list(context))}"
                    else:
                        inputs = acoustic

                    train_acc, test_acc = get_hist_data(
                        hist_dir, model, inputs, _output, folds, max_epochs
                    )
                    for i in range(folds):
                        fold_i_list.append(i)
                        train_acc_list.append(train_acc[i])
                        test_acc_list.append(test_acc[i])
                        inputs_list.append(inputs)
                        outputs_list.append(_output)
                        model_list.append(model)

    df = pd.DataFrame()
    df["Model"] = model_list
    df["Train Acc"] = train_acc_list
    df["Test Acc"] = test_acc_list
    df["Fold index"] = fold_i_list
    df["Input type"] = inputs_list
    df["Output type"] = outputs_list

    new_df = pd.concat([df, df])
    new_df["Acc"] = pd.concat([df["Train Acc"], df["Test Acc"]])
    new_df["Acc type"] = ["Train"] * len(df) + ["Test"] * len(df)
    new_df.index = range(len(new_df))
    return new_df


def get_dataset(roi_time=10, roi_radius=3):
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
    del music_dir

    with open(
        os.path.join(
            os.path.dirname(daq_dir), "intermediate", f"lpbf_line_wise_data.pkl"
        ),
        "rb",
    ) as handle:
        lpbf_data = pickle.load(handle)

    sc_power = StandardScaler().fit(
        np.unique(lpbf_data.laser_power).astype(float).reshape(-1, 1)
    )
    # sc_direction = StandardScaler().fit(np.unique(lpbf_data.print_vector[1]).astype(float).reshape(-1,1))
    le_direction = LabelEncoder().fit(
        np.unique(np.asarray(np.round(lpbf_data.print_vector[1]), dtype=str))
    )
    le_speed = LabelEncoder().fit(np.asarray(lpbf_data.scanning_speed, dtype=str))
    le_region = LabelEncoder().fit(np.asarray(lpbf_data.regime_info, dtype=str))

    laser_power = sc_power.transform(
        np.asarray(lpbf_data.laser_power).astype(float).reshape(-1, 1)
    ).reshape(-1)
    # print_direction = sc_direction.transform(np.asarray(lpbf_data.print_vector[1]).astype(float).reshape(-1,1)).reshape(-1)
    print_direction = le_direction.transform(
        np.asarray(np.round(lpbf_data.print_vector[1]), dtype=str)
    ).astype(int)
    scanning_speed = le_speed.transform(
        np.asarray(lpbf_data.scanning_speed).astype(float)
    )
    regime_info = le_region.transform(np.asarray(lpbf_data.regime_info, dtype=str))

    dataset = LPBFDataset(
        lpbf_data.cube_position,
        laser_power,
        lpbf_data.scanning_speed,
        regime_info,
        print_direction,
        lpbf_data.microphone,
        lpbf_data.AE,
        lpbf_data.defect_labels,
    )
    return dataset, sc_power, le_direction, le_speed, le_region


def make_model(model_name, input_type, output_type, time_series_length=5888):
    meta_data_size = len(input_type.split("+")) - 1
    # time_series_length = 5888
    # print(f"Input type: {input_type}")
    # print(f"Output type: {output_type}")
    # print(f"Lenght of time series data: {time_series_length}")
    # print(f"Lenght of context data: {meta_data_size}")
    # print(f"Model name: {model_name}")

    num_classes = 1
    if output_type == "regime":
        num_classes = 4
    if output_type == "defect":
        num_classes = 2
    if output_type == "direction":
        num_classes = 5
    if output_type == "position":
        num_classes = 5

    if model_name == "SVM":
        model = SVMModel(
            time_series_length + meta_data_size, num_classes=num_classes
        ).double()
    if model_name == "CNN":
        model = CNN_Base_1D_Model(
            time_series_length=time_series_length,
            meta_data_size=meta_data_size,
            num_classes=num_classes,
        ).double()
    if model_name == "Res15":
        model = ResNet15_1D_Model(
            time_series_length=time_series_length,
            meta_data_size=meta_data_size,
            num_classes=num_classes,
        ).double()
    return model


def read_trained_model(
    snap_dir, model_name, acoustic_type, context_type, output_type, folds, max_epochs
):
    snap_list = []
    context_combinations = []
    for r in range(len(context_type) + 1):
        context_combinations.extend(itertools.combinations(context_type, r))
    train_acc_list = []
    test_acc_list = []
    fold_i_list = []
    inputs_list = []
    outputs_list = []
    model_list = []

    for _output in output_type:
        for _model in model_name:
            for acoustic in acoustic_type:
                for context in context_combinations:
                    if len(context) > 0:
                        inputs = f"{acoustic}+{'+'.join(list(context))}"
                    else:
                        inputs = acoustic

                    for fold in range(folds):
                        file_path = f"{_model}_classification_input_{inputs}_output_{_output}_roi_time10_roi_radius3_fold{fold}_of_folds10.pt"
                        file_path = os.path.join(snap_dir, file_path)
                        model = make_model(_model, inputs, _output)
                        snap_list.append(file_path)
    return model, snap_list


def get_confusion_matrix(
    model, snap_dir, _model_name, _inputs, _outputs, class_num=5, folds_num=10
):
    import torch
    from sklearn.metrics import confusion_matrix

    _model_name = "CNN"
    _inputs = "ae"
    _outputs = "direction"
    meta_list = []
    pred_list = []
    label_list = []
    cf = np.zeros((folds_num, class_num, class_num), dtype=np.int64)
    for f_i in range(folds_num):
        snap_name = f"{_model_name}_classification_input_{_inputs}_output_{_outputs}_roi_time10_roi_radius3_fold{f_i}_of_folds10.pt"
        snap_name = os.path.join(snap_dir, snap_name)
        snapshot = torch.load(snap_name, map_location=f"cuda:0", weights_only=True)
        _state_dict = snapshot["model_state_dict"]
        model.load_state_dict(_state_dict)
        model = model.to("cuda")
        # _cube_position, _laser_power, _scanning_speed, _regime_info, _print_direction, _mic, _ae, _defect_labels = next(iter(data_loader))
        with torch.no_grad():
            for (
                _cube_position,
                _laser_power,
                _scanning_speed,
                _regime_info,
                _print_direction,
                _mic,
                _ae,
                _defect_labels,
            ) in data_loader:
                time_series = (transform_ft()(standardize_tensor(_ae))).double()
                meta_list.append(_laser_power.double())
                logits = model(time_series.to("cuda"), meta_list)
                probs = torch.sigmoid(logits)
                preds = torch.argmax(probs, axis=1).clone().int().detach().cpu()
                pred_list.extend(preds.cpu().numpy())
                label_list.extend(_print_direction.cpu().numpy())
        cf[f_i, :, :] = confusion_matrix(label_list, pred_list)
    return cf


def compute_and_plot_confusion_matrix(
    model, dataloader, class_names, device="cuda", visualization=False
):
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import torch

    """
    Computes and visualizes the confusion matrix for a given model and dataloader.

    Parameters:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): The dataloader for evaluation data.
        class_names (list): List of class names.
        device (str): Device to use for computation (default: "cuda").

    Returns:
        np.ndarray: The confusion matrix.
    """
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Store predictions and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Flatten the lists
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Normalize confusion matrix (optional)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if visualization:
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()
    # Print classification report
    print(
        "Classification Report:\n",
        classification_report(all_labels, all_preds, target_names=class_names),
    )
    return cm


def find_matching_models(model_dir, model_name, spec, fold_num):
    """
    Finds models in a directory that match a specific naming pattern.

    Args:
        model_dir: The directory containing the models.
        model_name: The base name of the model.
        fold_num: The fold number.

    Returns:
        A list of strings, where each string is the name of a matching model file.
        Returns an empty list if no matching models are found.
    """

    matching_models = []
    pattern = (
        f"^{model_name}.*{spec}.*_of_folds{fold_num}\.pt$"  # Regular expression pattern
    )

    for filename in os.listdir(model_dir):
        if re.match(pattern, filename):
            matching_models.append(filename)

    return matching_models


def make_confusion_matrix(
    cf,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap="Blues",
    title=None,
):
    import seaborn as sns

    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.3}\%".format(value * 100) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )
    plt.tick_params(
        axis="both",  # changes apply to both axis
        which="both",  # both major and minor ticks are affected
        top=False,
        bottom=False,
        left=False,
        right=False,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def merge_confusion_matrix(merge_mapping, cm):
    # Number of unique classes after merging
    num_new_classes = len(set(merge_mapping.values()))

    # Initialize the new confusion matrix
    new_cm = np.zeros((num_new_classes, num_new_classes), dtype=int)

    # Populate the new confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            new_cm[merge_mapping[i], merge_mapping[j]] += cm[i, j]
    return new_cm


def merge_confusion_matrix_categories(confusion_matrix, merge_map):
    """
    Merges categories in a confusion matrix based on a provided mapping.

    Args:
    confusion_matrix: A NumPy array representing the confusion matrix.
    merge_map: A dictionary where keys are the original category names and values are the new merged category names.

    Returns:
        A new NumPy array representing the merged confusion matrix.

    Example usage:
        confusion_matrix = np.array([[10, 2, 3],
                                    [1, 8, 1],
                                    [2, 1, 5]])

        merge_map = {0: 'A', 1: 'B', 2: 'A'}

        merged_matrix = merge_confusion_matrix_categories(confusion_matrix, merge_map)
        print(merged_matrix)
    """

    num_classes = len(confusion_matrix)
    new_classes = set(merge_map.values())
    new_num_classes = len(new_classes)
    merged_matrix = np.zeros((new_num_classes, new_num_classes), dtype=int)

    # Create a mapping from original to new class indices
    class_mapping = {old_class: i for i, old_class in enumerate(new_classes)}

    # Merge the confusion matrix
    for i in range(num_classes):
        new_row = class_mapping[merge_map[i]]
        for j in range(num_classes):
            new_col = class_mapping[merge_map[j]]
            merged_matrix[new_row][new_col] += confusion_matrix[i][j]

    return merged_matrix


import numpy as np


def calculate_metrics(confusion_matrix):
    """
    Calculates accuracy, precision, recall, and F1-score from a given confusion matrix.

    Args:
        confusion_matrix: A NumPy array representing the confusion matrix.

    Returns:
        A dictionary containing the calculated metrics:
            - accuracy
            - precision (per class)
            - recall (per class)
            - f1_score (per class)
            - macro_precision
            - macro_recall
            - macro_f1_score
            - weighted_precision
            - weighted_recall
            - weighted_f1_score
    Example usage:
        confusion_matrix = np.array([[50, 5, 0],
                                    [5, 45, 10],
                                    [0, 10, 40]])

        metrics = calculate_metrics(confusion_matrix)
        print(metrics)
    """

    num_classes = confusion_matrix.shape[0]
    metrics = {}

    # Calculate true positives, true negatives, false positives, and false negatives
    tp = np.diag(confusion_matrix)
    fp = np.sum(confusion_matrix, axis=0) - tp
    fn = np.sum(confusion_matrix, axis=1) - tp
    tn = np.sum(confusion_matrix) - (tp + fp + fn)

    # Calculate accuracy
    accuracy = np.sum(tp) / np.sum(confusion_matrix)
    metrics["accuracy"] = accuracy

    # Calculate precision, recall, and F1-score per class
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    # Handle potential division by zero
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    f1_score[np.isnan(f1_score)] = 0

    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_score"] = f1_score

    # Calculate macro-averaged metrics
    metrics["macro_precision"] = np.mean(precision)
    metrics["macro_recall"] = np.mean(recall)
    metrics["macro_f1_score"] = np.mean(f1_score)

    # Calculate weighted-averaged metrics
    weights = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    metrics["weighted_precision"] = np.sum(precision * weights)
    metrics["weighted_recall"] = np.sum(recall * weights)
    metrics["weighted_f1_score"] = np.sum(f1_score * weights)
    return metrics


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


def interp_array(arr,n=10):
    new_x = np.linspace(1, len(arr), len(arr)*n, endpoint=True)
    linear_interp = scipy.interpolate.interp1d(range(1,len(arr)+1), arr, kind='linear')
    # return np.interp(new_x, range(13), arr)
    return new_x, linear_interp(new_x)


def normalize_array(arr):
    """
    Normalize a NumPy array between 0 and 1.
    Parameters:
        arr (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: Normalized array.
    """
    # Calculate the minimum and maximum values
    min_val = np.min(arr)
    max_val = np.max(arr)
    # Normalize the array
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr


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


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics: MAE, MSE, RMSE, R²
    
    Parameters:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        dict: Dictionary of regression metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
