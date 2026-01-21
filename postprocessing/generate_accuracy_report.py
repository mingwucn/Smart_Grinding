import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import matplotlib as mpl

# Add project root to path to import MyDataset
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Set plotting style
try:
    import scienceplots
    plt.style.use(['science', 'nature'])
except ImportError:
    print("SciencePlots not found, using default style.")
    sns.set_style("whitegrid")

# Default settings matching the notebook
DEFAULT_ALLOWED_INPUT_TYPES = [
    'ae_spec',
    'ae_features',
    'ae_features+pp',
    'ae_spec+ae_features',
    'vib_spec',
    'vib_features',
    'vib_features+pp',
    'vib_spec+vib_features',
    'ae_features+vib_features',
    'ae_features+vib_features+pp',
    'ae_spec+vib_spec',
    'ae_spec+ae_features+vib_spec+vib_features',
    'all',
]

def get_hist_data(hist_dir, input_type, repeat, folds, max_epochs=100):
    train_mae = []
    test_mae = []
    train_mse = []
    test_mse = []
    
    expected_files = int(folds * repeat)
    
    for i in range(expected_files):
        file_name = f"{input_type}_fold{i}_of_folds{folds}.csv"
        file_path = os.path.join(hist_dir, file_name)
        
        if not os.path.exists(file_path):
            # Try finding without repeat count in name if generic pattern fails
            # But based on file listing, the pattern seems consistent
            # We'll skip if missing but warn
            # print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path, index_col=0)
            df = df[~df.index.get_level_values(0).duplicated(keep="last")]
            
            # Check if max_epochs exists in index
            if max_epochs in df.index:
                idx = max_epochs
            else:
                idx = df.index[-1] # Use last available if specific epoch not found
                
            train_mse.append(df.loc[idx, "train_mse"])
            test_mse.append(df.loc[idx, "test_mse"])
            train_mae.append(df.loc[idx, "train_mae"])
            test_mae.append(df.loc[idx, "test_mae"])
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
    return train_mse, test_mse, train_mae, test_mae

def generate_hist_df(hist_dir, allowed_input_types, repeat, folds, max_epochs, scale_factor=0.1):
    train_mse_list = []
    test_mse_list = []
    train_mae_list = []
    test_mae_list = []
    fold_i_list = []
    model_list = []

    for _input_type in allowed_input_types:
        train_mse, test_mse, train_mae, test_mae = get_hist_data(
            hist_dir, _input_type, repeat, folds, max_epochs
        )
        
        # We might have fewer files than folds*repeat if some failed or are missing
        num_samples = len(train_mae)
        
        for i in range(num_samples):
            fold_i_list.append(i % folds)
            train_mae_list.append(train_mae[i] * scale_factor)
            test_mae_list.append(test_mae[i] * scale_factor)
            train_mse_list.append(train_mse[i] * scale_factor)
            test_mse_list.append(test_mse[i] * scale_factor)
            model_list.append(_input_type)

    df = pd.DataFrame()
    df["Input Type"] = model_list
    df["Train MAE"] = train_mae_list
    df["Test MAE"] = test_mae_list
    df["Train MSE"] = train_mse_list
    df["Test MSE"] = test_mse_list
    df["Fold index"] = fold_i_list

    # Prepare DataFrame for plotting (MAE)
    mae_df = pd.concat([df, df])
    mae_df["MAE"] = pd.concat([df["Train MAE"], df["Test MAE"]])
    mae_df["MAE type"] = ["Train"] * len(df) + ["Test"] * len(df)
    mae_df.index = range(len(mae_df))
    
    # Prepare DataFrame for plotting (MSE)
    mse_df = pd.concat([df, df])
    mse_df["MSE"] = pd.concat([df["Train MSE"], df["Test MSE"]])
    mse_df["MSE type"] = ["Train"] * len(df) + ["Test"] * len(df)
    mse_df.index = range(len(mse_df))
    
    return mae_df, mse_df

def plot_metric(df, metric_name, input_types, output_path):
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(
        data=df,
        x="Input Type",
        y=metric_name,
        hue=f"{metric_name} type",
        split=True,
        saturation=0.55,
        density_norm="width",
        hue_order=["Train", "Test"],
        dodge=True,
        inner='quartile',
        linewidth=0.3,
        width=0.8,
        palette=['#0C5DA5', '#00B945'],
        cut=0,
    )
    
    # Format x-labels to wrap text
    ax.set_xticklabels([("\n").join(x.split('_')) for x in input_types])
    ax.set_ylabel(f"Accuracy ({metric_name})")
    ax.set_title(f"Model {metric_name} Comparison")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()

def generate_text_report(mae_df, mse_df, output_path):
    report_content = "# Smart Grinding Model Accuracy Report\n\n"
    
    # Calculate summary statistics for Test MAE
    test_mae_stats = mae_df[mae_df["MAE type"] == "Test"].groupby("Input Type")["MAE"].agg(['mean', 'std', 'min', 'max']).sort_values('mean')
    
    report_content += "## MAE Performance (Test Set)\n\n"
    report_content += "| Model Input Type | Mean MAE | Std Dev | Min | Max |\n"
    report_content += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    for index, row in test_mae_stats.iterrows():
        report_content += f"| {index} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |\n"
        
    # Best Model
    best_model = test_mae_stats.index[0]
    report_content += f"\n**Best Performing Model (Lowest MAE):** `{best_model}` with MAE = {test_mae_stats.iloc[0]['mean']:.4f}\n\n"
    
    # Calculate summary statistics for Test MSE
    test_mse_stats = mse_df[mse_df["MSE type"] == "Test"].groupby("Input Type")["MSE"].agg(['mean', 'std']).sort_values('mean')
    
    report_content += "## MSE Performance (Test Set)\n\n"
    report_content += "| Model Input Type | Mean MSE | Std Dev |\n"
    report_content += "| :--- | :--- | :--- |\n"
    
    for index, row in test_mse_stats.iterrows():
        report_content += f"| {index} | {row['mean']:.4f} | {row['std']:.4f} |\n"

    # Write to file
    with open(output_path, "w") as f:
        f.write(report_content)
    
    print(f"Saved accuracy report to {output_path}")
    print("\n--- Summary ---")
    print(f"Best Model: {best_model} (MAE: {test_mae_stats.iloc[0]['mean']:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Generate accuracy plots and report for Smart Grinding models.")
    parser.add_argument("--hist_dir", type=str, default=os.path.join(project_root, "lfs", "train_his"), help="Directory containing training history CSVs")
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "Grinding Fusion", "images"), help="Directory to save plots")
    parser.add_argument("--report_dir", type=str, default=os.path.join(project_root, "report", "accuracy_reports"), help="Directory to save text report")
    parser.add_argument("--epochs", type=int, default=19, help="Epoch number to extract metrics from")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats")
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    
    print(f"Loading data from {args.hist_dir}...")
    mae_df, mse_df = generate_hist_df(
        args.hist_dir, 
        DEFAULT_ALLOWED_INPUT_TYPES, 
        args.repeats, 
        args.folds, 
        args.epochs
    )
    
    if mae_df.empty:
        print("No data found! Check your history directory and file naming.")
        return

    print("Generating plots...")
    plot_metric(mae_df, "MAE", DEFAULT_ALLOWED_INPUT_TYPES, os.path.join(args.output_dir, "raw_MAE_vs_model.png"))
    plot_metric(mse_df, "MSE", DEFAULT_ALLOWED_INPUT_TYPES, os.path.join(args.output_dir, "raw_MSE_vs_model.png"))
    
    print("Generating summary report...")
    generate_text_report(mae_df, mse_df, os.path.join(args.report_dir, "model_accuracy_summary.md"))

if __name__ == "__main__":
    main()
