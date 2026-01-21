#!/usr/bin/env python3
"""
Template for data analysis scripts in research projects.
This script follows the research publication template conventions.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src directory to path for project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

def load_environment():
    """Load environment variables for data paths."""
    # In production, use python-dotenv or similar
    data_dir = os.getenv('DATA_DIR', './data')
    output_dir = os.getenv('OUTPUT_DIR', './output')
    return {
        'data_dir': Path(data_dir),
        'output_dir': Path(output_dir),
        'feature_name': os.getenv('FEATURE_NAME', 'analysis')
    }

def create_output_structure(env):
    """Create output directory structure for the analysis."""
    feature_dir = env['output_dir'] / env['feature_name']
    viz_dir = feature_dir / 'visualization'
    report_dir = feature_dir / 'report'
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'feature_dir': feature_dir,
        'viz_dir': viz_dir,
        'report_dir': report_dir
    }

def load_data(data_dir):
    """Load data from the data directory."""
    raw_data_path = data_dir / 'raw'
    processed_data_path = data_dir / 'processed'
    
    # Example: Load CSV files
    data_files = list(raw_data_path.glob('*.csv')) + list(processed_data_path.glob('*.csv'))
    
    if not data_files:
        print(f"No CSV files found in {raw_data_path} or {processed_data_path}")
        return None
    
    # Load first CSV file as example
    data_file = data_files[0]
    print(f"Loading data from: {data_file}")
    return pd.read_csv(data_file)

def analyze_data(data, output_dirs):
    """Perform analysis and generate outputs."""
    if data is None:
        print("No data to analyze")
        return
    
    # Example analysis: Basic statistics
    stats = data.describe()
    
    # Save statistics report
    stats_report_path = output_dirs['report_dir'] / 'statistics.md'
    with open(stats_report_path, 'w') as f:
        f.write("# Data Analysis Report\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(stats.to_markdown())
    
    # Example visualization: Distribution plot
    numeric_cols = data.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(15, 5))
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for ax, col in zip(axes, numeric_cols[:3]):
            sns.histplot(data[col], ax=ax, kde=True)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        viz_path = output_dirs['viz_dir'] / 'distributions.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {viz_path}")
    
    print(f"Report saved to: {stats_report_path}")
    return stats

def main():
    """Main analysis workflow."""
    print("Starting data analysis...")
    
    # Load environment configuration
    env = load_environment()
    print(f"Data directory: {env['data_dir']}")
    print(f"Output directory: {env['output_dir']}")
    print(f"Feature name: {env['feature_name']}")
    
    # Create output structure
    output_dirs = create_output_structure(env)
    print(f"Output structure created in: {output_dirs['feature_dir']}")
    
    # Load data
    data = load_data(env['data_dir'])
    
    # Analyze data
    results = analyze_data(data, output_dirs)
    
    print("Analysis complete!")
    return results

if __name__ == '__main__':
    main()