"""
Created on Thu Jan 22 12:42:42 2026

@author: Ali Pilehvar Meibody
Metrics Summary

Spyder Plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['figure.dpi'] = 300

# Read data from Excel
results_mape = pd.read_excel('METRICS_DATA.xlsx', sheet_name='MAPE')
results_r2 = pd.read_excel('METRICS_DATA.xlsx', sheet_name='R2')
results_mae = pd.read_excel('METRICS_DATA.xlsx', sheet_name='MAE')
results_rmse = pd.read_excel('METRICS_DATA.xlsx', sheet_name='RMSE')

# Clean and prepare data
def prepare_data(df):
    """Prepare data by setting proper column names and removing header row"""
    df_clean = df.copy()
    # Set column names from first row
    df_clean.columns = df_clean.iloc[0]
    # Remove first row (header row)
    df_clean = df_clean.iloc[1:].reset_index(drop=True)
    # Rename columns
    df_clean.columns = ['Model', 'L*', 'a*', 'b*', 'Average']
    # Convert numeric columns
    for col in ['L*', 'a*', 'b*', 'Average']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean

# Prepare all datasets
mape_data = prepare_data(results_mape)
r2_data = prepare_data(results_r2)
mae_data = prepare_data(results_mae)
rmse_data = prepare_data(results_rmse)

# Print data for verification
print("MAPE Data:")
print(mape_data)
print("\nR2 Data:")
print(r2_data)
print("\nMAE Data:")
print(mae_data)
print("\nRMSE Data:")
print(rmse_data)

# ============================================================
# Create Publication-Quality Comparison Graphs
# ============================================================

def create_comparison_graph(df, metric_name, ylabel, filename, colors=None,legend=True):
    """
    Create a grouped bar chart comparing models across L*, a*, b*
    
    Parameters:
    -----------
    df : DataFrame
        Data with columns: Model, L*, a*, b*, Average
    metric_name : str
        Name of the metric (for title)
    ylabel : str
        Y-axis label
    filename : str
        Output filename
    colors : list
        Colors for L*, a*, b* bars
    """
    if colors is None:
        colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    models = df['Model'].values
    x = np.arange(len(models))
    width = 0.25  # Width of bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars for L*, a*, b*
    bars1 = ax.bar(x - width, df['L*'].values, width, label='L*', 
                   color=colors[0], edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, df['a*'].values, width, label='a*', 
                   color=colors[1], edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + width, df['b*'].values, width, label='b*', 
                   color=colors[2], edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison Across Models', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    if legend==True:
        ax.legend(loc='upper left', fontsize=12, frameon=True, 
                 fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename, dpi=600, bbox_inches='tight', format='png')
    print(f"Saved: {filename}")
    plt.close()

# Create all comparison graphs
print("\nGenerating comparison graphs...")

# Graph 1: MAPE Comparison
create_comparison_graph(mape_data, 'MAPE', 'Mean Absolute Percentage Error (MAPE)', 
                       'MAPE_Comparison.png')

# Graph 2: R² Comparison
create_comparison_graph(r2_data, 'R²', 'Coefficient of Determination (R²)', 
                       'R2_Comparison.png',legend=False)

# Graph 3: MAE Comparison
create_comparison_graph(mae_data, 'MAE', 'Mean Absolute Error (MAE)', 
                       'MAE_Comparison.png',legend=False)

# Graph 4: RMSE Comparison
create_comparison_graph(rmse_data, 'RMSE', 'Root Mean Square Error (RMSE)', 
                       'RMSE_Comparison.png',legend=False)

print("\nAll graphs generated successfully!")

# ============================================================
# Alternative: Combined Multi-Panel Figure (Optional)
# ============================================================

def create_combined_figure():
    """Create a 2x2 subplot figure with all metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison Across L*, a*, and b*', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    datasets = [
        (mape_data, 'MAPE', 'Mean Absolute Percentage Error (MAPE)', axes[0, 0]),
        (r2_data, 'R²', 'Coefficient of Determination (R²)', axes[0, 1]),
        (mae_data, 'MAE', 'Mean Absolute Error (MAE)', axes[1, 0]),
        (rmse_data, 'RMSE', 'Root Mean Square Error (RMSE)', axes[1, 1])
    ]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for df, metric_name, ylabel, ax in datasets:
        models = df['Model'].values
        x = np.arange(len(models))
        width = 0.25
        
        bars1 = ax.bar(x - width, df['L*'].values, width, label='L*', 
                      color=colors[0], edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x, df['a*'].values, width, label='a*', 
                      color=colors[1], edgecolor='black', linewidth=1.2)
        bars3 = ax.bar(x + width, df['b*'].values, width, label='b*', 
                      color=colors[2], edgecolor='black', linewidth=1.2)
        
        # Add value labels (smaller for combined figure)
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=10, frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('Combined_Metrics_Comparison.png', dpi=300, bbox_inches='tight', format='png')
    print("Saved: Combined_Metrics_Comparison.png")
    plt.close()

# Create combined figure
create_combined_figure()

# ============================================================
# Create Spider/Radar Plots
# ============================================================

from math import pi

def create_radar_plot(df, metric_name, filename, normalize=True, invert_for_lower_better=True,legend=False):
    """
    Create a radar/spider plot comparing models across L*, a*, b*
    
    Parameters:
    -----------
    df : DataFrame
        Data with columns: Model, L*, a*, b*, Average
    metric_name : str
        Name of the metric (for title)
    filename : str
        Output filename
    normalize : bool
        Whether to normalize values to 0-1 scale
    invert_for_lower_better : bool
        For metrics where lower is better (MAPE, MAE, RMSE), invert the scale
    """
    # Number of variables (L*, a*, b*)
    categories = ['L*', 'a*', 'b*']
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Prepare data
    models = df['Model'].values
    values = df[['L*', 'a*', 'b*']].values
    
    # Normalize if requested
    if normalize:
        if invert_for_lower_better:
            # For lower-is-better metrics, invert so higher normalized value = better
            max_val = values.max()
            min_val = values.min()
            if max_val != min_val:
                values = 1 - (values - min_val) / (max_val - min_val)
            else:
                values = np.ones_like(values)
        else:
            # For higher-is-better metrics (like R²)
            max_val = values.max()
            min_val = values.min()
            if max_val != min_val:
                values = (values - min_val) / (max_val - min_val)
            else:
                values = np.ones_like(values)
    
    # Define colors for each model (distinct colors for all 6 models)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8B4CBF']
    
    # Plot each model
    for idx, (model, model_values) in enumerate(zip(models, values)):
        values_plot = list(model_values)
        values_plot += values_plot[:1]  # Complete the circle
        
        ax.plot(angles, values_plot, 'o-', linewidth=2.5, label=model, 
               color=colors[idx % len(colors)], markersize=8)
        ax.fill(angles, values_plot, alpha=0.15, color=colors[idx % len(colors)])
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=18, fontweight='bold')
    
    # Set y-axis limits
    if normalize:
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    else:
        # Use actual data range
        all_values = df[['L*', 'a*', 'b*']].values.flatten()
        y_min = all_values.min() * 0.9
        y_max = all_values.max() * 1.1
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Add title
    title_text = f'{metric_name}'
    '''
    if normalize and invert_for_lower_better:
        title_text += ' (Normalized - Higher is Better)'
    elif normalize:
        title_text += ' (Normalized)'
    '''
    ax.set_title(title_text, size=16, fontweight='bold', pad=20)
    
    # Add legend
    if legend==True:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, 
                 frameon=True, fancybox=True, shadow=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight', format='png')
    print(f"Saved: {filename}")
    plt.close()

def create_radar_plot_individual_models(df, metric_name, base_filename):
    """
    Create individual radar plots for each model
    """
    categories = ['L*', 'a*', 'b*']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    models = df['Model'].values
    values = df[['L*', 'a*', 'b*']].values
    
    # Create a figure with subplots for each model
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows), 
                            subplot_kw=dict(projection='polar'))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'{metric_name} Radar Plots - Individual Models', 
                fontsize=16, fontweight='bold', y=0.995)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8B4CBF']
    
    for idx, (model, model_values) in enumerate(zip(models, values)):
        ax = axes[idx]
        values_plot = list(model_values)
        values_plot += values_plot[:1]
        
        ax.plot(angles, values_plot, 'o-', linewidth=3, 
               color=colors[idx % len(colors)], markersize=10)
        ax.fill(angles, values_plot, alpha=0.25, color=colors[idx % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        
        # Set y-axis to show actual values
        all_values = df[['L*', 'a*', 'b*']].values.flatten()
        y_min = all_values.min() * 0.9
        y_max = all_values.max() * 1.1
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
        
        ax.set_title(model, size=13, fontweight='bold', pad=15)
        
        # Add value labels
        for angle, value, cat in zip(angles[:-1], model_values, categories):
            ax.text(angle, value, f'{value:.3f}', 
                   fontsize=9, fontweight='bold', ha='center', va='bottom')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(base_filename, dpi=600, bbox_inches='tight', format='png')
    print(f"Saved: {base_filename}")
    plt.close()

# Create radar plots for each metric
print("\nGenerating radar/spider plots...")

# Radar plot 1: MAPE (lower is better, so invert normalization)
create_radar_plot(mape_data, 'MAPE', 'MAPE_Radar_Plot.png', 
                 normalize=True, invert_for_lower_better=True)

# Radar plot 2: R² (higher is better)
create_radar_plot(r2_data, 'R²', 'R2_Radar_Plot.png', 
                 normalize=True, invert_for_lower_better=False)

# Radar plot 3: MAE (lower is better, so invert normalization)
create_radar_plot(mae_data, 'MAE', 'MAE_Radar_Plot.png', 
                 normalize=True, invert_for_lower_better=True ,)

# Radar plot 4: RMSE (lower is better, so invert normalization)
create_radar_plot(rmse_data, 'RMSE', 'RMSE_Radar_Plot.png', 
                 normalize=True, invert_for_lower_better=True)

# Create individual model radar plots (non-normalized, showing actual values)
print("\nGenerating individual model radar plots...")
create_radar_plot_individual_models(mape_data, 'MAPE', 'MAPE_Radar_Individual.png')
create_radar_plot_individual_models(r2_data, 'R²', 'R2_Radar_Individual.png')
create_radar_plot_individual_models(mae_data, 'MAE', 'MAE_Radar_Individual.png')
create_radar_plot_individual_models(rmse_data, 'RMSE', 'RMSE_Radar_Individual.png')

print("\nAll radar plots generated successfully!")

























