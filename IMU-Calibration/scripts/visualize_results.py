#!/usr/bin/env python3
"""
Visualize ALL methods matching reference plot style
Creates plots like the reference image with proper formatting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def compute_rmse(pred, truth, valid_mask):
    """Compute RMSE for valid samples"""
    valid_pred = pred[valid_mask]
    valid_truth = truth[valid_mask]
    
    if len(valid_pred) == 0:
        return np.nan
    
    return np.sqrt(np.mean((valid_pred - valid_truth)**2))

def plot_dataset(dataset_num, df, output_dir):
    """Create plot matching reference style"""
    
    valid_mask = df['Valid'].values == 1
    timestamps = df['Timestamp'].values
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Define colors and styles matching reference
    methods = {
        'Vicon': {'color': 'black', 'linewidth': 2.0, 'linestyle': '-', 'label': 'Vicon'},
        'Gyro': {'color': 'blue', 'linewidth': 1.2, 'linestyle': '--', 'label': 'Gyro'},
        'Accel': {'color': 'red', 'linewidth': 1.2, 'linestyle': ':', 'label': 'Acc'},
        'Complementary': {'color': 'green', 'linewidth': 1.5, 'linestyle': '-', 'label': 'Complementary'},
        'Madgwick': {'color': 'magenta', 'linewidth': 1.5, 'linestyle': '-', 'label': 'Madgwick'},
        'UKF': {'color': 'cyan', 'linewidth': 2.0, 'linestyle': '-', 'label': 'UKF'}
    }
    
    angles = ['Yaw', 'Roll', 'Pitch']
    ylabels = ['Yaw (deg)', 'Roll (deg)', 'Pitch (deg)']
    
    for idx, (angle, ylabel) in enumerate(zip(angles, ylabels)):
        ax = axes[idx]
        
        # Plot each method
        for method, style in methods.items():
            col_name = f'{method}_{angle}'
            if col_name in df.columns:
                ax.plot(timestamps, df[col_name].values,
                       color=style['color'],
                       linewidth=style['linewidth'],
                       linestyle=style['linestyle'],
                       label=f"{style['label']} {angle}",
                       alpha=0.8)
        
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Only bottom plot gets x-label
        if idx == 2:
            ax.set_xlabel('Time (s)', fontsize=11)
        
        # Title only on top plot
        if idx == 0:
            ax.set_title(f'Orientation Comparison: Dataset {dataset_num} (Train)', 
                        fontsize=13, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'all_methods_dataset{dataset_num}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Compute and return RMSE
    rmse = {}
    for method in ['Gyro', 'Accel', 'Complementary', 'Madgwick', 'UKF']:
        if valid_mask.sum() > 0:
            rmse[method] = {
                'yaw': compute_rmse(df[f'{method}_Yaw'].values, df['Vicon_Yaw'].values, valid_mask),
                'roll': compute_rmse(df[f'{method}_Roll'].values, df['Vicon_Roll'].values, valid_mask),
                'pitch': compute_rmse(df[f'{method}_Pitch'].values, df['Vicon_Pitch'].values, valid_mask)
            }
        else:
            rmse[method] = {'yaw': np.nan, 'roll': np.nan, 'pitch': np.nan}
    
    return rmse

def main():
    print("="*80)
    print("=== Orientation Tracking Visualization ===")
    print("="*80)
    print("\nMethods: Vicon (ground truth), Gyro, Accel, Complementary, Madgwick, UKF\n")
    
    results_dir = '../processed_data'
    plots_dir = os.path.join(results_dir, 'plots')
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run the C++ program first to generate CSV files.")
        return
    
    all_rmse = {}
    
    # Process each dataset
    for dataset_num in range(1, 7):
        csv_file = os.path.join(results_dir, f'all_methods_dataset{dataset_num}.csv')
        
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"=== Dataset {dataset_num} ===")
        print(f"{'='*60}")
        
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} samples")
        
        valid_count = (df['Valid'] == 1).sum()
        print(f"Valid samples: {valid_count}")
        
        # Create plot
        rmse = plot_dataset(dataset_num, df, plots_dir)
        all_rmse[dataset_num] = rmse
        
        # Print RMSE for this dataset
        print(f"\nRMSE vs Vicon Ground Truth:")
        print(f"{'Method':<15} {'Yaw':>10} {'Roll':>10} {'Pitch':>10}")
        print("-" * 50)
        for method in ['Gyro', 'Accel', 'Complementary', 'Madgwick', 'UKF']:
            r = rmse[method]
            print(f"{method:<15} {r['yaw']:>9.2f}° {r['roll']:>9.2f}° {r['pitch']:>9.2f}°")
    
    # Print summary table
    if all_rmse:
        print("\n" + "="*80)
        print("=== SUMMARY: RMSE for All Datasets (degrees) ===")
        print("="*80)
        
        for method in ['Gyro', 'Accel', 'Complementary', 'Madgwick', 'UKF']:
            print(f"\n{method}:")
            print("-" * 70)
            print(f"{'Dataset':<10} {'Yaw RMSE':>12} {'Roll RMSE':>12} {'Pitch RMSE':>12}")
            print("-" * 70)
            
            yaw_vals, roll_vals, pitch_vals = [], [], []
            
            for ds_num in sorted(all_rmse.keys()):
                r = all_rmse[ds_num][method]
                print(f"{ds_num:<10} {r['yaw']:>11.2f}° {r['roll']:>11.2f}° {r['pitch']:>11.2f}°")
                yaw_vals.append(r['yaw'])
                roll_vals.append(r['roll'])
                pitch_vals.append(r['pitch'])
            
            print("-" * 70)
            print(f"{'Average':<10} {np.nanmean(yaw_vals):>11.2f}° "
                  f"{np.nanmean(roll_vals):>11.2f}° {np.nanmean(pitch_vals):>11.2f}°")
        
        print("\n" + "="*80)
    
    print(f"\n✓ Visualization complete!")
    print(f"✓ Plots saved in: {plots_dir}/")
    print(f"\nTo view results:")
    print(f"  - Check plots: {plots_dir}/all_methods_dataset*.png")
    print(f"  - Check data: {results_dir}/all_methods_dataset*.csv")

if __name__ == "__main__":
    main()