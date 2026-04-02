#!/usr/bin/env python3
"""
IMU Data Processing and Visualization
Converts MATLAB .mat files to text format and plots results
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import os

def convert_imu_mat_to_txt(mat_file, txt_file):
    """
    Convert IMU .mat file to text format
    Output format: timestamp ax ay az wz wx wy
    """
    print(f"Converting {mat_file} to {txt_file}...")
    
    data = loadmat(mat_file)
    ts = data['ts'].flatten()
    vals = data['vals']
    
    # vals columns: [ax, ay, az, wz, wx, wy]
    with open(txt_file, 'w') as f:
        for i in range(len(ts)):
            f.write(f"{ts[i]:.6f} {vals[0,i]} {vals[1,i]} {vals[2,i]} "
                   f"{vals[3,i]} {vals[4,i]} {vals[5,i]}\n")
    
    print(f"Converted {len(ts)} samples")

def convert_vicon_mat_to_txt(mat_file, txt_file):
    """
    Convert Vicon .mat file to text format
    Output format: timestamp r11 r12 r13 r21 r22 r23 r31 r32 r33
    """
    print(f"Converting {mat_file} to {txt_file}...")
    
    data = loadmat(mat_file)
    ts = data['ts'].flatten()
    rots = data['rots']  # 3x3xN
    
    with open(txt_file, 'w') as f:
        for i in range(len(ts)):
            R = rots[:, :, i]
            f.write(f"{ts[i]:.6f} "
                   f"{R[0,0]} {R[0,1]} {R[0,2]} "
                   f"{R[1,0]} {R[1,1]} {R[1,2]} "
                   f"{R[2,0]} {R[2,1]} {R[2,2]}\n")
    
    print(f"Converted {len(ts)} samples")

def calibrate_imu_data(imu_file, params_file, output_file):
    """
    Apply calibration to IMU data according to project specifications
    """
    print(f"Calibrating IMU data...")
    
    # Load IMU parameters
    params = loadmat(params_file)
    imu_params = params['IMUParams']
    scale = imu_params[0, :]  # [sx, sy, sz]
    bias = imu_params[1, :]   # [bias_x, bias_y, bias_z]
    
    # Load IMU data
    imu_data = loadmat(imu_file)
    ts = imu_data['ts'].flatten()
    vals = imu_data['vals']
    
    # Calibrate accelerometer: a_calibrated = (a_raw + bias) * scale
    accel_cal = np.zeros((3, vals.shape[1]))
    for i in range(3):
        accel_cal[i, :] = (vals[i, :] + bias[i]) * scale[i]
    
    # Calibrate gyroscope: omega_calibrated = (3300/1023 * pi/180 * 0.3) * (omega_raw - bias)
    # Calculate bias as average of first 100 samples
    gyro_bias = np.mean(vals[3:6, :100], axis=1)
    print(f"Gyro bias: {gyro_bias}")
    
    gyro_scale = 3300.0 / 1023.0 * np.pi / 180.0 * 0.3
    gyro_cal = np.zeros((3, vals.shape[1]))
    for i in range(3):
        gyro_cal[i, :] = (vals[i+3, :] - gyro_bias[i]) * gyro_scale
    
    # Save calibrated data
    with open(output_file, 'w') as f:
        for i in range(len(ts)):
            f.write(f"{ts[i]:.6f} "
                   f"{accel_cal[0,i]} {accel_cal[1,i]} {accel_cal[2,i]} "
                   f"{gyro_cal[0,i]} {gyro_cal[1,i]} {gyro_cal[2,i]}\n")
    
    print(f"Calibrated data saved to {output_file}")

def plot_results(csv_file, output_image='results_plot.png'):
    """
    Plot attitude estimation results from CSV file
    """
    print(f"Plotting results from {csv_file}...")
    
    # Load data
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    
    timestamps = data[:, 0]
    timestamps = timestamps - timestamps[0]  # Start from 0
    
    gyro_euler = data[:, 1:4]    # [yaw, pitch, roll]
    accel_euler = data[:, 4:7]
    comp_euler = data[:, 7:10]
    ukf_euler = data[:, 10:13]
    vicon_euler = data[:, 13:16]
    
    # Create figure with 3 subplots (one for each angle)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    angle_names = ['Yaw', 'Pitch', 'Roll']
    
    for i, (ax, name) in enumerate(zip(axes, angle_names)):
        # Plot all methods
        ax.plot(timestamps, gyro_euler[:, i], 'b-', label='Gyro Integration', alpha=0.7)
        ax.plot(timestamps, accel_euler[:, i], 'g-', label='Accelerometer', alpha=0.7)
        ax.plot(timestamps, comp_euler[:, i], 'r-', label='Complementary Filter', alpha=0.7)
        ax.plot(timestamps, ukf_euler[:, i], 'm-', label='UKF', alpha=0.7)
        ax.plot(timestamps, vicon_euler[:, i], 'k--', label='Vicon (Ground Truth)', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{name} (degrees)')
        ax.set_title(f'{name} Angle - ZYX Convention')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_image}")
    plt.show()

def calculate_errors(csv_file):
    """
    Calculate and print error statistics
    """
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    # Load data
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    
    gyro_euler = data[:, 1:4]
    comp_euler = data[:, 7:10]
    ukf_euler = data[:, 10:13]
    vicon_euler = data[:, 13:16]
    
    # Remove rows with missing Vicon data
    valid_rows = ~np.isnan(vicon_euler).any(axis=1)
    
    gyro_euler = gyro_euler[valid_rows]
    comp_euler = comp_euler[valid_rows]
    ukf_euler = ukf_euler[valid_rows]
    vicon_euler = vicon_euler[valid_rows]
    
    # Calculate errors (handle angle wrapping)
    def wrap_angle(angle):
        return (angle + 180) % 360 - 180
    
    gyro_error = np.abs(wrap_angle(gyro_euler - vicon_euler))
    comp_error = np.abs(wrap_angle(comp_euler - vicon_euler))
    ukf_error = np.abs(wrap_angle(ukf_euler - vicon_euler))
    
    # Print statistics
    angle_names = ['Yaw', 'Pitch', 'Roll']
    
    print("\nMean Absolute Error (MAE):")
    print("-" * 60)
    print(f"{'Method':<25} {'Yaw (°)':<12} {'Pitch (°)':<12} {'Roll (°)':<12}")
    print("-" * 60)
    
    gyro_mae = np.mean(gyro_error, axis=0)
    print(f"{'Gyro Integration':<25} {gyro_mae[0]:>10.2f}  {gyro_mae[1]:>10.2f}  {gyro_mae[2]:>10.2f}")
    
    comp_mae = np.mean(comp_error, axis=0)
    print(f"{'Complementary Filter':<25} {comp_mae[0]:>10.2f}  {comp_mae[1]:>10.2f}  {comp_mae[2]:>10.2f}")
    
    ukf_mae = np.mean(ukf_error, axis=0)
    print(f"{'UKF':<25} {ukf_mae[0]:>10.2f}  {ukf_mae[1]:>10.2f}  {ukf_mae[2]:>10.2f}")
    
    print("\nRoot Mean Square Error (RMSE):")
    print("-" * 60)
    print(f"{'Method':<25} {'Yaw (°)':<12} {'Pitch (°)':<12} {'Roll (°)':<12}")
    print("-" * 60)
    
    gyro_rmse = np.sqrt(np.mean(gyro_error**2, axis=0))
    print(f"{'Gyro Integration':<25} {gyro_rmse[0]:>10.2f}  {gyro_rmse[1]:>10.2f}  {gyro_rmse[2]:>10.2f}")
    
    comp_rmse = np.sqrt(np.mean(comp_error**2, axis=0))
    print(f"{'Complementary Filter':<25} {comp_rmse[0]:>10.2f}  {comp_rmse[1]:>10.2f}  {comp_rmse[2]:>10.2f}")
    
    ukf_rmse = np.sqrt(np.mean(ukf_error**2, axis=0))
    print(f"{'UKF':<25} {ukf_rmse[0]:>10.2f}  {ukf_rmse[1]:>10.2f}  {ukf_rmse[2]:>10.2f}")
    
    print("\nMaximum Error:")
    print("-" * 60)
    print(f"{'Method':<25} {'Yaw (°)':<12} {'Pitch (°)':<12} {'Roll (°)':<12}")
    print("-" * 60)
    
    gyro_max = np.max(gyro_error, axis=0)
    print(f"{'Gyro Integration':<25} {gyro_max[0]:>10.2f}  {gyro_max[1]:>10.2f}  {gyro_max[2]:>10.2f}")
    
    comp_max = np.max(comp_error, axis=0)
    print(f"{'Complementary Filter':<25} {comp_max[0]:>10.2f}  {comp_max[1]:>10.2f}  {comp_max[2]:>10.2f}")
    
    ukf_max = np.max(ukf_error, axis=0)
    print(f"{'UKF':<25} {ukf_max[0]:>10.2f}  {ukf_max[1]:>10.2f}  {ukf_max[2]:>10.2f}")
    
    print("="*60 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Convert MAT to TXT:")
        print("    python process_data.py convert_imu imuRaw1.mat imuRaw1.txt")
        print("    python process_data.py convert_vicon viconRot1.mat viconRot1.txt")
        print("  Calibrate IMU:")
        print("    python process_data.py calibrate imuRaw1.mat IMUParams.mat imuRaw1_cal.txt")
        print("  Plot results:")
        print("    python process_data.py plot results.csv [output.png]")
        print("  Calculate errors:")
        print("    python process_data.py errors results.csv")
        return
    
    command = sys.argv[1]
    
    if command == 'convert_imu':
        if len(sys.argv) < 4:
            print("Usage: python process_data.py convert_imu <input.mat> <output.txt>")
            return
        convert_imu_mat_to_txt(sys.argv[2], sys.argv[3])
    
    elif command == 'convert_vicon':
        if len(sys.argv) < 4:
            print("Usage: python process_data.py convert_vicon <input.mat> <output.txt>")
            return
        convert_vicon_mat_to_txt(sys.argv[2], sys.argv[3])
    
    elif command == 'calibrate':
        if len(sys.argv) < 5:
            print("Usage: python process_data.py calibrate <imu.mat> <params.mat> <output.txt>")
            return
        calibrate_imu_data(sys.argv[2], sys.argv[3], sys.argv[4])
    
    elif command == 'plot':
        if len(sys.argv) < 3:
            print("Usage: python process_data.py plot <results.csv> [output.png]")
            return
        output_img = sys.argv[3] if len(sys.argv) >= 4 else 'results_plot.png'
        plot_results(sys.argv[2], output_img)
    
    elif command == 'errors':
        if len(sys.argv) < 3:
            print("Usage: python process_data.py errors <results.csv>")
            return
        calculate_errors(sys.argv[2])
    
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: convert_imu, convert_vicon, calibrate, plot, errors")

if __name__ == '__main__':
    main()