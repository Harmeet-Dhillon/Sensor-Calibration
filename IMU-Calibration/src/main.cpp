#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <matio.h>
#include "ukf.h"

// Structures
struct IMUData {
    std::vector<double> timestamps;
    std::vector<Eigen::Vector3d> accel_raw;
    std::vector<Eigen::Vector3d> gyro_raw;
};

struct CalibratedIMU {
    std::vector<double> timestamps;
    std::vector<Eigen::Vector3d> accel_phys;
    std::vector<Eigen::Vector3d> gyro_phys;
};

struct ViconData {
    std::vector<double> timestamps;
    std::vector<Eigen::Matrix3d> rotations;
};

struct IMUParams {
    Eigen::Vector3d accel_scale;
    Eigen::Vector3d accel_bias;
};

// Helper functions
bool loadIMUParams(const std::string& filename, IMUParams& params) {
    mat_t* matfp = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
    if (!matfp) {
        std::cerr << "Error opening IMUParams.mat" << std::endl;
        return false;
    }
    matvar_t* matvar = Mat_VarRead(matfp, "IMUParams");
    if (!matvar) {
        std::cerr << "Error reading IMUParams" << std::endl;
        Mat_Close(matfp);
        return false;
    }
    double* data = static_cast<double*>(matvar->data);
    params.accel_scale << data[0], data[2], data[4];
    params.accel_bias << data[1], data[3], data[5];
    Mat_VarFree(matvar);
    Mat_Close(matfp);
    return true;
}

bool loadIMURaw(const std::string& filename, IMUData& imu_data) {
    mat_t* matfp = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
    if (!matfp) return false;
    
    matvar_t* ts_var = Mat_VarRead(matfp, "ts");
    if (!ts_var) { Mat_Close(matfp); return false; }
    
    int n_samples = ts_var->dims[1];
    double* ts_data = static_cast<double*>(ts_var->data);
    imu_data.timestamps.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        imu_data.timestamps[i] = ts_data[i];
    }
    Mat_VarFree(ts_var);
    
    matvar_t* vals_var = Mat_VarRead(matfp, "vals");
    if (!vals_var) { Mat_Close(matfp); return false; }
    
    double* vals_data = static_cast<double*>(vals_var->data);
    imu_data.accel_raw.resize(n_samples);
    imu_data.gyro_raw.resize(n_samples);
    
    for (int i = 0; i < n_samples; i++) {
        imu_data.accel_raw[i] << vals_data[0 + i*6], vals_data[1 + i*6], vals_data[2 + i*6];
        imu_data.gyro_raw[i] << vals_data[3 + i*6], vals_data[4 + i*6], vals_data[5 + i*6];
    }
    
    Mat_VarFree(vals_var);
    Mat_Close(matfp);
    std::cout << "Loaded " << n_samples << " IMU samples" << std::endl;
    return true;
}

bool loadViconData(const std::string& filename, ViconData& vicon_data) {
    mat_t* matfp = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
    if (!matfp) return false;
    
    matvar_t* ts_var = Mat_VarRead(matfp, "ts");
    if (!ts_var) { Mat_Close(matfp); return false; }
    
    int n_samples = ts_var->dims[1];
    double* ts_data = static_cast<double*>(ts_var->data);
    vicon_data.timestamps.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        vicon_data.timestamps[i] = ts_data[i];
    }
    Mat_VarFree(ts_var);
    
    matvar_t* rots_var = Mat_VarRead(matfp, "rots");
    if (!rots_var) { Mat_Close(matfp); return false; }
    
    double* rots_data = static_cast<double*>(rots_var->data);
    vicon_data.rotations.resize(n_samples);
    
    for (int i = 0; i < n_samples; i++) {
        Eigen::Matrix3d R;
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                R(row, col) = rots_data[row + col*3 + i*9];
            }
        }
        vicon_data.rotations[i] = R;
    }
    
    Mat_VarFree(rots_var);
    Mat_Close(matfp);
    std::cout << "Loaded " << n_samples << " Vicon samples" << std::endl;
    return true;
}

void calibrateAccel(const IMUData& raw, const IMUParams& params, CalibratedIMU& calibrated) {
    calibrated.timestamps = raw.timestamps;
    calibrated.accel_phys.resize(raw.accel_raw.size());
    for (size_t i = 0; i < raw.accel_raw.size(); i++) {
        for (int j = 0; j < 3; j++) {
            calibrated.accel_phys[i](j) = raw.accel_raw[i](j) * params.accel_scale(j) + params.accel_bias(j);
        }
    }
}

void calibrateGyro(const IMUData& raw, CalibratedIMU& calibrated) {
    calibrated.gyro_phys.resize(raw.gyro_raw.size());
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    int n_bias_samples = std::min(200, (int)raw.gyro_raw.size());
    for (int i = 0; i < n_bias_samples; i++) {
        gyro_bias += raw.gyro_raw[i];
    }
    gyro_bias /= n_bias_samples;
    const double scale = (3300.0 / 1023.0) * (M_PI / 180.0) * 0.3;
    for (size_t i = 0; i < raw.gyro_raw.size(); i++) {
        calibrated.gyro_phys[i] = scale * (raw.gyro_raw[i] - gyro_bias);
    }
}

Eigen::Quaterniond slerp(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2, double t) {
    Eigen::Quaterniond q1_norm = q1.normalized();
    Eigen::Quaterniond q2_norm = q2.normalized();
    double dot = q1_norm.dot(q2_norm);
    if (dot < 0.0) {
        q2_norm.coeffs() = -q2_norm.coeffs();
        dot = -dot;
    }
    if (dot > 0.9995) {
        Eigen::Quaterniond result;
        result.coeffs() = q1_norm.coeffs() * (1.0 - t) + q2_norm.coeffs() * t;
        return result.normalized();
    }
    double theta = std::acos(dot);
    double sin_theta = std::sin(theta);
    double w1 = std::sin((1.0 - t) * theta) / sin_theta;
    double w2 = std::sin(t * theta) / sin_theta;
    Eigen::Quaterniond result;
    result.coeffs() = w1 * q1_norm.coeffs() + w2 * q2_norm.coeffs();
    return result.normalized();
}

std::vector<Eigen::Quaterniond> synchronizeVicon(const ViconData& vicon, 
                                                  const std::vector<double>& imu_ts,
                                                  std::vector<bool>& valid_mask) {
    std::vector<Eigen::Quaterniond> synced_vicon;
    synced_vicon.reserve(imu_ts.size());
    valid_mask.resize(imu_ts.size(), false);
    
    std::vector<Eigen::Quaterniond> vicon_quats;
    for (const auto& R : vicon.rotations) {
        vicon_quats.push_back(Eigen::Quaterniond(R));
    }
    
    double vicon_t_min = vicon.timestamps.front();
    double vicon_t_max = vicon.timestamps.back();
    
    for (size_t i = 0; i < imu_ts.size(); i++) {
        double t = imu_ts[i];
        if (t < vicon_t_min || t > vicon_t_max) {
            synced_vicon.push_back(Eigen::Quaterniond::Identity());
            continue;
        }
        valid_mask[i] = true;
        size_t idx_low = 0;
        for (size_t j = 0; j < vicon.timestamps.size() - 1; j++) {
            if (vicon.timestamps[j] <= t && t <= vicon.timestamps[j + 1]) {
                idx_low = j;
                break;
            }
        }
        size_t idx_high = idx_low + 1;
        double t_low = vicon.timestamps[idx_low];
        double t_high = vicon.timestamps[idx_high];
        double alpha = (t - t_low) / (t_high - t_low);
        Eigen::Quaterniond q_interp = slerp(vicon_quats[idx_low], vicon_quats[idx_high], alpha);
        synced_vicon.push_back(q_interp);
    }
    
    int valid_count = std::count(valid_mask.begin(), valid_mask.end(), true);
    std::cout << "Synchronized " << valid_count << " samples" << std::endl;
    return synced_vicon;
}

Eigen::Vector3d quaternionToEulerZXY(const Eigen::Quaterniond& q) {
    Eigen::Quaterniond qn = q.normalized();
    double qw = qn.w();
    double qx = qn.x();
    double qy = qn.y();
    double qz = qn.z();
    
    double roll = std::atan2(2.0 * (qw * qx + qy * qz), 
                             1.0 - 2.0 * (qx * qx + qy * qy));
    
    double pitch_sin = 2.0 * (qw * qy - qz * qx);
    pitch_sin = std::clamp(pitch_sin, -1.0, 1.0);
    double pitch = std::asin(pitch_sin);
    
    double yaw = std::atan2(2.0 * (qw * qz + qx * qy), 
                            1.0 - 2.0 * (qy * qy + qz * qz));
    
    return Eigen::Vector3d(yaw * 180.0 / M_PI,
                          roll * 180.0 / M_PI,
                          pitch * 180.0 / M_PI);
}

Eigen::Vector3d orientationFromAccel(const Eigen::Vector3d& accel) {
    Eigen::Vector3d acc_norm = accel.normalized();
    double pitch = std::atan2(-acc_norm.x(), std::sqrt(acc_norm.y()*acc_norm.y() + acc_norm.z()*acc_norm.z()));
    double roll = std::atan2(acc_norm.y(), acc_norm.z());
    return Eigen::Vector3d(0.0, roll * 180.0 / M_PI, pitch * 180.0 / M_PI);
}

std::vector<Eigen::Vector3d> orientationFromGyro(const CalibratedIMU& imu, 
                                                  const Eigen::Vector3d& initial_euler) {
    std::vector<Eigen::Vector3d> orientations;
    Eigen::Vector3d current_euler = initial_euler;
    orientations.push_back(current_euler);
    
    for (size_t i = 1; i < imu.timestamps.size(); i++) {
        double dt = imu.timestamps[i] - imu.timestamps[i-1];
        if (dt <= 0 || dt > 0.1) {
            orientations.push_back(current_euler);
            continue;
        }
        Eigen::Vector3d omega_deg = imu.gyro_phys[i] * 180.0 / M_PI;
        current_euler += omega_deg * dt;
        orientations.push_back(current_euler);
    }
    return orientations;
}

std::vector<Eigen::Vector3d> orientationFromComplementary(const CalibratedIMU& imu,
                                                           const Eigen::Vector3d& initial_euler,
                                                           double alpha = 0.98) {
    std::vector<Eigen::Vector3d> orientations;
    Eigen::Vector3d current = initial_euler;
    orientations.push_back(current);
    
    for (size_t i = 1; i < imu.timestamps.size(); i++) {
        double dt = imu.timestamps[i] - imu.timestamps[i-1];
        if (dt <= 0 || dt > 0.1) {
            orientations.push_back(current);
            continue;
        }
        
        Eigen::Vector3d omega_deg = imu.gyro_phys[i] * 180.0 / M_PI;
        Eigen::Vector3d gyro_pred = current + omega_deg * dt;
        Eigen::Vector3d accel_meas = orientationFromAccel(imu.accel_phys[i]);
        
        current(0) = gyro_pred(0);
        current(1) = alpha * gyro_pred(1) + (1 - alpha) * accel_meas(1);
        current(2) = alpha * gyro_pred(2) + (1 - alpha) * accel_meas(2);
        
        orientations.push_back(current);
    }
    return orientations;
}

std::vector<Eigen::Vector3d> orientationFromMadgwick(const CalibratedIMU& imu,
                                                      const Eigen::Vector3d& initial_euler,
                                                      double beta = 0.1) {
    std::vector<Eigen::Vector3d> orientations;
    
    double y = initial_euler(0) * M_PI / 180.0;
    double r = initial_euler(1) * M_PI / 180.0;
    double p = initial_euler(2) * M_PI / 180.0;
    
    double cy = std::cos(y * 0.5), sy = std::sin(y * 0.5);
    double cr = std::cos(r * 0.5), sr = std::sin(r * 0.5);
    double cp = std::cos(p * 0.5), sp = std::sin(p * 0.5);
    
    Eigen::Quaterniond q;
    q.w() = cy * cr * cp + sy * sr * sp;
    q.x() = cy * sr * cp - sy * cr * sp;
    q.y() = cy * cr * sp + sy * sr * cp;
    q.z() = sy * cr * cp - cy * sr * sp;
    q.normalize();
    
    orientations.push_back(initial_euler);
    
    for (size_t i = 1; i < imu.timestamps.size(); i++) {
        double dt = imu.timestamps[i] - imu.timestamps[i-1];
        if (dt <= 0 || dt > 0.1) {
            orientations.push_back(quaternionToEulerZXY(q));
            continue;
        }
        
        Eigen::Vector3d omega = imu.gyro_phys[i];
        Eigen::Vector3d accel = imu.accel_phys[i].normalized();
        
        Eigen::Quaterniond q_omega(0, omega.x(), omega.y(), omega.z());
        Eigen::Quaterniond q_dot_gyro;
        q_dot_gyro.coeffs() = 0.5 * (q * q_omega).coeffs();
        
        double qw = q.w(), qx = q.x(), qy = q.y(), qz = q.z();
        Eigen::Vector3d f;
        f(0) = 2.0*(qx*qz - qw*qy) - accel.x();
        f(1) = 2.0*(qw*qx + qy*qz) - accel.y();
        f(2) = 2.0*(0.5 - qx*qx - qy*qy) - accel.z();
        
        Eigen::Matrix<double, 3, 4> J;
        J << -2*qy,  2*qz, -2*qw,  2*qx,
              2*qx,  2*qw,  2*qz,  2*qy,
              0,    -4*qx, -4*qy,  0;
        
        Eigen::Vector4d gradient = J.transpose() * f;
        double norm = gradient.norm();
        if (norm > 1e-10) {
            gradient /= norm;
        }
        
        Eigen::Quaterniond q_dot;
        q_dot.coeffs() = q_dot_gyro.coeffs() - beta * gradient;
        
        q.coeffs() += q_dot.coeffs() * dt;
        q.normalize();
        
        orientations.push_back(quaternionToEulerZXY(q));
    }
    return orientations;
}

bool saveAllResults(const std::string& filename,
                    const std::vector<double>& timestamps,
                    const std::vector<Eigen::Vector3d>& vicon_euler,
                    const std::vector<Eigen::Vector3d>& gyro_euler,
                    const std::vector<Eigen::Vector3d>& accel_euler,
                    const std::vector<Eigen::Vector3d>& comp_euler,
                    const std::vector<Eigen::Vector3d>& madg_euler,
                    const std::vector<Eigen::Vector3d>& ukf_euler,
                    const std::vector<bool>& valid_mask) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file" << std::endl;
        return false;
    }
    
    // ========== FIX: Set high precision for timestamps ==========
    outfile << std::fixed << std::setprecision(10);
    // ============================================================
    
    outfile << "Timestamp,";
    outfile << "Vicon_Yaw,Vicon_Roll,Vicon_Pitch,";
    outfile << "Gyro_Yaw,Gyro_Roll,Gyro_Pitch,";
    outfile << "Accel_Yaw,Accel_Roll,Accel_Pitch,";
    outfile << "Complementary_Yaw,Complementary_Roll,Complementary_Pitch,";
    outfile << "Madgwick_Yaw,Madgwick_Roll,Madgwick_Pitch,";
    outfile << "UKF_Yaw,UKF_Roll,UKF_Pitch,Valid" << std::endl;
    
    for (size_t i = 0; i < timestamps.size(); i++) {
        outfile << timestamps[i] << ",";
        outfile << vicon_euler[i](0) << "," << vicon_euler[i](1) << "," << vicon_euler[i](2) << ",";
        outfile << gyro_euler[i](0) << "," << gyro_euler[i](1) << "," << gyro_euler[i](2) << ",";
        outfile << accel_euler[i](0) << "," << accel_euler[i](1) << "," << accel_euler[i](2) << ",";
        outfile << comp_euler[i](0) << "," << comp_euler[i](1) << "," << comp_euler[i](2) << ",";
        outfile << madg_euler[i](0) << "," << madg_euler[i](1) << "," << madg_euler[i](2) << ",";
        outfile << ukf_euler[i](0) << "," << ukf_euler[i](1) << "," << ukf_euler[i](2) << ",";
        outfile << (valid_mask[i] ? 1 : 0) << std::endl;
    }
    
    outfile.close();
    std::cout << "Results saved to: " << filename << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Orientation Tracking: All Methods ===" << std::endl;
    
    std::string pp_dir = "../pp/";
    std::string data_dir = pp_dir + "Data/Train/IMU/";
    std::string vicon_dir = pp_dir + "Data/Train/Vicon/";
    std::string params_file = pp_dir + "IMUParams.mat";
    std::string output_dir = "../processed_data/";
    
    system(("mkdir -p " + output_dir).c_str());
    
    IMUParams imu_params;
    if (!loadIMUParams(params_file, imu_params)) {
        return 1;
    }
    
    for (int dataset_num = 1; dataset_num <= 6; dataset_num++) {
        std::cout << "\n=== Dataset " << dataset_num << " ===" << std::endl;
        
        IMUData imu_raw;
        if (!loadIMURaw(data_dir + "imuRaw" + std::to_string(dataset_num) + ".mat", imu_raw)) {
            continue;
        }
        
        ViconData vicon_data;
        bool has_vicon = loadViconData(vicon_dir + "viconRot" + std::to_string(dataset_num) + ".mat", vicon_data);
        
        CalibratedIMU imu_calib;
        calibrateAccel(imu_raw, imu_params, imu_calib);
        calibrateGyro(imu_raw, imu_calib);
        
        std::vector<Eigen::Quaterniond> vicon_synced;
        std::vector<bool> valid_mask;
        
        if (has_vicon) {
            vicon_synced = synchronizeVicon(vicon_data, imu_calib.timestamps, valid_mask);
        } else {
            valid_mask.resize(imu_calib.timestamps.size(), false);
            vicon_synced.resize(imu_calib.timestamps.size(), Eigen::Quaterniond::Identity());
        }
        
        // Convert Vicon to Euler
        std::vector<Eigen::Vector3d> vicon_euler;
        for (const auto& q : vicon_synced) {
            vicon_euler.push_back(quaternionToEulerZXY(q));
        }
        
        // ========== CRITICAL FIX: Match Python's coordinate frame conversion ==========
        // Get initial orientation from first valid Vicon (use RAW values)
        Eigen::Vector3d initial_euler = Eigen::Vector3d::Zero();
        if (has_vicon) {
            for (size_t i = 0; i < valid_mask.size(); i++) {
                if (valid_mask[i]) {
                    Eigen::Vector3d vicon_raw = vicon_euler[i];  // Use RAW Vicon
                    // Python does: initial_orientation[1] = -vicon_raw[1]
                    //              initial_orientation[2] = -vicon_raw[2]
                    // This converts from Vicon frame to IMU body frame
                    initial_euler(0) = vicon_raw(0);   // Yaw unchanged
                    initial_euler(1) = -vicon_raw(1);  // NEGATE roll (coordinate frame difference)
                    initial_euler(2) = -vicon_raw(2);  // NEGATE pitch (coordinate frame difference)
                    break;
                }
            }
        }
        // ==============================================================================
        
        std::cout << "Computing all methods..." << std::endl;
        
        // Gyro-only
        auto gyro_euler = orientationFromGyro(imu_calib, initial_euler);
        
        // Accel-only
        std::vector<Eigen::Vector3d> accel_euler;
        for (const auto& acc : imu_calib.accel_phys) {
            accel_euler.push_back(orientationFromAccel(acc));
        }
        
        // Complementary Filter
        auto comp_euler = orientationFromComplementary(imu_calib, initial_euler, 0.98);
        
        // Madgwick Filter
        auto madg_euler = orientationFromMadgwick(imu_calib, initial_euler, 0.1);
        
        // ========== UKF with FIXED INITIAL ORIENTATION ==========
        std::cout << "Running UKF..." << std::endl;
        UKF ukf;
        
        // Match Python noise parameters exactly
        ukf.Q = Eigen::Matrix<double, 6, 6>::Zero();
        ukf.Q(0,0) = 3.4; ukf.Q(1,1) = 3.4; ukf.Q(2,2) = 3.4;
        ukf.Q(3,3) = 0.5; ukf.Q(4,4) = 0.5; ukf.Q(5,5) = 0.5;
        
        ukf.R_gyro = Eigen::Matrix3d::Zero();
        ukf.R_gyro(0,0) = 15.0; ukf.R_gyro(1,1) = 15.0; ukf.R_gyro(2,2) = 15.0;
        
        ukf.R_accel = Eigen::Matrix3d::Zero();
        ukf.R_accel(0,0) = 15.0; ukf.R_accel(1,1) = 15.0; ukf.R_accel(2,2) = 25.0;
        
        ukf.Xk = Eigen::Matrix<double, 6, 1>::Zero();
        ukf.Pk = Eigen::Matrix<double, 6, 6>::Identity() * 1.0;
        
        if (has_vicon) {
            // CRITICAL: Match Python's initialization EXACTLY
            // Python uses first gyro measurement for initial angular velocity (line ~630)
            // Find first valid sample
            size_t first_valid = 0;
            for (size_t i = 0; i < valid_mask.size(); i++) {
                if (valid_mask[i]) {
                    first_valid = i;
                    break;
                }
            }
            
            // Initialize with BOTH orientation AND first gyro measurement
            Eigen::Vector3d first_gyro = imu_calib.gyro_phys[first_valid];
            ukf.setInitialState(initial_euler, first_gyro);
            
            std::cout << "  Initial orientation: " << initial_euler.transpose() << " deg" << std::endl;
            std::cout << "  Initial angular vel: " << first_gyro.transpose() << " rad/s" << std::endl;
        }
        // ============================================================
        
        std::vector<Eigen::Vector3d> ukf_euler;
        ukf_euler.push_back(ukf.getEulerZXY());
        
        for (size_t i = 1; i < imu_calib.timestamps.size(); i++) {
            double dt = imu_calib.timestamps[i] - imu_calib.timestamps[i-1];
            if (dt <= 0 || dt > 0.1) {
                ukf_euler.push_back(ukf_euler.back());
                continue;
            }
            
            ukf.predict(dt);
            ukf.measurementGyro(imu_calib.gyro_phys[i]);
            ukf.measurementAccel(imu_calib.accel_phys[i]);
            
            ukf_euler.push_back(ukf.getEulerZXY());
            
            if (i % 1000 == 0) {
                std::cout << "  " << i << "/" << imu_calib.timestamps.size() << std::endl;
            }
        }
        
        // Print UKF diagnostics
        ukf.printDiagnostics();
        
        // Save all results
        std::string output_file = output_dir + "all_methods_dataset" + std::to_string(dataset_num) + ".csv";
        saveAllResults(output_file, imu_calib.timestamps, vicon_euler, gyro_euler, accel_euler,
                      comp_euler, madg_euler, ukf_euler, valid_mask);
    }
    
    std::cout << "\n=== All datasets processed ===" << std::endl;
    std::cout << "Run: cd ../scripts && python3 visualize_all_methods.py" << std::endl;
    
    return 0;
}