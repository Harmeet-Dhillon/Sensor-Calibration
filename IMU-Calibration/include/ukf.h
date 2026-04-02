#ifndef UKF_H
#define UKF_H

#include <Eigen/Dense>
#include <cmath>
#include <cfloat>

class UKF {
public:
    // Constructor
    UKF();
    
    // Main UKF functions
    void predict(double dt);
    void measurementGyro(const Eigen::Vector3d &z_gyro);
    void measurementAccel(const Eigen::Vector3d &z_accel);
    void measurementMag(const Eigen::Vector3d &z_mag);
    
    // Helper functions
    Eigen::Vector3d quaternionToRotationVector(const Eigen::Quaterniond &q) const;
    Eigen::Quaterniond rotationVectorToQuaternion(const Eigen::Vector3d &v) const;
    Eigen::MatrixXd choleskydecomp(const Eigen::MatrixXd &M);
    
    // Ensure matrix is positive definite
    Eigen::MatrixXd makePositiveDefinite(const Eigen::MatrixXd& M, double min_eigenvalue = 1e-6);
    
    // State access functions
    Eigen::Quaterniond getOrientation() const;
    Eigen::Vector3d getAngularVelocity() const;
    Eigen::Matrix3d getRotationMatrix() const;
    Eigen::Vector3d getEulerZXY() const;  // Returns [yaw, roll, pitch] in degrees
    
    // State initialization functions
    void setInitialOrientation(const Eigen::Quaterniond& q);
    void setInitialOrientation(const Eigen::Vector3d& euler_zxy_deg);
    void setAngularVelocity(const Eigen::Vector3d& omega);
    
    // ========== NEW: Match Python's initialization exactly ==========
    // Python initializes UKF with BOTH orientation AND first gyro measurement
    // Usage: ukf.setInitialState(initial_euler, first_gyro_measurement);
    void setInitialState(const Eigen::Vector3d& euler_zxy_deg, const Eigen::Vector3d& omega_xyz);
    // ================================================================
    
    // ========== NEW: Diagnostics functions ==========
    void printDiagnostics() const;
    void resetDiagnostics();
    // ===============================================
    
    // State and covariance matrices
    Eigen::Matrix<double, 6, 1> Xk;  // State vector [rotation_vector(3), angular_velocity(3)]
    Eigen::Matrix<double, 6, 6> Pk;  // State covariance
    
    // Noise matrices
    Eigen::Matrix<double, 6, 6> Q;   // Process noise
    Eigen::Matrix3d R_gyro;           // Gyroscope measurement noise
    Eigen::Matrix3d R_accel;          // Accelerometer measurement noise
    Eigen::Matrix3d R_mag;            // Magnetometer measurement noise
    
    // Global reference vectors
    Eigen::Vector3d gravity_global;   // Gravity in global frame
    Eigen::Vector3d mag_global;       // Magnetic field in global frame
    
private:
    void update(const Eigen::MatrixXd &Pxy, const Eigen::MatrixXd &Pvv, const Eigen::VectorXd &vk);
    
    // Convergence threshold for quaternion mean
    double threshold;
    
    // Sigma points storage
    Eigen::MatrixXd sigma_points_Y;  // Transformed sigma points
    Eigen::MatrixXd Wprime;          // Error matrix for covariance computation
    
    // ========== NEW: Diagnostics counters ==========
    mutable int iteration_count;     // Total predict() calls
    mutable int cholesky_failures;   // Number of Cholesky decomposition failures
    // ==============================================
};

#endif // UKF_H