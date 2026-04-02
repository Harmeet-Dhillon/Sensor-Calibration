#include "ukf.h"
#include <iostream>

UKF::UKF() : threshold(1e-6) {
    Xk = Eigen::Matrix<double, 6, 1>::Zero();
    Pk = Eigen::Matrix<double, 6, 6>::Identity() * 1.0;  // Initial uncertainty
    
    // ========== MATCH WRAPPER.PY EXACTLY ==========
    // Wrapper.py line 631-633
    Q = Eigen::Matrix<double, 6, 6>::Zero();
    Q(0,0) = 3.4; Q(1,1) = 3.4; Q(2,2) = 3.4;  // Rotation noise
    Q(3,3) = 0.5; Q(4,4) = 0.5; Q(5,5) = 0.5;  // Angular velocity noise
    
    R_gyro = Eigen::Matrix3d::Zero();
    R_gyro(0,0) = 15.0; R_gyro(1,1) = 15.0; R_gyro(2,2) = 15.0;
    
    R_accel = Eigen::Matrix3d::Zero();
    R_accel(0,0) = 15.0; R_accel(1,1) = 15.0; R_accel(2,2) = 25.0;
    
    R_mag = Eigen::Matrix3d::Identity() * 1e-1;
    // ==============================================
    
    // ========== FIX: POSITIVE GRAVITY (match Wrapper.py) ==========
    gravity_global = Eigen::Vector3d(0, 0, 9.81);  // Was -9.81, now +9.81
    // ==============================================================
    
    mag_global = Eigen::Vector3d(1, 0, 0);
    sigma_points_Y = Eigen::MatrixXd::Zero(12, 7);
    Wprime = Eigen::MatrixXd::Zero(12, 6);
    
    // Diagnostics
    iteration_count = 0;
    cholesky_failures = 0;
}

// Helper to ensure matrix is positive definite
Eigen::MatrixXd UKF::makePositiveDefinite(const Eigen::MatrixXd& M, double min_eigenvalue) {
    int n = M.rows();
    
    // Force symmetry
    Eigen::MatrixXd M_sym = 0.5 * (M + M.transpose());
    
    // Eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M_sym);
    
    if (es.info() != Eigen::Success) {
        std::cerr << "Eigenvalue decomposition failed, returning identity\n";
        return Eigen::MatrixXd::Identity(n, n) * min_eigenvalue;
    }
    
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXd eigenvectors = es.eigenvectors();
    
    // Clamp all eigenvalues to minimum
    bool modified = false;
    for (int i = 0; i < eigenvalues.size(); i++) {
        if (eigenvalues(i) < min_eigenvalue) {
            eigenvalues(i) = min_eigenvalue;
            modified = true;
        }
    }
    
    if (modified) {
        return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    }
    
    return M_sym;
}

void UKF::predict(double dt) {
    if (dt <= 0 || dt > 0.2) {
        std::cerr << "Invalid dt: " << dt << ", skipping prediction\n";
        return;
    }
    
    iteration_count++;
    
    // Ensure Pk is positive definite
    Pk = makePositiveDefinite(Pk, 1e-6);
    
    Eigen::MatrixXd tot = Pk + Q;
    tot = makePositiveDefinite(tot, 1e-6);
    
    // Cholesky decomposition
    Eigen::MatrixXd S = choleskydecomp(tot);
    int n = S.rows();
    
    // Scale by sqrt(12) to match Wrapper.py line 284
    Eigen::MatrixXd S_T = S.transpose();
    double scale = std::sqrt(2.0 * n);  // sqrt(12) for n=6
    Eigen::MatrixXd Xmat(2*n, 7);
    
    // Extract current state
    Eigen::Vector3d rot_vec_k = Xk.head<3>();
    Eigen::Quaterniond qk = rotationVectorToQuaternion(rot_vec_k);
    qk.normalize();
    Eigen::Vector3d wk = Xk.tail<3>();
    
    // Generate sigma points
    for(int i = 0; i < n; i++) {
        int c = 2*i;
        
        Eigen::Vector3d rot_i = scale * S_T.block<1,3>(i, 0).transpose();
        Eigen::Vector3d wi = scale * S_T.block<1,3>(i, 3).transpose();
        
        // POSITIVE sigma point
        Eigen::Quaterniond qi = rotationVectorToQuaternion(rot_i);
        Eigen::Quaterniond q = qk * qi;
        q.normalize();
        Eigen::Vector3d w_new = wk + wi;
        
        Xmat(c, 0) = q.w();
        Xmat(c, 1) = q.x();
        Xmat(c, 2) = q.y();
        Xmat(c, 3) = q.z();
        Xmat.block<1,3>(c, 4) = w_new.transpose();
        
        c++;
        
        // NEGATIVE sigma point
        Eigen::Quaterniond qineg = rotationVectorToQuaternion(-rot_i);
        Eigen::Quaterniond qneg = qk * qineg;
        qneg.normalize();
        Eigen::Vector3d w_newneg = wk - wi;
        
        Xmat(c, 0) = qneg.w();
        Xmat(c, 1) = qneg.x();
        Xmat(c, 2) = qneg.y();
        Xmat(c, 3) = qneg.z();
        Xmat.block<1,3>(c, 4) = w_newneg.transpose();
    }
    
    // Apply process model
    for(int j = 0; j < 2*n; j++) {
        Eigen::Quaterniond qj(Xmat(j,0), Xmat(j,1), Xmat(j,2), Xmat(j,3));
        qj.normalize();
        Eigen::Vector3d wj = Xmat.block<1,3>(j, 4).transpose();
        
        double angle = wj.norm() * dt;
        Eigen::Quaterniond qw;
        
        if(angle < 1e-10) {
            qw = Eigen::Quaterniond(1, 0, 0, 0);
        }
        else {
            Eigen::Vector3d axis = wj.normalized();
            double half = angle / 2.0;
            qw.w() = std::cos(half);
            qw.x() = axis.x() * std::sin(half);
            qw.y() = axis.y() * std::sin(half);
            qw.z() = axis.z() * std::sin(half);
            qw.normalize();
        }
        
        Eigen::Quaterniond q_new = qj * qw;
        q_new.normalize();
        
        Xmat.block<1,4>(j, 0) << q_new.w(), q_new.x(), q_new.y(), q_new.z();
        Xmat.block<1,3>(j, 4) = wj.transpose();  // Constant velocity
    }
    
    // Compute mean quaternion
    Eigen::Quaterniond qref = qk;
    double error = DBL_MAX;
    Eigen::MatrixXd Emat(2*n, 4);
    int max_iterations = 10;
    int iteration = 0;
    
    while(error > threshold && iteration < max_iterations) {
        Eigen::Vector3d e_avg = Eigen::Vector3d::Zero();
        
        for(int i = 0; i < 2*n; i++) {
            Eigen::Quaterniond qi(Xmat(i,0), Xmat(i,1), Xmat(i,2), Xmat(i,3));
            qi.normalize();
            
            Eigen::Quaterniond ei = qi * qref.inverse();
            
            // Handle sign ambiguity
            if (ei.w() < 0) {
                ei.coeffs() = -ei.coeffs();
            }
            
            Emat.row(i) << ei.w(), ei.x(), ei.y(), ei.z();
            Eigen::Vector3d ei_vec = quaternionToRotationVector(ei);
            e_avg += ei_vec;
        }
        
        e_avg /= (2*n);
        
        Eigen::Quaterniond qe = rotationVectorToQuaternion(e_avg);
        qe.normalize();
        
        error = e_avg.norm();
        qref = qe * qref;
        qref.normalize();
        
        iteration++;
    }
    
    // Compute mean angular velocity
    Eigen::Vector3d wref = Eigen::Vector3d::Zero();
    for(int i = 0; i < 2*n; i++) {
        Eigen::Vector3d wi = Xmat.block<1,3>(i, 4).transpose();
        wref += wi;
    }
    wref /= (2*n);
    
    // Update state
    Eigen::Vector3d qref_v = quaternionToRotationVector(qref);
    Xk.head<3>() = qref_v;
    Xk.tail<3>() = wref;
    
    // Compute covariance
    Wprime.resize(2*n, 6);
    
    for(int i = 0; i < 2*n; i++) {
        Eigen::Quaterniond ei(Emat(i,0), Emat(i,1), Emat(i,2), Emat(i,3));
        Eigen::Vector3d ei_vec = quaternionToRotationVector(ei);
        
        Eigen::Vector3d wi = Xmat.block<1,3>(i, 4).transpose();
        Eigen::Vector3d w_diff = wi - wref;
        
        Wprime.block<1,3>(i, 0) = ei_vec.transpose();
        Wprime.block<1,3>(i, 3) = w_diff.transpose();
    }
    
    Eigen::MatrixXd Wprime_T = Wprime.transpose();
    Pk = (Wprime_T * Wprime) / (2.0 * n);
    
    // Ensure positive definite
    Pk = makePositiveDefinite(Pk, 1e-6);
    
    sigma_points_Y = Xmat;
}

void UKF::measurementGyro(const Eigen::Vector3d &z_gyro) {
    int n = 6;
    int n_sigma = 2 * n;
    
    // ========== FIX: REORDER AXES TO MATCH WRAPPER.PY ==========
    // Wrapper.py line 690: z_gyro_actual = omega[[1,2,0], i]
    // This means: [y, z, x] order
    // Input z_gyro is [x, y, z], we need to reorder it
    Eigen::Vector3d z_gyro_reordered;
    z_gyro_reordered(0) = z_gyro(1);  // y
    z_gyro_reordered(1) = z_gyro(2);  // z
    z_gyro_reordered(2) = z_gyro(0);  // x
    // ===========================================================
    
    Eigen::MatrixXd Z_sigma(n_sigma, 3);
    for(int i = 0; i < n_sigma; i++) {
        // Also need to reorder sigma point angular velocities
        Eigen::Vector3d omega = sigma_points_Y.block<1,3>(i, 4).transpose();
        Z_sigma(i, 0) = omega(0);  // y
        Z_sigma(i, 1) = omega(1);  // z
        Z_sigma(i, 2) = omega(2);  // x
    }
    
    Eigen::Vector3d z_pred = Z_sigma.colwise().mean();
    
    Eigen::Matrix3d Pzz = Eigen::Matrix3d::Zero();
    for(int i = 0; i < n_sigma; i++) {
        Eigen::Vector3d diff = Z_sigma.row(i).transpose() - z_pred;
        Pzz += diff * diff.transpose();
    }
    Pzz /= n_sigma;
    
    Eigen::Matrix3d Pvv = Pzz + R_gyro;
    Pvv = makePositiveDefinite(Pvv, 1e-6);
    
    Eigen::MatrixXd Pxy = Eigen::MatrixXd::Zero(6, 3);
    for(int i = 0; i < n_sigma; i++) {
        Eigen::Matrix<double, 6, 1> w_diff = Wprime.row(i).transpose();
        Eigen::Vector3d z_diff = Z_sigma.row(i).transpose() - z_pred;
        Pxy += w_diff * z_diff.transpose();
    }
    Pxy /= n_sigma;
    
    Eigen::Vector3d vk = z_gyro_reordered - z_pred;
    
    // NO CLAMPING - match Wrapper.py exactly
    
    update(Pxy, Pvv, vk);
}

void UKF::measurementAccel(const Eigen::Vector3d &z_accel) {
    int n = 6;
    int n_sigma = 2 * n;
    
    Eigen::MatrixXd Z_sigma(n_sigma, 3);
    for(int i = 0; i < n_sigma; i++) {
        Eigen::Quaterniond qi(sigma_points_Y(i,0), sigma_points_Y(i,1),
                             sigma_points_Y(i,2), sigma_points_Y(i,3));
        qi.normalize();
        
        // gravity_global is now +9.81 (matches Wrapper.py)
        Eigen::Quaterniond g_quat(0, gravity_global.x(), gravity_global.y(), gravity_global.z());
        Eigen::Quaterniond g_body_quat = qi.inverse() * g_quat * qi;
        
        Z_sigma.row(i) << g_body_quat.x(), g_body_quat.y(), g_body_quat.z();
    }
    
    Eigen::Vector3d z_pred = Z_sigma.colwise().mean();
    
    Eigen::Matrix3d Pzz = Eigen::Matrix3d::Zero();
    for(int i = 0; i < n_sigma; i++) {
        Eigen::Vector3d diff = Z_sigma.row(i).transpose() - z_pred;
        Pzz += diff * diff.transpose();
    }
    Pzz /= n_sigma;
    
    Eigen::Matrix3d Pvv = Pzz + R_accel;
    Pvv = makePositiveDefinite(Pvv, 1e-6);
    
    Eigen::MatrixXd Pxy = Eigen::MatrixXd::Zero(6, 3);
    for(int i = 0; i < n_sigma; i++) {
        Eigen::Matrix<double, 6, 1> w_diff = Wprime.row(i).transpose();
        Eigen::Vector3d z_diff = Z_sigma.row(i).transpose() - z_pred;
        Pxy += w_diff * z_diff.transpose();
    }
    Pxy /= n_sigma;
    
    Eigen::Vector3d vk = z_accel - z_pred;
    
    // NO CLAMPING - match Wrapper.py exactly
    
    update(Pxy, Pvv, vk);
}

void UKF::update(const Eigen::MatrixXd &Pxy,
                 const Eigen::MatrixXd &Pvv,
                 const Eigen::VectorXd &vk)
{
    // Compute Kalman gain
    Eigen::MatrixXd K = Pxy * Pvv.inverse();
    Eigen::VectorXd dx = K * vk;

    // --- Rotation update (SO(3)) ---
    Eigen::Quaterniond q_old = rotationVectorToQuaternion(Xk.head<3>());
    q_old.normalize();
    
    Eigen::Quaterniond q_delta = rotationVectorToQuaternion(dx.head<3>());
    q_delta.normalize();
    
    // ========== FIX: MATCH WRAPPER.PY ORDER ==========
    // Wrapper.py line 548: q_updated = q_adjustment * q_mean
    // This means: q_delta * q_old (adjustment FIRST)
    Eigen::Quaterniond q_new = q_delta * q_old;  // CHANGED ORDER
    // =================================================
    
    q_new.normalize();
    Xk.head<3>() = quaternionToRotationVector(q_new);

    // --- Angular velocity update ---
    Xk.tail<3>() += dx.tail<3>();

    // --- Covariance update ---
    Pk -= K * Pvv * K.transpose();
    
    // Ensure positive definite
    Pk = makePositiveDefinite(Pk, 1e-6);
}

Eigen::MatrixXd UKF::choleskydecomp(const Eigen::MatrixXd &M) {
    int n = M.rows();
    
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    
    if (llt.info() == Eigen::NumericalIssue) {
        cholesky_failures++;
        if (cholesky_failures % 100 == 1) {
            std::cerr << "Cholesky failure #" << cholesky_failures 
                     << " at iteration " << iteration_count << std::endl;
        }
        
        // Use eigenvalue decomposition as fallback
        Eigen::MatrixXd M_fixed = makePositiveDefinite(M, 1e-6);
        
        Eigen::LLT<Eigen::MatrixXd> llt_fixed(M_fixed);
        if (llt_fixed.info() == Eigen::Success) {
            return llt_fixed.matrixL();
        }
        
        // Last resort: return diagonal matrix
        Eigen::VectorXd diag = M.diagonal().cwiseAbs().cwiseSqrt();
        for (int i = 0; i < diag.size(); i++) {
            if (diag(i) < 0.01) diag(i) = 0.01;
        }
        return diag.asDiagonal();
    }
    
    return llt.matrixL();
}

Eigen::Vector3d UKF::quaternionToRotationVector(const Eigen::Quaterniond &q) const {
    double angle = 2.0 * std::acos(std::clamp(q.w(), -1.0, 1.0));
    if(angle < 1e-10) return Eigen::Vector3d::Zero();
    double scale = angle / std::sin(angle / 2.0);
    return scale * Eigen::Vector3d(q.x(), q.y(), q.z());
}

Eigen::Quaterniond UKF::rotationVectorToQuaternion(const Eigen::Vector3d &v) const {
    double angle = v.norm();
    if(angle < 1e-10) return Eigen::Quaterniond(1, 0, 0, 0);
    Eigen::Vector3d axis = v.normalized();
    double half_angle = angle / 2.0;
    Eigen::Quaterniond q;
    q.w() = std::cos(half_angle);
    q.x() = axis.x() * std::sin(half_angle);
    q.y() = axis.y() * std::sin(half_angle);
    q.z() = axis.z() * std::sin(half_angle);
    return q;
}

Eigen::Quaterniond UKF::getOrientation() const {
    Eigen::Vector3d rot_vec = Xk.head<3>();
    Eigen::Quaterniond q = rotationVectorToQuaternion(rot_vec);
    q.normalize();
    return q;
}

Eigen::Vector3d UKF::getAngularVelocity() const {
    return Xk.tail<3>();
}

Eigen::Matrix3d UKF::getRotationMatrix() const {
    return getOrientation().toRotationMatrix();
}

Eigen::Vector3d UKF::getEulerZXY() const {
    Eigen::Quaterniond q = getOrientation();
    double qw = q.w(), qx = q.x(), qy = q.y(), qz = q.z();
    
    double roll = std::atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));
    double pitch_sin = 2.0 * (qw * qy - qz * qx);
    pitch_sin = std::clamp(pitch_sin, -1.0, 1.0);
    double pitch = std::asin(pitch_sin);
    double yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
    
    return Eigen::Vector3d(yaw * 180.0 / M_PI, roll * 180.0 / M_PI, pitch * 180.0 / M_PI);
}

void UKF::setInitialOrientation(const Eigen::Quaterniond& q) {
    Eigen::Quaterniond q_normalized = q.normalized();
    Eigen::Vector3d rot_vec = quaternionToRotationVector(q_normalized);
    Xk.head<3>() = rot_vec;
}

void UKF::setInitialOrientation(const Eigen::Vector3d& euler_zxy_deg) {
    double yaw_rad = euler_zxy_deg(0) * M_PI / 180.0;
    double roll_rad = euler_zxy_deg(1) * M_PI / 180.0;
    double pitch_rad = euler_zxy_deg(2) * M_PI / 180.0;
    
    Eigen::Quaterniond q;
    double cy = std::cos(yaw_rad * 0.5), sy = std::sin(yaw_rad * 0.5);
    double cr = std::cos(roll_rad * 0.5), sr = std::sin(roll_rad * 0.5);
    double cp = std::cos(pitch_rad * 0.5), sp = std::sin(pitch_rad * 0.5);
    
    q.w() = cy * cr * cp + sy * sr * sp;
    q.x() = cy * sr * cp - sy * cr * sp;
    q.y() = cy * cr * sp + sy * sr * cp;
    q.z() = sy * cr * cp - cy * sr * sp;
    q.normalize();
    
    Eigen::Vector3d rot_vec = quaternionToRotationVector(q);
    Xk.head<3>() = rot_vec;
}

void UKF::setAngularVelocity(const Eigen::Vector3d& omega) {
    Xk.tail<3>() = omega;
}

// ========== NEW: Initialize with first gyro measurement (match Python) ==========
void UKF::setInitialState(const Eigen::Vector3d& euler_zxy_deg, const Eigen::Vector3d& omega_xyz) {
    // Set orientation
    setInitialOrientation(euler_zxy_deg);
    
    // Set initial angular velocity (matches Python line 630)
    // Python uses: omega[0,0], omega[1,0], omega[2,0] (first measurement)
    Xk.tail<3>() = omega_xyz;
}
// ================================================================================

void UKF::measurementMag(const Eigen::Vector3d &z_mag) {
    int n = 6;
    int n_sigma = 2 * n;
    
    Eigen::MatrixXd Z_sigma(n_sigma, 3);
    for(int i = 0; i < n_sigma; i++) {
        Eigen::Quaterniond qi(sigma_points_Y(i,0), sigma_points_Y(i,1),
                             sigma_points_Y(i,2), sigma_points_Y(i,3));
        qi.normalize();
        
        Eigen::Quaterniond m_quat(0, mag_global.x(), mag_global.y(), mag_global.z());
        Eigen::Quaterniond m_body_quat = qi * m_quat * qi.inverse();
        
        Z_sigma.row(i) << m_body_quat.x(), m_body_quat.y(), m_body_quat.z();
    }
    
    Eigen::Vector3d z_pred = Z_sigma.colwise().mean();
    
    Eigen::Matrix3d Pzz = Eigen::Matrix3d::Zero();
    for(int i = 0; i < n_sigma; i++) {
        Eigen::Vector3d diff = Z_sigma.row(i).transpose() - z_pred;
        Pzz += diff * diff.transpose();
    }
    Pzz /= n_sigma;
    
    Eigen::Matrix3d Pvv = Pzz + R_mag;
    Pvv = makePositiveDefinite(Pvv, 1e-6);
    
    Eigen::MatrixXd Pxy = Eigen::MatrixXd::Zero(6, 3);
    for(int i = 0; i < n_sigma; i++) {
        Eigen::Matrix<double, 6, 1> w_diff = Wprime.row(i).transpose();
        Eigen::Vector3d z_diff = Z_sigma.row(i).transpose() - z_pred;
        Pxy += w_diff * z_diff.transpose();
    }
    Pxy /= n_sigma;
    
    Eigen::Vector3d vk = z_mag - z_pred;
    update(Pxy, Pvv, vk);
}

// ========== NEW: Diagnostics ==========
void UKF::printDiagnostics() const {
    std::cout << "\n=== UKF Diagnostics ===" << std::endl;
    std::cout << "Total iterations: " << iteration_count << std::endl;
    std::cout << "Cholesky failures: " << cholesky_failures << std::endl;
    if (iteration_count > 0) {
        double failure_rate = 100.0 * cholesky_failures / iteration_count;
        std::cout << "Failure rate: " << failure_rate << "%" << std::endl;
    }
    std::cout << "Current angular velocity: " << Xk.tail<3>().transpose() << " rad/s" << std::endl;
    std::cout << "Current orientation: " << getEulerZXY().transpose() << " deg" << std::endl;
}

void UKF::resetDiagnostics() {
    iteration_count = 0;
    cholesky_failures = 0;
}
// =====================================