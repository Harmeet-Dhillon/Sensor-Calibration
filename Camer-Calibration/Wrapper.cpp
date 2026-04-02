/**
 * Wrapper.cpp
 * Camera Calibration using Zhang's Method
 * C++ port of Wrapper.py
 *
 * Dependencies: OpenCV 4.x, C++17
 * Build:
 *   g++ -std=c++17 -O2 Wrapper.cpp -o Wrapper \
 *       $(pkg-config --cflags --libs opencv4)
 */

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

struct ProjectionResult {
    std::vector<double>                          error;          // length-7 residual vector (total_error, 0, 0, ...)
    std::vector<std::vector<cv::Point3d>>        reprojected;    // [image_i][corner_j] = (u_hat, v_hat, 1)
    std::vector<double>                          imgwiseErrors;  // per-image mean error
};

// ─────────────────────────────────────────────────────────────────────────────
// 1. get_gray_images
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Loads colour images from disk and produces corresponding grayscale copies.
 *
 * @param files     Sorted list of filenames (basename only).
 * @param directory Path to the image directory.
 * @param images    [out] Colour images.
 * @param grayImages [out] Grayscale images.
 */
void getGrayImages(const std::vector<std::string>& files,
                   const std::string&               directory,
                   std::vector<cv::Mat>&            images,
                   std::vector<cv::Mat>&            grayImages)
{
    for (const auto& file : files) {
        std::string path = directory + "/" + file;
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "[WARN] Could not read image: " << path << "\n";
            continue;
        }
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        images.push_back(img);
        grayImages.push_back(gray);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. get_world_coords
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Generates 2-D world coordinates for every inner corner of the checkerboard,
 * replicating np.meshgrid(range(length), range(width)) * square_size.
 *
 * @param checkerboardSize  (cols, rows) of inner corners.
 * @param squareSize        Physical size of one square in mm (or any unit).
 * @return                  Mat of shape (cols*rows, 2), type CV_64F.
 */
cv::Mat getWorldCoords(cv::Size checkerboardSize, double squareSize)
{
    const int cols = checkerboardSize.width;   // "length" in Python
    const int rows = checkerboardSize.height;  // "width"  in Python
    const int N    = cols * rows;

    cv::Mat worldCoords(N, 2, CV_64F);

    // meshgrid: outer loop over y (rows), inner loop over x (cols)
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int idx = y * cols + x;
            worldCoords.at<double>(idx, 0) = x * squareSize;
            worldCoords.at<double>(idx, 1) = y * squareSize;
        }
    }
    return worldCoords;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. get_checkerboard_corners
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Detects inner checkerboard corners.  Saves annotated images to
 * Results/Corners/<filename>.
 *
 * @return Vector of Mats, each (N, 2) CV_64F containing (u, v) pixel coords.
 */
std::vector<cv::Mat> getCheckerboardCorners(const std::vector<cv::Mat>& grayImages,
                                             cv::Size                    checkerboardSize,
                                             const std::vector<std::string>& files)
{
    const std::string resultsDir = "Results/Corners";
    fs::create_directories(resultsDir);

    std::vector<cv::Mat> cornersList;

    for (size_t i = 0; i < grayImages.size(); ++i) {
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(grayImages[i], checkerboardSize, corners);

        if (found) {
            // Convert to (N, 2) double matrix — mirrors np.squeeze(corner)
            cv::Mat cornerMat(static_cast<int>(corners.size()), 2, CV_64F);
            for (size_t j = 0; j < corners.size(); ++j) {
                cornerMat.at<double>(static_cast<int>(j), 0) = corners[j].x;
                cornerMat.at<double>(static_cast<int>(j), 1) = corners[j].y;
            }

            // Save annotated image
            cv::Mat colourCopy;
            cv::cvtColor(grayImages[i], colourCopy, cv::COLOR_GRAY2BGR);
            cv::drawChessboardCorners(colourCopy, checkerboardSize, corners, found);
            cv::imwrite(resultsDir + "/" + files[i], colourCopy);

            cornersList.push_back(cornerMat);
        }
    }
    return cornersList;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. get_homography
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Computes the 3×3 homography H that maps world_coords → corners via SVD.
 * Normalises so that H(2,2) == 1.
 *
 * @param worldCoords  (N, 2) CV_64F
 * @param corners      (N, 2) CV_64F
 */
cv::Mat getHomography(const cv::Mat& worldCoords, const cv::Mat& corners)
{
    if (worldCoords.size() != corners.size())
        throw std::invalid_argument("getHomography: world and image coord shapes must match.");

    const int N = worldCoords.rows;
    cv::Mat A(2 * N, 9, CV_64F, cv::Scalar(0.0));

    for (int i = 0; i < N; ++i) {
        const double X1 = worldCoords.at<double>(i, 0);
        const double Y1 = worldCoords.at<double>(i, 1);
        const double x2 = corners.at<double>(i, 0);
        const double y2 = corners.at<double>(i, 1);

        // Row 2i
        A.at<double>(2*i, 0) = -X1;
        A.at<double>(2*i, 1) = -Y1;
        A.at<double>(2*i, 2) = -1.0;
        A.at<double>(2*i, 6) =  x2 * X1;
        A.at<double>(2*i, 7) =  x2 * Y1;
        A.at<double>(2*i, 8) =  x2;

        // Row 2i+1
        A.at<double>(2*i+1, 3) = -X1;
        A.at<double>(2*i+1, 4) = -Y1;
        A.at<double>(2*i+1, 5) = -1.0;
        A.at<double>(2*i+1, 6) =  y2 * X1;
        A.at<double>(2*i+1, 7) =  y2 * Y1;
        A.at<double>(2*i+1, 8) =  y2;
    }

    cv::Mat U, S, Vt;
    cv::SVD::compute(A, S, U, Vt);

    // Last row of Vt is the right singular vector for the smallest singular value
    cv::Mat h = Vt.row(Vt.rows - 1).clone();
    cv::Mat H = h.reshape(0, 3);  // 3×3

    // Normalise so H(2,2) == 1
    double h33 = H.at<double>(2, 2);
    H /= h33;
    return H;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. calculate_v  (Zhang 1999 eq.)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Returns the 1×6 row vector v_ij as defined in Zhang's paper.
 * i and j are 0-based column indices of H.
 */
cv::Mat calculateV(const cv::Mat& H, int i, int j)
{
    cv::Mat vij(1, 6, CV_64F);
    vij.at<double>(0, 0) = H.at<double>(0, i) * H.at<double>(0, j);
    vij.at<double>(0, 1) = H.at<double>(0, i) * H.at<double>(1, j)
                         + H.at<double>(1, i) * H.at<double>(0, j);
    vij.at<double>(0, 2) = H.at<double>(1, i) * H.at<double>(1, j);
    vij.at<double>(0, 3) = H.at<double>(2, i) * H.at<double>(0, j)
                         + H.at<double>(0, i) * H.at<double>(2, j);
    vij.at<double>(0, 4) = H.at<double>(2, i) * H.at<double>(1, j)
                         + H.at<double>(1, i) * H.at<double>(2, j);
    vij.at<double>(0, 5) = H.at<double>(2, i) * H.at<double>(2, j);
    return vij;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. calculate_b
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Stacks the v12 and (v11 − v22) rows for every homography and solves for b
 * using SVD (last right singular vector).
 *
 * @return  b as a (6, 1) CV_64F column vector.
 */
cv::Mat calculateB(const std::vector<cv::Mat>& hMatrixList)
{
    const int n = static_cast<int>(hMatrixList.size());
    cv::Mat Vij(2 * n, 6, CV_64F);

    for (int idx = 0; idx < n; ++idx) {
        const cv::Mat& H = hMatrixList[idx];
        cv::Mat v11 = calculateV(H, 0, 0);
        cv::Mat v12 = calculateV(H, 0, 1);
        cv::Mat v22 = calculateV(H, 1, 1);

        v12.copyTo(Vij.row(2 * idx));
        (v11 - v22).copyTo(Vij.row(2 * idx + 1));
    }

    cv::Mat U, S, Vt;
    cv::SVD::compute(Vij, S, U, Vt, cv::SVD::FULL_UV);

    // Last row of Vt → column vector of length 6
    cv::Mat b = Vt.row(Vt.rows - 1).clone().reshape(0, 6);
    return b;
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. get_intrinsic_matrix
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Recovers the 3×3 intrinsic matrix K from the b vector.
 */
cv::Mat getIntrinsicMatrix(const cv::Mat& b)
{
    const double b0 = b.at<double>(0);
    const double b1 = b.at<double>(1);
    const double b2 = b.at<double>(2);
    const double b3 = b.at<double>(3);
    const double b4 = b.at<double>(4);
    const double b5 = b.at<double>(5);

    const double vo     = (b1 * b3 - b0 * b4) / (b0 * b2 - b1 * b1);
    const double lambda = b5 - (b3 * b3 + vo * (b1 * b3 - b0 * b4)) / b0;
    const double alpha  = std::sqrt(lambda / b0);
    const double beta   = std::sqrt((lambda * b0) / (b0 * b2 - b1 * b1));
    const double skew   = -b1 * (alpha * alpha) * beta / lambda;
    const double uo     = (skew * vo / beta) - (b3 * (alpha * alpha)) / lambda;

    cv::Mat A = (cv::Mat_<double>(3, 3) <<
        alpha, skew, uo,
        0.0,   beta, vo,
        0.0,   0.0,  1.0);
    return A;
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. get_extrinsic_matrix
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Computes the 3×3 [r1 | r2 | t] extrinsic matrix for every homography.
 */
std::vector<cv::Mat> getExtrinsicMatrices(const cv::Mat&              A,
                                           const std::vector<cv::Mat>& hMatrixList)
{
    std::vector<cv::Mat> extrinsics;
    cv::Mat A_inv = A.inv();

    for (const auto& H : hMatrixList) {
        cv::Mat h1 = H.col(0).clone();
        cv::Mat h2 = H.col(1).clone();
        cv::Mat h3 = H.col(2).clone();

        double scale = 1.0 / cv::norm(A_inv * h1);

        cv::Mat r1 = scale * (A_inv * h1);
        cv::Mat r2 = scale * (A_inv * h2);
        cv::Mat t  = scale * (A_inv * h3);

        // Stack as [r1 | r2 | t] → 3×3
        cv::Mat ext(3, 3, CV_64F);
        r1.copyTo(ext.col(0));
        r2.copyTo(ext.col(1));
        t .copyTo(ext.col(2));

        extrinsics.push_back(ext);
    }
    return extrinsics;
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. projection_error
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Computes reprojection error with radial distortion (k1, k2).
 * The error vector is padded to length 7 (same as x0) to match
 * scipy.optimize.least_squares' residual convention.
 *
 * @param x0  [alpha, skew, beta, u0, v0, k1, k2]
 */
ProjectionResult projectionError(const std::vector<double>&   x0,
                                  const std::vector<cv::Mat>& R,
                                  const cv::Mat&              worldCoords,
                                  const std::vector<cv::Mat>& imgCorners)
{
    const double alpha = x0[0], skew = x0[1], beta = x0[2];
    const double u0    = x0[3], v0   = x0[4];
    const double k1    = x0[5], k2   = x0[6];

    cv::Mat A = (cv::Mat_<double>(3, 3) <<
        alpha, skew, u0,
        0.0,   beta, v0,
        0.0,   0.0,  1.0);

    ProjectionResult result;
    double totalError = 0.0;
    const int numImages = static_cast<int>(imgCorners.size());

    for (int i = 0; i < numImages; ++i) {
        const cv::Mat& corners    = imgCorners[i];
        const cv::Mat& extMatrix  = R[i];
        cv::Mat finalTransform    = A * extMatrix;

        double currError = 0.0;
        const int numCorners = corners.rows;
        std::vector<cv::Point3d> estCorners;

        for (int iter = 0; iter < numCorners; ++iter) {
            const double cx = corners.at<double>(iter, 0);
            const double cy = corners.at<double>(iter, 1);

            // Measured corner in homogeneous coords
            cv::Mat measuredHomo = (cv::Mat_<double>(3, 1) << cx, cy, 1.0);

            // World point in homogeneous coords
            cv::Mat worldPt = (cv::Mat_<double>(3, 1) <<
                worldCoords.at<double>(iter, 0),
                worldCoords.at<double>(iter, 1),
                1.0);

            // Camera-frame normalised coords
            cv::Mat camPt = extMatrix * worldPt;
            const double xc = camPt.at<double>(0) / camPt.at<double>(2);
            const double yc = camPt.at<double>(1) / camPt.at<double>(2);

            // Pixel coords (before distortion)
            cv::Mat pixPt = finalTransform * worldPt;
            const double u = pixPt.at<double>(0) / pixPt.at<double>(2);
            const double v = pixPt.at<double>(1) / pixPt.at<double>(2);

            // Radial distortion correction
            const double r2 = xc*xc + yc*yc;
            const double distortion = k1 * r2 + k2 * r2 * r2;
            const double u_hat = u + (u - u0) * distortion;
            const double v_hat = v + (v - v0) * distortion;

            // Homogeneous estimated corner
            cv::Mat estHomo = (cv::Mat_<double>(3, 1) << u_hat, v_hat, 1.0);

            // L2 norm between estimated and measured (homogeneous, matching Python)
            currError += cv::norm(estHomo - measuredHomo, cv::NORM_L2);

            estCorners.emplace_back(u_hat, v_hat, 1.0);
        }

        currError /= numCorners;
        result.imgwiseErrors.push_back(currError);
        totalError += currError / numImages;
        result.reprojected.push_back(estCorners);
    }

    // Pad error vector to length 7 (mirrors Python's while-loop padding)
    result.error.assign(x0.size(), 0.0);
    result.error[0] = totalError;

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Levenberg–Marquardt optimiser (numerical Jacobian)
//     Replaces scipy.optimize.least_squares(..., method="lm")
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Minimises  sum_i f_i(x)^2  using the Levenberg–Marquardt algorithm with a
 * central-difference numerical Jacobian.
 *
 * @param fn      Residual function f(x) → vector of residuals.
 * @param x0      Initial parameter vector.
 * @param maxIter Maximum iterations.
 * @param gtol    Gradient norm tolerance for convergence.
 * @param xtol    Step norm tolerance for convergence.
 * @return        Optimised parameter vector.
 */
std::vector<double> levenbergMarquardt(
    std::function<std::vector<double>(const std::vector<double>&)> fn,
    std::vector<double> x0,
    int    maxIter = 500,
    double gtol    = 1e-10,
    double xtol    = 1e-10)
{
    const int n = static_cast<int>(x0.size());
    std::vector<double> x = x0;
    double lambda = 1e-3;

    auto toMat = [](const std::vector<double>& v) {
        cv::Mat m(static_cast<int>(v.size()), 1, CV_64F);
        for (int i = 0; i < static_cast<int>(v.size()); ++i)
            m.at<double>(i) = v[i];
        return m;
    };

    std::vector<double> f = fn(x);
    int m = static_cast<int>(f.size());

    for (int iter = 0; iter < maxIter; ++iter) {
        // ── Numerical Jacobian (central differences) ────────────────────────
        cv::Mat J(m, n, CV_64F);
        for (int j = 0; j < n; ++j) {
            double h = std::max(1e-7, std::abs(x[j]) * 1e-5);

            std::vector<double> xp = x, xm = x;
            xp[j] += h;
            xm[j] -= h;

            std::vector<double> fp = fn(xp);
            std::vector<double> fm = fn(xm);

            for (int i = 0; i < m; ++i)
                J.at<double>(i, j) = (fp[i] - fm[i]) / (2.0 * h);
        }

        cv::Mat fMat = toMat(f);
        cv::Mat JT   = J.t();
        cv::Mat JTJ  = JT * J;
        cv::Mat JTf  = JT * fMat;

        // Check gradient norm for convergence
        if (cv::norm(JTf, cv::NORM_L2) < gtol) {
            std::cout << "[LM] Converged (gradient) at iter " << iter << "\n";
            break;
        }

        // ── Solve (J^T J + lambda * diag(J^T J)) dx = -J^T f ───────────────
        cv::Mat D = cv::Mat::zeros(n, n, CV_64F);
        for (int j = 0; j < n; ++j)
            D.at<double>(j, j) = JTJ.at<double>(j, j);  // diagonal scaling

        cv::Mat lhs = JTJ + lambda * D;
        cv::Mat dx;
        cv::solve(lhs, -JTf, dx, cv::DECOMP_SVD);

        // ── Trial step ───────────────────────────────────────────────────────
        std::vector<double> xNew(n);
        for (int j = 0; j < n; ++j)
            xNew[j] = x[j] + dx.at<double>(j);

        std::vector<double> fNew = fn(xNew);

        double costOld = 0.0, costNew = 0.0;
        for (double v : f)    costOld += v * v;
        for (double v : fNew) costNew += v * v;

        if (costNew < costOld) {
            x = xNew;
            f = fNew;
            lambda = std::max(lambda * 0.1, 1e-15);

            if (cv::norm(dx, cv::NORM_L2) < xtol) {
                std::cout << "[LM] Converged (step) at iter " << iter << "\n";
                break;
            }
        } else {
            lambda = std::min(lambda * 10.0, 1e12);
        }

        if (iter % 50 == 0)
            std::cout << "[LM] iter " << iter << "  cost=" << costOld << "  lambda=" << lambda << "\n";
    }

    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: sort filenames by trailing integer (e.g. img_3.jpg < img_12.jpg)
// Mirrors:  files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
// ─────────────────────────────────────────────────────────────────────────────
static int trailingInt(const std::string& name)
{
    // Find last '_' then take digits up to '.'
    size_t us  = name.rfind('_');
    size_t dot = name.rfind('.');
    if (us == std::string::npos || dot == std::string::npos || dot <= us)
        return 0;
    try { return std::stoi(name.substr(us + 1, dot - us - 1)); }
    catch (...) { return 0; }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    const std::string directory = "./Data/Calibration_Imgs";

    // ── 1. Collect and sort .jpg files ───────────────────────────────────────
    std::vector<std::string> files;
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.path().extension() == ".jpg")
                files.push_back(entry.path().filename().string());
        }
        std::sort(files.begin(), files.end(),
                  [](const std::string& a, const std::string& b) {
                      return trailingInt(a) < trailingInt(b);
                  });
    } catch (const std::exception& e) {
        std::cerr << "Error occurred: " << e.what() << "\n";
        return 1;
    }

    // ── 2. Load images ───────────────────────────────────────────────────────
    std::vector<cv::Mat> images, grayImages;
    getGrayImages(files, directory, images, grayImages);

    // ── 3. World coordinates (9×6 checkerboard, 21.5 mm squares) ────────────
    const cv::Size checkerboardSize(9, 6);
    const double   squareSize = 21.5;
    cv::Mat worldCoords = getWorldCoords(checkerboardSize, squareSize);

    // ── 4. Detect corners ────────────────────────────────────────────────────
    std::vector<cv::Mat> imgCorners =
        getCheckerboardCorners(grayImages, checkerboardSize, files);

    std::cout << "Images with corners detected: " << imgCorners.size()
              << "  corners per image: " << (imgCorners.empty() ? 0 : imgCorners[0].rows)
              << "\n";

    // ── 5. Homographies ──────────────────────────────────────────────────────
    std::vector<cv::Mat> hMatrixList;
    for (const auto& corner : imgCorners)
        hMatrixList.push_back(getHomography(worldCoords, corner));

    // ── 6. b vector ──────────────────────────────────────────────────────────
    cv::Mat b = calculateB(hMatrixList);
    std::cout << "b: " << b.t() << "\n";

    // ── 7. Intrinsic matrix ──────────────────────────────────────────────────
    cv::Mat A = getIntrinsicMatrix(b);
    std::cout << "Initial Intrinsic Matrix:\n" << A << "\n\n";

    // ── 8. Extrinsic matrices ────────────────────────────────────────────────
    std::vector<cv::Mat> R = getExtrinsicMatrices(A, hMatrixList);

    // ── 9. Initial optimisation params: [alpha, skew, beta, u0, v0, k1, k2] ─
    std::vector<double> optParams = {
        A.at<double>(0, 0),  // alpha
        A.at<double>(0, 1),  // skew
        A.at<double>(1, 1),  // beta
        A.at<double>(0, 2),  // u0
        A.at<double>(1, 2),  // v0
        0.0,                 // k1
        0.0                  // k2
    };

    std::cout << "Initial Optimisation Parameters: [";
    for (size_t i = 0; i < optParams.size(); ++i)
        std::cout << optParams[i] << (i + 1 < optParams.size() ? ", " : "");
    std::cout << "]\n\n";

    // ── 10. Pre-optimisation error ───────────────────────────────────────────
    ProjectionResult beforeOpt = projectionError(optParams, R, worldCoords, imgCorners);
    std::cout << "Before optimisation total error: " << beforeOpt.error[0] << "\n";
    std::cout << "Before optimisation image-wise errors: [";
    for (size_t i = 0; i < beforeOpt.imgwiseErrors.size(); ++i)
        std::cout << beforeOpt.imgwiseErrors[i]
                  << (i + 1 < beforeOpt.imgwiseErrors.size() ? ", " : "");
    std::cout << "]\n\n";

    // ── 11. Levenberg–Marquardt optimisation ─────────────────────────────────
    // The objective function returns the same padded residual vector as Python.
    auto objectiveFn = [&](const std::vector<double>& x) -> std::vector<double> {
        return projectionError(x, R, worldCoords, imgCorners).error;
    };

    std::cout << "Running Levenberg-Marquardt optimisation...\n";
    std::vector<double> optResult = levenbergMarquardt(objectiveFn, optParams);

    const double alphaOpt = optResult[0];
    const double skewOpt  = optResult[1];
    const double betaOpt  = optResult[2];
    const double u0Opt    = optResult[3];
    const double v0Opt    = optResult[4];
    const double k1       = optResult[5];
    const double k2       = optResult[6];

    cv::Mat A_optimized = (cv::Mat_<double>(3, 3) <<
        alphaOpt, skewOpt, u0Opt,
        0.0,      betaOpt, v0Opt,
        0.0,      0.0,     1.0);

    std::cout << "\nOptimized Intrinsic Matrix:\n" << A_optimized << "\n";
    std::cout << "Optimized distortion coefficients: k1=" << k1 << "  k2=" << k2 << "\n\n";

    // ── 12. Post-optimisation error ──────────────────────────────────────────
    ProjectionResult finalResult = projectionError(optResult, R, worldCoords, imgCorners);
    std::cout << "Final total error: " << finalResult.error[0] << "\n";
    std::cout << "Final image-wise errors: [";
    for (size_t i = 0; i < finalResult.imgwiseErrors.size(); ++i)
        std::cout << finalResult.imgwiseErrors[i]
                  << (i + 1 < finalResult.imgwiseErrors.size() ? ", " : "");
    std::cout << "]\n\n";

    // ── 13. Save undistorted images with reprojected corners ─────────────────
    const std::string finalDir = "Results/Final_Corners";
    fs::create_directories(finalDir);

    // Build OpenCV-style distortion vector [k1, k2, 0, 0, 0]
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, 0.0, 0.0, 0.0);

    for (size_t i = 0; i < images.size() && i < finalResult.reprojected.size(); ++i) {
        cv::Mat undistorted;
        cv::undistort(images[i], undistorted, A_optimized, distCoeffs);

        for (const auto& pt : finalResult.reprojected[i]) {
            int px = static_cast<int>(pt.x);
            int py = static_cast<int>(pt.y);
            cv::circle(undistorted, {px, py}, 10, {255, 0, 0},  4);
            cv::circle(undistorted, {px, py},  6, {0,   0, 255}, cv::FILLED);
        }

        cv::imwrite(finalDir + "/" + files[i], undistorted);
    }

    std::cout << "Saved undistorted images with reprojected corners to " << finalDir << "\n";
    return 0;
}
