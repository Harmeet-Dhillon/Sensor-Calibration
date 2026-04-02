# AutoCalib — Zhang's Camera Calibration in C++

<img width="360" height="634" alt="Screenshot from 2026-04-01 22-39-59" src="https://github.com/user-attachments/assets/47d05e42-db87-4910-b975-ecd2401b5742" />


> A from-scratch C++ implementation of **Zhengyou Zhang's** checkerboard-based camera calibration method. Estimates the full intrinsic matrix **K**, radial distortion coefficients **k₁/k₂**, and per-image extrinsics using closed-form linear algebra followed by non-linear refinement via a custom Levenberg–Marquardt optimizer — without ever calling `cv::calibrateCamera`.

---

## Table of Contents

- [Overview](#overview)
- [Theory](#theory)
  - [Homography Estimation](#1-homography-estimation)
  - [Intrinsic Matrix K](#2-solving-for-the-intrinsic-matrix-k)
  - [Extrinsic Parameters](#3-extrinsic-parameters)
  - [Radial Distortion](#4-radial-distortion-model)
  - [Non-linear Refinement](#5-non-linear-refinement)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Build Instructions](#build-instructions)
- [Usage](#usage)
- [Calibration Target](#calibration-target)
- [Implementation Notes](#implementation-notes)
- [References](#references)
- [License](#license)

---

## Overview

Camera calibration — recovering the focal length, principal point, and lens distortion of a camera — is a foundational step in any 3D computer vision pipeline. This project implements **Zhang's 1999 planar calibration method** end-to-end in C++17, going through every stage of the math manually.

The full pipeline:

```
13 checkerboard images  (Google Pixel XL)
          │
          ▼
 Corner Detection  ──  cv::findChessboardCorners
          │
          ▼
 Per-image Homography  ──  SVD on (2N × 9) system
          │
          ▼
 Closed-form K  ──  Zhang's V-matrix / SVD on (2N × 6) system
          │
          ▼
 Initial Extrinsics  ──  [r₁ | r₂ | t] from K⁻¹ H
          │
          ▼
 Levenberg–Marquardt Refinement  ──  numerical Jacobian, diagonal scaling
          │
          ▼
 Optimised K  +  k₁, k₂  +  undistorted output images
```

---

## Theory

### 1. Homography Estimation

For each calibration image a 3×3 homography **H** maps world plane coordinates **(X, Y)** to image pixels **(u, v)**. Expanding the constraint `[u, v, 1]ᵀ ∝ H [X, Y, 1]ᵀ` gives two linear equations per point:

```
[ -X  -Y  -1   0   0   0   u·X   u·Y   u ]
[  0   0   0  -X  -Y  -1   v·X   v·Y   v ]  ·  h  =  0
```

Stacking all N corner pairs produces a **(2N × 9)** matrix **A**. The solution **h** is the right singular vector of **A** for the smallest singular value (last row of **Vᵀ** from full SVD). The result is reshaped to 3×3 and normalised so H(2,2) = 1.

### 2. Solving for the Intrinsic Matrix K

```
K = [ α   γ   u₀ ]
    [ 0   β   v₀ ]
    [ 0   0    1 ]
```

where α, β are focal lengths in pixels, (u₀, v₀) is the principal point, and γ is the axis skew (typically near zero).

The symmetric matrix **B = K⁻ᵀ K⁻¹** satisfies the homography orthonormality constraints. Each image contributes two rows to a **(2N × 6)** system **Vb = 0**, where **b** encodes the 6 independent entries of **B**. Solving via SVD gives **b**, then **K** is extracted analytically:

```
v₀     = (b₁b₃ − b₀b₄) / (b₀b₂ − b₁²)
λ      = b₅ − (b₃² + v₀(b₁b₃ − b₀b₄)) / b₀
α      = √(λ / b₀)
β      = √(λ b₀ / (b₀b₂ − b₁²))
γ      = −b₁ α² β / λ
u₀     = γ v₀/β − b₃ α²/λ
```

### 3. Extrinsic Parameters

For image *i*, the rotation columns and translation are recovered from the corresponding homography:

```
λ  =  1 / ‖K⁻¹ h₁‖₂
r₁ =  λ K⁻¹ h₁
r₂ =  λ K⁻¹ h₂
t  =  λ K⁻¹ h₃
```

The extrinsic matrix is stored as **[r₁ | r₂ | t]** (3×3), which is sufficient since world points lie on the Z=0 plane.

### 4. Radial Distortion Model

Two-parameter radial distortion (Section 3.3 of Zhang 1999):

```
r²  =  xc² + yc²          (normalised camera-frame radius)

û   =  u  +  (u − u₀)(k₁ r² + k₂ r⁴)
v̂   =  v  +  (v − v₀)(k₁ r² + k₂ r⁴)
```

where (xc, yc) are the **normalised** camera-frame coordinates (before K multiplication). Initial values k₁ = k₂ = 0 are used as the linear estimate already accounts for the major distortion-free projection.

### 5. Non-linear Refinement

All parameters are jointly refined by minimising the total reprojection error:

$$\underset{K,\,k_1,\,k_2}{\text{argmin}} \sum_{i=1}^{N} \sum_{j=1}^{M} \left\| x_{ij} - \hat{x}_{ij}\!\left(K,\, R_i,\, t_i,\, X_j,\, k\right) \right\|_2$$

The 7-parameter vector `[α, γ, β, u₀, v₀, k₁, k₂]` is optimised with a **from-scratch Levenberg–Marquardt** solver using central-difference numerical Jacobians:

```
J[:,j]  =  (f(x + hₑⱼ) − f(x − hₑⱼ)) / 2h
```

The update step solves `(JᵀJ + λ·diag(JᵀJ)) Δx = −Jᵀf` using OpenCV SVD, with λ adapting as in the standard LM trust-region scheme.

---

## Results

Calibration was run on 13 images of a 9×6 inner-corner checkerboard (21.5 mm squares) captured with a Google Pixel XL.

### Recovered Intrinsic Matrix

```
K = [ 1678.24    0.82    953.17 ]
    [    0.00  1673.90   542.89 ]
    [    0.00    0.00      1.00 ]
```

### Distortion Coefficients

| k₁      | k₂     |
|---------|--------|
| −0.2296 | 0.1752 |

### Reprojection Error

| Stage              | Mean Error (px) |
|--------------------|-----------------|
| Before refinement  | ~1.8            |
| After refinement   | ~0.3            |

Corners detected on raw images are saved to `Results/Corners/`. After calibration, `Results/Final_Corners/` contains undistorted images with reprojected corners overlaid in blue (outer ring) and red (filled centre).

---

## Project Structure

```
AutoCalib/
│
├── Wrapper.cpp            # Full calibration pipeline
├── CMakeLists.txt         # CMake build config
├── README.md
├── .gitignore
│
├── Data/
│   └── Calibration_Imgs/  # ← Put downloaded .jpg files here (not tracked by git)
│       ├── Img_1.jpg
│       ├── Img_2.jpg
│       └── ...
│
└── Results/               # Generated at runtime (not tracked by git)
    ├── Corners/           # Corner-annotated raw images
    └── Final_Corners/     # Undistorted + reprojected corner images
```

---

## Dependencies

| Dependency | Minimum Version | Notes |
|------------|-----------------|-------|
| C++ compiler | GCC 8 / Clang 7 / MSVC 2019 | C++17 required (`std::filesystem`) |
| CMake | 3.16 | |
| OpenCV | 4.0 | Modules: core, calib3d, imgproc, imgcodecs |

### Install OpenCV

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install -y libopencv-dev
```

**macOS (Homebrew)**
```bash
brew install opencv
```

**Windows (vcpkg)**
```bash
vcpkg install opencv4:x64-windows
cmake .. -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
```

---

## Build Instructions

```bash
# 1. Clone
git clone https://github.com/<your-username>/AutoCalib.git
cd AutoCalib

# 2. Configure & build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

# Binary location: build/AutoCalib  (or build/AutoCalib.exe on Windows)
```

> **GCC < 9 note:** `std::filesystem` requires linking `-lstdc++fs`.  
> The CMakeLists.txt handles this automatically.

---

## Usage

### Step 1 — Download the calibration images

The 13 checkerboard images were captured with a Google Pixel XL with focus locked.

**[⬇ Download Calibration Images (Box)](https://app.box.com/s/z18zwbs11609z988x0kod3o1058rdop5)**

Place all `.jpg` files in `Data/Calibration_Imgs/`:

```bash
mkdir -p Data/Calibration_Imgs
# copy/move your downloaded images here
```

### Step 2 — Run

```bash
# From the project root directory:
./build/AutoCalib
```

Images are expected to follow the naming convention `*_<number>.jpg`  
(e.g., `Img_1.jpg`, `Img_2.jpg`, …) so they are sorted in the correct order.

### Step 3 — Inspect outputs

| Output directory | Contents |
|-----------------|----------|
| `Results/Corners/` | Detected checkerboard corners drawn on raw grayscale images |
| `Results/Final_Corners/` | Undistorted colour images with reprojected corners overlaid |

### Expected console output

```
Images with corners detected: 13  corners per image: 54
b: [b0, b1, b2, b3, b4, b5]

Initial Intrinsic Matrix:
[1651.3,   0.6, 951.2;
    0.0, 1647.8, 541.3;
    0.0,   0.0,    1.0]

Initial Optimisation Parameters: [1651.3, 0.6, 1647.8, 951.2, 541.3, 0.0, 0.0]

Before optimisation total error: 1.823
[LM] iter   0   cost=3.324   lambda=0.001
[LM] iter  50   cost=0.091   lambda=1e-07
[LM] Converged (step) at iter 138

Optimized Intrinsic Matrix:
[1678.2,  0.82, 953.2;
    0.0, 1673.9, 542.9;
    0.0,    0.0,   1.0]
Optimized distortion coefficients: k1=-0.2296  k2=0.1752

Final total error: 0.291
Saved undistorted images with reprojected corners to Results/Final_Corners
```

---

## Calibration Target

The pattern used is a **9×6 inner-corner** grid (outermost border squares excluded) printed on A4 paper with **21.5 mm per square**.

[⬇ Download checkerboard PDF](https://github.com/cmsc733/cmsc733.github.io/raw/master/assets/2019/hw1/checkerboardPattern.pdf)

When capturing your own calibration images:
- Keep the checkerboard flat and rigidly attached to a hard surface.
- Vary the angle and position significantly across images — don't just translate.
- Aim for 10–20 images covering the full field of view.
- Lock focus if your camera allows it.

---

## Implementation Notes

- **No `cv::calibrateCamera`** anywhere. Every algebraic step is manual.
- All linear systems are solved via OpenCV's `cv::SVD::compute` (full SVD).
- The LM optimizer damps on `diag(JᵀJ)` rather than the identity matrix, matching the behaviour of `scipy.optimize.least_squares(method='lm')`.
- Distortion is applied in the **normalised camera frame** (coordinates before K multiplication), which is the formulation in Zhang's paper and ensures physically correct correction.
- The extrinsic matrix is stored as `[r₁ | r₂ | t]` (3×3) rather than a full `[R | t]` (3×4), since the world points are on the Z=0 plane and r₃ = r₁ × r₂ is never needed explicitly.
- Images are sorted by the trailing integer in their filename (e.g., `Img_3.jpg` before `Img_12.jpg`) to guarantee a consistent processing order.

---

## References

1. **Zhang, Z.** (1999). *A Flexible New Technique for Camera Calibration*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11), 1330–1334.  
   [Paper PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)

2. **OpenCV** — Camera Calibration and 3D Reconstruction:  
   [https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
