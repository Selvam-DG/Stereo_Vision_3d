# Stereo Vision (3D) System for Bin-Picking

A stereo vision system that computes 3D coordinates of parts inside a bin, so a robot arm can
locate and grasp objects that are non-oriented, jumbled and partially occluded.

Master's Project (12 ECTS) — Technische Hochschule Rosenheim, Feb–Jul 2024.
Examiner: Prof. Dr.-Ing. Michael Wagner.

📄 Full report: [Master_Project_Report.pdf](Master_Project_Report.pdf)

---

## The problem

Bin-picking is hard because parts arrive unsorted: overlapping, interlocked and often hidden
behind each other. A robot cannot pick what it cannot locate in 3D. This project builds the
perception half of that problem — from two synchronized cameras to XYZ coordinates for the
parts in the bin.

---

## Pipeline

```
Synchronized capture  ->  Undistort  ->  Rectify  ->  Stereo matching
                                                          |
                                          Disparity map ---+
                                                          |
                             Q matrix reprojection  ->  Depth  ->  XYZ coordinates
```

---

## Hardware

| Item | Detail |
|---|---|
| Cameras | 2 × Allied Vision Alvium G1-240m/c |
| Sensor | Sony IMX392 CMOS, global shutter |
| Resolution | 1936 × 1216 (2.4 MP) |
| Baseline | 150 mm, fixed |
| Mount | Custom-designed in CAD — see [`AutoCAD/`](AutoCAD) |
| Interface | 1000BASE-T over PoE |
| Control | GenICam via the `vmbpy` (Vimba) API |

Both cameras are software-triggered on a shared loop to keep frame capture synchronized.
Exposure and white balance run in continuous mode; GVSP packet size is auto-adjusted per
stream. A `Handler` class queues incoming frames and converts pixel formats for OpenCV.

---

## Calibration

Chessboard calibration (7 × 9 board) using `cv2.findChessboardCorners` with sub-pixel corner
refinement, then:

1. `cv2.calibrateCamera` — intrinsics and distortion coefficients per camera
2. `cv2.getOptimalNewCameraMatrix` — refined intrinsics with free scaling
3. `cv2.stereoCalibrate` — rotation, translation, essential and fundamental matrices, with
   intrinsics fixed (`CALIB_FIX_INTRINSIC`)
4. `cv2.stereoRectify` — rectification transforms, projection matrices and the
   disparity-to-depth matrix Q
5. Parameters saved as `.npy` files for reuse

Depth follows from reprojecting the disparity map through Q with `cv2.reprojectImageTo3D`,
giving XYZ per valid pixel.

Calibration output for the setup used in this project is committed under
[`pixel2WCos/calibParams/`](pixel2WCos/calibParams) — camera matrices, distortion
coefficients, rectification transforms, projection matrices and the Q matrix.

---

## Experiments

Five experiments, run in sequence, each addressing a limitation found in the previous one.
Figures and full discussion are in the report.

**1 — StereoBM on live captures.**
Built a GUI with sliders for `numDisparities`, `blockSize`, `preFilterCap`, `textureThreshold`,
`uniquenessRatio` and the speckle parameters, so their effect could be observed live.
Result: disparity maps were noisy and incomplete, especially in textureless regions. Larger
block sizes smoothed the map but blurred edges. Not usable as-is.

**2 — StereoSGBM on the same captures.**
Semi-global matching gave clearer contours and better object definition than StereoBM, but
noise persisted around edges and low-texture areas. Applying a weighted least squares (WLS)
filter was what made the output acceptable — post-processing turned out to matter as much as
the matching algorithm.

**3 — Both algorithms on the Tsukuba reference pair.**
Run to separate algorithm limitations from input-quality limitations. Both algorithms produced
good disparity maps on the standard dataset, which located the problem in the captured images
rather than in the matching code.

**4 — Structured light projection.**
A projector cast a pattern onto the scene to add texture where surfaces had none, with Gaussian
filtering applied before matching. Both algorithms improved substantially. This addressed the
root cause identified in experiment 3: textureless surfaces give block matching nothing to
correlate.

**5 — Synthetic validation in Blender.**
Stereo pairs rendered with known ground-truth disparity, to check the pipeline against a
reference. Disparity maps from synthetic input were the most accurate of the five experiments,
confirming the implementation was correct and that real-world accuracy was bounded by image
quality and calibration precision.

Output disparity maps are in [`DisparityResult/`](DisparityResult).

---

## What the experiments established

- StereoSGBM outperforms StereoBM on real captures, but neither is usable without
  post-processing on low-texture scenes.
- WLS filtering is not optional here — it is the step that makes the disparity map usable.
- Adding texture to the scene with a projector improves results more than tuning matching
  parameters does.
- Validating against both a public reference pair and synthetic ground truth was what
  distinguished implementation errors from input-quality limits.

Accuracy was assessed qualitatively across the test scenarios. Quantified error against ground
truth was not measured and is left for future work.

---

## Repository layout

```
.
├── Master_Project_Report.pdf      # full project report
├── Alvium-G1_User-Guide.pdf       # camera reference
├── AutoCAD/                       # camera mount design
├── DisparityResult/               # output disparity maps
├── pixel2WCos/                    # current pipeline
│   ├── caliberation.py            # single-camera calibration
│   ├── stereoCaliberate.py        # stereo calibration + rectification
│   ├── undistort.py               # undistortion using saved parameters
│   ├── calibParams/               # saved .npy calibration output
│   ├── TestImages/                # chessboard pairs (Cam0 / Cam1)
│   └── test/                      # scene image pairs
└── Python_Files/                  # earlier iterations, kept for reference
    ├── camSet.py, readCamera.py   # camera access and streaming (vmbpy)
    ├── camCalibration.py
    ├── stereoCalibration.py
    ├── ocvDisparity.py            # disparity GUI
    ├── stereoVision.py
    ├── depthCal.py
    ├── pixel2WCos.py
    └── Images/                    # calibration and test captures
```

> `Python_Files/` holds the working scripts from the development phase, in the order they were
> written. `pixel2WCos/` is the cleaned-up version of the pipeline and is the one to read first.

---

## Running the pipeline

Requires Python 3, OpenCV, NumPy, and — for live capture — the Allied Vision Vimba SDK with the
`vmbpy` bindings.

```bash
pip install opencv-contrib-python numpy
```

`opencv-contrib-python` rather than `opencv-python`: the WLS disparity filter lives in
`cv2.ximgproc`, which is contrib-only.

```bash
# 1. Calibrate each camera from chessboard captures
python pixel2WCos/caliberation.py

# 2. Stereo calibration and rectification; writes calibParams/*.npy
python pixel2WCos/stereoCaliberate.py

# 3. Undistort a stereo pair using the saved parameters
python pixel2WCos/undistort.py
```

Image paths are currently hardcoded in the scripts and will need adjusting for your setup.

---

## Stack

Python · OpenCV · NumPy · vmbpy (Allied Vision Vimba) · Blender (synthetic data) · AutoCAD
(camera mount)

---

## Future work

- Quantified error metrics against the Blender ground truth
- Configurable paths instead of hardcoded ones
- Advanced pre-processing: adaptive filtering, edge-preserving smoothing, learned denoising
- Improved calibration for tighter rectification
- Real-time optimization of the matching loop
- Learning-based disparity refinement
- Hybrid matching combining local and semi-global strategies

---

## Related work

The Master's thesis that followed extended the same domain into photogrammetry:
[photogrammetry-image-preprocessing](https://github.com/Selvam-DG/photogrammetry-image-preprocessing).

---

## Author

Selvam Dasari Gnanaprakash ·
[GitHub](https://github.com/Selvam-DG) ·
[LinkedIn](https://www.linkedin.com/in/selvamdasari55/)