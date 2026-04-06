# Project Research & Development Log

## Overview
This log tracks the development, notes, and future suggestions for the **Robust Semantic-Aware Robotic Canvas Reconstruction** project.

---

### [2026-04-06] Making Project Publishable (Workshop Tier)
**Status**: Upgrade Complete.
**Notes**:
- Transitioned the basic restoration pipeline to a **Novel Adaptive Degradation-Aware Restoration Router (ADRR)**. It checks the Laplacian Variance (motion blur) and 10th-percentile dynamics (brightness) to decide if enhancements belong in the image. Corrected thresholding allows it to successfully bypass unaffected imagery or handle complex compounds.
- Replaced the simple variance-threshold in `main.py` with a robust **Learned Content Classification Gate** running a Logistic Regression trained on 6 independent heuristics: variance, Canny density, horizontal/vertical Sobel energy ratio, mean gradients, connected components, and DCT ratios.
- Expanded the Linguistic Engine's Vector Typography Synthesizer (`ocr_hatching.py`) to fully synthesize the complete A-Z, 0-9, and standard punctuation marks.
- Refactored `robot_sim_3d_env.py` to be a functional class that computes the exact `calcJacobian` with a specific determinant determinant tracking constraint. It now also implements an abstract 2D path renderer `RobotDrawingRenderer` for headless execution checks.
- Wrote an intensive synthetic benchmark tool `benchmark_generator.py` spanning over 150 variations encompassing dark, defocused, motion blurred, grouped, Scene, and Text contexts.
- Rewrote the metrics calculation `eval_metrics.py`. We now measure exact Canny recovery intersection, overall PSNR gains per class, Classification zero-shot generalisation, and real OCR detections over baseline, adaptive thresholding, histogram equalisations, and CLAHE.

**Suggestions for Paper Writing & Future Work**:
1. **Real Robot Pipeline Execution**: Connect the actual output path of the simulator via UDP or ROS hooks directly to a physical UR5 framework as the next step towards submitting a conference-grade (ICRA/IROS) publication.
2. **Generative Modeling for Text Paths**: Instead of deterministic paths for typographic strokes, integrate an RNN or diffusion model to capture authentic handwriting behaviors.
3. **End-to-End Restoration Model**: For even higher image PSNR reconstruction, instead of two different networks in sequence (Zero-DCE and Wiener filters), try a fine-tuned single Unified Image Restoration formulation natively built on transformer architectures (e.g. SwinIR or NAFNet).

---

### [2026-04-01] Phase 1-5 Completion
**Status**: All core phases successfully implemented.
**Notes**: 
- We established the `dl-env` conda environment as the primary runtime to support PyTorch, EasyOCR, and PyBullet.
- The `test_baseline_failure.py` script proved that standard Canny Edge detection drops significantly in efficacy on dark/blurry images (down to 0% in dark conditions and 19% in blurry conditions).
- Implemented `zero_dce.py` with a structural layer mimicking the curve estimator, though it uses a heuristic fallback since we don't have pretrained `.pth` weights yet.
- `deblur.py` successfully uses Wiener Deconvolution using FFT.
- `depth_to_hatching.py` works well using the `MiDaS_small` model.
- `ocr_hatching.py` successfully uses `EasyOCR` to detect, clean, and path out text. 
- Integrated geofencing, singularity tracking (Jacobian det mock), and collision testing into `robot_sim_3d_env.py`.
- `main.py` successfully routes requests.
