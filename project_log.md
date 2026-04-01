# Project Research & Development Log

## Overview
This log tracks the development, notes, and future suggestions for the **Robust Semantic-Aware Robotic Canvas Reconstruction** project.

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

**Suggestions for Paper Writing & Future Work**:
1. **Model Weights**: To generate publication-quality *empirical results*, we need to train the Zero-DCE model on a paired low-light/normal dataset and load actual `.pth` weights. Currently, the code structure is perfect for the *methodology* section of the paper, but empirical result tables will need proper weight loading.
2. **Text Simulation**: The Text LLM is currently an offline dictionary proxy (e.g., swapping `smentic` -> `semantic`). For the paper, you may want to swap this out with an API call to a local LLM or OpenAI for more dynamic spellchecking if requested by reviewers.
3. **IK Solver Integration**: The Simulation script implements safety boundaries but the final continuous feedback loop between `main.py` stroke paths and `p.setJointMotorControl2` relies on numerical IK (Damped Least Squares). Real experimental results for the robot drawing on a physical canvas will require tuning the PyBullet friction and mass properties.
4. **Metrics Generation**: The pipeline architecture is ready. You can now start processing large datasets of images through `main.py` to calculate exact PSNR, SSIM, and Path Efficiency ($\eta$) averages for the paper's results section.

---
