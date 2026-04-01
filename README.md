# Robust Semantic-Aware Robotic Canvas Reconstruction

This repository contains the codebase for the **Robust Semantic-Aware Robotic Canvas Reconstruction** project. The system is designed to act as a "superhuman" robotic drawing pipeline that transcends traditional "edge tracing" algorithms by understanding 3D volumes, cleaning corrupted data (low light and motion blur), and intelligently repairing linguistic text. 

## Project Architecture (Tri-Stream System)

1. **Restoration Gatekeeper:**
   - **Low-Light Enhancement**: Utilizes a Zero-DCE proxy to mathematically enhance darkened images.
   - **Motion Deblurring**: Utilizes a Wiener Deconvolution function in the frequency domain to reverse motion blur and sharpen edges.
2. **Volumetric Perception Engine (Visual Stream):**
   - Integrates **MiDaS** for Monocular Depth Estimation.
   - Computes X/Y surface normals from the depth gradients.
   - Outputs parametric hatching strokes corresponding to 3D curvature.
3. **Linguistic Restoration Engine (Text Stream):**
   - Utilizes **EasyOCR** for bounding box text detection and extraction.
   - Repairs spelling anomalies using an offline rule-based proxy.
   - Generates vector-synthesized Bézier paths corresponding to perfect typography.
4. **Physical Simulation (PyBullet):**
   - Integrated a 6-DOF robotic manipulator simulation.
   - Implements safety protocols: *Geofencing*, *Singularity Avoidance*, and *Collision Prediction*.

## Environment Setup
It is highly recommended to run this repository inside the dedicated `dl-env` Conda environment to ensure PyTorch and EasyOCR run efficiently.

```bash
conda activate dl-env
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Baseline Test ("Dirty Data")
To verify the failure of standard edge-tracing algorithms (Canny) under degraded conditions:
```bash
python src/tests/test_baseline_failure.py
```
Outputs will be generated in `outputs/` showing the drastic loss of edge-detection efficacy.

### 2. Main Orchestrator (Router)
To process an image through the entire pipeline:
```bash
python src/main.py --input outputs/01_dark.jpg
```
The router will automatically clean the image, classify the content (Text vs Scene), invoke the correct perception engine, and simulate the vector paths for the IK solver.

## Directory Structure
- `src/utils/` - Contains data degradation hooks and general utilities.
- `src/perception/restoration/` - Zero-DCE and Deblurring logic.
- `src/perception/vision/` - Depth estimation and hatching logic.
- `src/perception/text/` - OCR and typographic generation.
- `src/simulation/` - PyBullet physics environments.
- `outputs/` - Generated imagery and test cases.
