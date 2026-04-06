# Robust Semantic-Aware Robotic Canvas Reconstruction

This repository contains the codebase for the **Robust Semantic-Aware Robotic Canvas Reconstruction** project. The system is designed to act as a robotic drawing pipeline that transcends traditional "edge tracing" algorithms by understanding 3D volumes vs text, intelligently evaluating corrupted data (low light and motion blur) via an Adaptive Degradation-Aware Restoration Router (ADRR), and generating execution-safe stroke paths.

## Project Architecture (Tri-Stream System)

1. **Restoration Gatekeeper (ADRR):**
   - **Adaptive Degradation-Aware Restoration Router**: A novel module that analyzes the input image to identify its degradations, avoiding standard 'blind' correction issues.
   - **Low-Light Enhancement**: Utilizes a Zero-DCE neural network to mathematically enhance darkened images without overblowing already well-lit areas.
   - **Motion Deblurring**: Utilizes Wiener Deconvolution in the frequency domain to reverse motion blur and sharpen edges.
   
2. **Learned Content Classification Gate**
   - Extracts 6 distinct computational features (variance, Canny density, horizontal/vertical Sobel energy ratio, mean gradients, connected components, DCT ratios).
   - Utilizes a Logistic Regression classifier trained on synthetic data to route the data efficiently between Text vs Object pipelines.

3. **Volumetric Perception Engine (Visual Stream):**
   - Integrates **MiDaS** for Monocular Depth Estimation.
   - Computes X/Y surface normals from the depth gradients.
   - Outputs parametric hatching strokes corresponding to 3D curvature.

4. **Linguistic Restoration Engine (Text Stream):**
   - Utilizes **EasyOCR** for bounding box text detection and extraction.
   - Repairs spelling anomalies using an offline rule-based proxy.
   - Generates vector-synthesized Bézier/piecewise paths corresponding to perfect typography (Full A-Z, 0-9 & Punctuation Support).

5. **Physical Simulation (PyBullet):**
   - Integrated a 6-DOF robotic manipulator simulation.
   - Implements robust safety protocols: *Geofencing*, *Singularity Avoidance* (via Jacobian Determinant rank calculation), and *Collision Prediction*.

## Comprehensive Evaluation

The pipeline is quantitatively verified on a synthetically generated internal test suite of over 150 benchmark images across Text, Scenes, and mixed content types.

| Pipeline Feature                     | Results Snapshot                            |
| ------------------------------------ | ------------------------------------------- |
| **Adaptive Restoration Quality (PSNR)** | **+2.26 dB** average on low-light inputs.     |
| **Classification Accuracy**          | **90%+** zero-shot generalization.          |
| **Canny Edge Recovery (Intersection)** | Substantial improvements vs generic CLAHE. |
| **Safety Integration**               | 100% adherence to virtual work cell bounds. |

*(For full results, see `outputs/empirical_results.md` and `TECHNICAL_DOCUMENTATION.md`)*

## Environment Setup
It is highly recommended to run this repository inside the dedicated `dl-env` Conda environment to ensure PyTorch and EasyOCR run efficiently.

```bash
conda activate dl-env
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Main Orchestrator (Router)
To process an image through the entire pipeline:
```bash
python src/main.py --input <path_to_image> --render
```
The router will automatically analyze, clean, classify the content (Text vs Scene), invoke the correct perception engine, and render a 2D stroke image mimicking robot drawing.

### 2. Auto-Benchmarking (For Testing)
To verify the performance of the pipeline on 150+ synthetically degraded configurations:
```bash
python src/tests/benchmark_generator.py
python src/tests/eval_metrics.py
```
Outputs will be generated in `outputs/` demonstrating pathing efficacy, PSNR changes, and classification results.

## Directory Structure
- `src/utils/` - Contains data degradation hooks and general utilities.
- `src/perception/` - Content classifiers, depth to hatching logic, and general classification.
  - `restoration/` - ADRR (Adaptive Restoration), Zero-DCE, and Deblurring.
  - `vision/` - Dept estimation and 3D perception.
  - `text/` - OCR and typographic generation.
- `src/simulation/` - PyBullet physics environments and 2D vector path renderers.
- `src/tests/` - Automatic benchmark synthesis and evaluation logging scripts.
- `outputs/` - Generated imagery, rendering paths, and empirical results.
