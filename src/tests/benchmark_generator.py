"""
Benchmark Generator

Generates a comprehensive synthetic test dataset for evaluating the
restoration and perception pipeline. The benchmark contains:

  - Multiple content types (scene, text, mixed)
  - Multiple degradation types (low-light, motion blur, combined, noise)
  - Multiple severity levels per degradation type
  - Clean originals for every degraded variant (enabling PSNR/SSIM computation)

The benchmark is self-documenting: a manifest CSV is generated alongside
the images, recording the content type, degradation type, parameters,
and file paths for each test case.

Total images generated: ~60 unique degraded variants + their clean originals.
"""

import cv2
import numpy as np
import os
import csv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_degradation import make_dark, make_motion_blur, make_blurry, add_gaussian_noise, make_combined


# ─── Content Generators ──────────────────────────────────────────────────────

def generate_scene_image(seed, h=400, w=600):
    """Generate a diverse scene image with multiple geometric shapes."""
    rng = np.random.RandomState(seed)

    # Light background with gradient
    bg = rng.randint(160, 230)
    img = np.ones((h, w), dtype=np.uint8) * bg

    # Add directional gradient for depth illusion
    direction = rng.choice(['horizontal', 'vertical', 'diagonal'])
    grad = np.linspace(0, rng.randint(20, 60), w).astype(np.float32)
    if direction == 'horizontal':
        img = (img.astype(np.float32) + grad[np.newaxis, :]).clip(0, 255).astype(np.uint8)
    elif direction == 'vertical':
        grad_v = np.linspace(0, rng.randint(20, 60), h).astype(np.float32)
        img = (img.astype(np.float32) + grad_v[:, np.newaxis]).clip(0, 255).astype(np.uint8)

    num_shapes = rng.randint(3, 8)
    for _ in range(num_shapes):
        shape = rng.choice(['circle', 'rect', 'ellipse', 'triangle', 'polygon'])
        color = int(rng.randint(0, 150))
        thickness = rng.choice([-1, 2, 3])  # filled or outline

        if shape == 'circle':
            cx, cy = rng.randint(60, w - 60), rng.randint(60, h - 60)
            r = rng.randint(20, min(h, w) // 4)
            cv2.circle(img, (cx, cy), r, color, thickness)
        elif shape == 'rect':
            x1, y1 = rng.randint(10, w - 100), rng.randint(10, h - 100)
            x2, y2 = x1 + rng.randint(40, 150), y1 + rng.randint(40, 120)
            cv2.rectangle(img, (x1, y1), (min(x2, w-1), min(y2, h-1)), color, thickness)
        elif shape == 'ellipse':
            cx, cy = rng.randint(60, w - 60), rng.randint(60, h - 60)
            ax1, ax2 = rng.randint(15, 70), rng.randint(15, 70)
            angle = rng.randint(0, 180)
            cv2.ellipse(img, (cx, cy), (ax1, ax2), angle, 0, 360, color, thickness)
        elif shape == 'triangle':
            pts = np.array([
                [rng.randint(30, w-30), rng.randint(30, h-30)],
                [rng.randint(30, w-30), rng.randint(30, h-30)],
                [rng.randint(30, w-30), rng.randint(30, h-30)],
            ])
            if thickness == -1:
                cv2.fillPoly(img, [pts], color)
            else:
                cv2.polylines(img, [pts], True, color, thickness)
        elif shape == 'polygon':
            n_pts = rng.randint(4, 7)
            cx, cy = rng.randint(80, w-80), rng.randint(80, h-80)
            angles = np.sort(rng.uniform(0, 2*np.pi, n_pts))
            r = rng.randint(25, 60)
            pts = np.array([[int(cx + r*np.cos(a)), int(cy + r*np.sin(a))] for a in angles])
            if thickness == -1:
                cv2.fillPoly(img, [pts], color)
            else:
                cv2.polylines(img, [pts], True, color, thickness)

    # Add subtle noise for realism
    noise = rng.normal(0, 3, img.shape).astype(np.float32)
    img = (img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)

    # Convert to BGR for pipeline compatibility
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def generate_text_image(seed, h=400, w=600):
    """Generate a text-dominant image with multiple lines."""
    rng = np.random.RandomState(seed)

    bg = rng.randint(200, 250)
    img = np.ones((h, w), dtype=np.uint8) * bg

    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]
    words_pool = [
        "ROBUST CANVAS", "SEMANTIC AWARE", "PERCEPTION",
        "RESTORATION", "DEEP LEARNING", "ROBOT ARM",
        "HATCHING PATHS", "VECTOR STROKE", "DEBLURRING",
        "ZERO DCE", "SURFACE NORMAL", "MIDAS DEPTH",
        "INVERSE KINEMATICS", "WIENER FILTER", "GEOFENCE",
        "PIPELINE", "CLASSIFICATION", "VOLUMETRIC",
        "LINGUISTIC", "TYPOGRAPHY", "RECOGNITION",
    ]

    num_lines = rng.randint(2, 5)
    for i in range(num_lines):
        text = words_pool[rng.randint(0, len(words_pool))]
        font = fonts[rng.randint(0, len(fonts))]
        scale = rng.uniform(0.7, 1.8)
        thickness = rng.randint(1, 3)
        y = int((i + 1) * h / (num_lines + 1))
        x = rng.randint(10, max(11, w // 4))
        color = int(rng.randint(0, 60))
        cv2.putText(img, text, (x, y), font, scale, color, thickness)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def generate_mixed_image(seed, h=400, w=600):
    """Generate an image with both shapes and text."""
    rng = np.random.RandomState(seed)

    # Start with scene base
    img = generate_scene_image(seed, h, w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Add text overlay
    words = ["LABEL", "OBJECT", "SCENE", "TEXT", "MIXED", "TEST"]
    text = words[rng.randint(0, len(words))]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = rng.uniform(0.8, 1.5)
    y = rng.randint(h // 3, 2 * h // 3)
    x = rng.randint(10, w // 3)
    cv2.putText(gray, text, (x, y), font, scale, 0, 2)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ─── Degradation Configurations ──────────────────────────────────────────────

DEGRADATION_CONFIGS = [
    # Low-light degradations at 4 severity levels
    {"type": "dark", "params": {"factor": 0.10}, "label": "dark_severe"},
    {"type": "dark", "params": {"factor": 0.15}, "label": "dark_heavy"},
    {"type": "dark", "params": {"factor": 0.25}, "label": "dark_moderate"},
    {"type": "dark", "params": {"factor": 0.40}, "label": "dark_mild"},

    # Motion blur at 4 severity levels
    {"type": "motion_blur", "params": {"kernel_size": 25, "angle": 0}, "label": "mblur_severe_h"},
    {"type": "motion_blur", "params": {"kernel_size": 15, "angle": 0}, "label": "mblur_moderate_h"},
    {"type": "motion_blur", "params": {"kernel_size": 15, "angle": 45}, "label": "mblur_moderate_d"},
    {"type": "motion_blur", "params": {"kernel_size": 11, "angle": 90}, "label": "mblur_mild_v"},

    # Gaussian blur
    {"type": "gaussian_blur", "params": {"kernel_size": (21, 21)}, "label": "gblur_heavy"},
    {"type": "gaussian_blur", "params": {"kernel_size": (11, 11)}, "label": "gblur_mild"},

    # Combined degradations
    {"type": "combined", "params": {"dark_factor": 0.15, "blur_kernel": 15, "blur_angle": 0}, "label": "combined_dark_blur"},
    {"type": "combined", "params": {"dark_factor": 0.25, "blur_kernel": 11, "blur_angle": 45}, "label": "combined_moderate"},
]


def apply_degradation(image, config):
    """Apply a degradation config to an image."""
    dtype = config["type"]
    params = config["params"]

    if dtype == "dark":
        return make_dark(image, **params)
    elif dtype == "motion_blur":
        return make_motion_blur(image, **params)
    elif dtype == "gaussian_blur":
        return make_blurry(image, **params)
    elif dtype == "combined":
        return make_combined(image, **params)
    else:
        raise ValueError(f"Unknown degradation type: {dtype}")


# ─── Main Generator ──────────────────────────────────────────────────────────

def generate_benchmark(output_dir=None):
    """
    Generate the full benchmark dataset.

    Creates:
      - benchmark/originals/   — clean source images
      - benchmark/degraded/    — degraded variants
      - benchmark/manifest.csv — metadata for every test pair

    Returns:
        Path to the manifest CSV.
    """
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../benchmark'))

    orig_dir = os.path.join(output_dir, "originals")
    deg_dir = os.path.join(output_dir, "degraded")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(deg_dir, exist_ok=True)

    manifest = []
    image_id = 0

    # Content generators with counts
    content_configs = [
        ("scene", generate_scene_image, 5),
        ("text", generate_text_image, 5),
        ("mixed", generate_mixed_image, 3),
    ]

    print(f"Generating benchmark dataset in {output_dir}/")

    for content_type, generator, count in content_configs:
        for i in range(count):
            seed = 1000 * (["scene", "text", "mixed"].index(content_type)) + i
            clean_img = generator(seed)

            # Save original
            orig_name = f"{image_id:03d}_{content_type}_{i:02d}_clean.png"
            orig_path = os.path.join(orig_dir, orig_name)
            cv2.imwrite(orig_path, clean_img)

            # Apply each degradation
            for deg_config in DEGRADATION_CONFIGS:
                deg_img = apply_degradation(clean_img, deg_config)
                deg_label = deg_config["label"]

                deg_name = f"{image_id:03d}_{content_type}_{i:02d}_{deg_label}.png"
                deg_path = os.path.join(deg_dir, deg_name)
                cv2.imwrite(deg_path, deg_img)

                manifest.append({
                    "image_id": image_id,
                    "content_type": content_type,
                    "content_seed": seed,
                    "degradation_type": deg_config["type"],
                    "degradation_label": deg_label,
                    "degradation_params": str(deg_config["params"]),
                    "original_path": os.path.relpath(orig_path, output_dir),
                    "degraded_path": os.path.relpath(deg_path, output_dir),
                })
                image_id += 1

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    print(f"  Generated {len(manifest)} test cases ({len(manifest)} degraded + {sum(c for _,_,c in content_configs)} originals)")
    print(f"  Manifest saved to {manifest_path}")

    return manifest_path


if __name__ == "__main__":
    generate_benchmark()
