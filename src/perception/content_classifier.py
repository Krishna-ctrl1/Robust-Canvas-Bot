"""
Learned Content Classification Gate

Replaces the naive variance-threshold classifier with a feature-based
logistic regression model trained on synthetic text vs. scene images.

Features extracted:
  1. Grayscale variance
  2. Edge density (fraction of Canny edge pixels)
  3. Horizontal/vertical Sobel energy ratio
  4. Mean gradient magnitude
  5. Connected component count (normalised by image area)
  6. High-frequency energy ratio (DCT-based)

The classifier is trained on auto-generated synthetic data at initialization
and can be persisted via pickle for production use.
"""

import cv2
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ContentClassifier:
    """
    A lightweight learned classifier that distinguishes text-dominant images
    from scene/object images using 6 handcrafted image features.
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier_model.pkl")

    def __init__(self, auto_train=True):
        self.model = None
        self.scaler = None

        # Try to load a persisted model first
        if os.path.exists(self.MODEL_PATH):
            self._load_model()
        elif auto_train:
            self._train_on_synthetic_data()

    @staticmethod
    def extract_features(image):
        """
        Extract a 6-dimensional feature vector from an image.

        Returns:
            np.ndarray of shape (6,)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        # F1: Grayscale variance (text = high contrast = high variance)
        f_variance = np.var(gray.astype(np.float64))

        # F2: Edge density (text = many fine edges)
        edges = cv2.Canny(gray, 50, 150)
        f_edge_density = np.sum(edges > 0) / (h * w)

        # F3: Horizontal vs vertical Sobel energy ratio
        # Text tends to have strong horizontal energy (baselines)
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy_h = np.sum(sobel_h ** 2)
        energy_v = np.sum(sobel_v ** 2) + 1e-8  # avoid division by zero
        f_hv_ratio = energy_h / energy_v

        # F4: Mean gradient magnitude
        mag = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
        f_mean_gradient = np.mean(mag)

        # F5: Connected component count (normalised)
        # Text images have many small, separated components
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, _ = cv2.connectedComponents(binary)
        f_cc_density = num_labels / (h * w) * 10000  # normalise to readable range

        # F6: High-frequency energy ratio (DCT)
        # Text tends to have proportionally more high-frequency content
        dct = cv2.dct(np.float32(gray) / 255.0)
        total_energy = np.sum(dct ** 2) + 1e-8
        # High-freq = bottom-right quadrant
        hf_energy = np.sum(dct[h // 2:, w // 2:] ** 2)
        f_hf_ratio = hf_energy / total_energy

        return np.array([
            f_variance, f_edge_density, f_hv_ratio,
            f_mean_gradient, f_cc_density, f_hf_ratio
        ])

    @staticmethod
    def _generate_text_image(h=400, w=600):
        """Generate a synthetic text-dominant image."""
        img = np.ones((h, w), dtype=np.uint8) * np.random.randint(180, 240)

        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
        ]
        words = [
            "HELLO WORLD", "TESTING", "ROBUST CANVAS",
            "PERCEPTION", "SEMANTIC", "ROBOT ARM",
            "DEEP LEARNING", "HATCHING", "VECTOR",
            "RESTORATION", "PIPELINE", "DRAWING",
        ]

        num_lines = np.random.randint(2, 6)
        for i in range(num_lines):
            word = words[np.random.randint(0, len(words))]
            font = fonts[np.random.randint(0, len(fonts))]
            scale = np.random.uniform(0.6, 2.0)
            thickness = np.random.randint(1, 4)
            y_pos = int((i + 1) * h / (num_lines + 1))
            x_pos = np.random.randint(10, max(11, w // 4))
            color = np.random.randint(0, 80)
            cv2.putText(img, word, (x_pos, y_pos), font, scale, int(color), thickness)

        return img

    @staticmethod
    def _generate_scene_image(h=400, w=600):
        """Generate a synthetic scene/object image."""
        # Create gradient background
        bg_val = np.random.randint(100, 200)
        img = np.ones((h, w), dtype=np.uint8) * bg_val

        # Add random gradient
        gradient = np.linspace(0, np.random.randint(30, 80), w).astype(np.uint8)
        img = (img.astype(np.int16) + gradient[np.newaxis, :]).clip(0, 255).astype(np.uint8)

        # add random shapes
        num_shapes = np.random.randint(3, 8)
        for _ in range(num_shapes):
            shape_type = np.random.choice(["circle", "rect", "ellipse", "triangle"])
            color = np.random.randint(0, 180)

            if shape_type == "circle":
                center = (np.random.randint(50, w - 50), np.random.randint(50, h - 50))
                radius = np.random.randint(20, min(h, w) // 4)
                cv2.circle(img, center, radius, int(color), -1)
            elif shape_type == "rect":
                pt1 = (np.random.randint(0, w - 50), np.random.randint(0, h - 50))
                pt2 = (pt1[0] + np.random.randint(30, 150), pt1[1] + np.random.randint(30, 150))
                cv2.rectangle(img, pt1, pt2, int(color), -1)
            elif shape_type == "ellipse":
                center = (np.random.randint(50, w - 50), np.random.randint(50, h - 50))
                axes = (np.random.randint(20, 80), np.random.randint(20, 80))
                angle = np.random.randint(0, 180)
                cv2.ellipse(img, center, axes, angle, 0, 360, int(color), -1)
            elif shape_type == "triangle":
                pts = np.array([
                    [np.random.randint(50, w - 50), np.random.randint(50, h - 50)],
                    [np.random.randint(50, w - 50), np.random.randint(50, h - 50)],
                    [np.random.randint(50, w - 50), np.random.randint(50, h - 50)],
                ])
                cv2.fillPoly(img, [pts], int(color))

        # Add more significant noise to better simulate the artifacts introduced
        # by the Zero-DCE restoration stage, preventing false classification as text.
        noise = np.random.normal(0, 25, img.shape).astype(np.int16)
        img = (img.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)

        return img

    def _train_on_synthetic_data(self):
        """
        Generate synthetic training data and train the classifier.
        Uses 200 samples per class for robust generalisation.
        """
        print("Training content classifier on synthetic data...")
        n_per_class = 200
        features = []
        labels = []

        np.random.seed(42)  # reproducibility for paper

        for _ in range(n_per_class):
            text_img = self._generate_text_image()
            feat = self.extract_features(text_img)
            features.append(feat)
            labels.append(1)  # TEXT

        for _ in range(n_per_class):
            scene_img = self._generate_scene_image()
            feat = self.extract_features(scene_img)
            features.append(feat)
            labels.append(0)  # SCENE

        X = np.array(features)
        y = np.array(labels)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)

        train_acc = self.model.score(X_scaled, y)
        print(f"  Classifier trained. Training accuracy: {train_acc:.4f}")

        # Persist model
        self._save_model()

    def _save_model(self):
        """Save trained model and scaler to disk."""
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def _load_model(self):
        """Load persisted model and scaler."""
        with open(self.MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]

    def predict(self, image):
        """
        Classify an image as 'TEXT' or 'SCENE'.

        Returns:
            (label: str, confidence: float, features: np.ndarray)
        """
        if self.model is None:
            # Fallback to variance heuristic
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            var = np.var(gray)
            label = "TEXT" if var > 2000 else "SCENE"
            return label, 0.5, np.array([var])

        features = self.extract_features(image)
        X = self.scaler.transform(features.reshape(1, -1))
        proba = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]

        label = "TEXT" if pred == 1 else "SCENE"
        confidence = float(max(proba))

        return label, confidence, features


if __name__ == "__main__":
    clf = ContentClassifier(auto_train=True)

    # Test on a sample
    test_text = ContentClassifier._generate_text_image()
    test_scene = ContentClassifier._generate_scene_image()

    label_t, conf_t, _ = clf.predict(test_text)
    label_s, conf_s, _ = clf.predict(test_scene)

    print(f"Text image  -> {label_t} (confidence: {conf_t:.3f})")
    print(f"Scene image -> {label_s} (confidence: {conf_s:.3f})")
