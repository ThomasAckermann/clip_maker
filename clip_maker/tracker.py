"""
Ball tracker wrapping the VballNet ONNX model.

Model: VballNetV1_seq9_grayscale_330_h288_w512.onnx
  Input : (1, 9, 288, 512)  float32 — 9 consecutive grayscale frames, normalised [0, 1]
  Output: (1, 1, 288, 512)  float32 — Gaussian heatmap

Post-processing:
  1. Threshold heatmap at `heatmap_threshold` (default 0.5)
  2. Find the largest contour in the thresholded mask
  3. Compute centroid → (x, y) in model space
  4. Scale back to original frame resolution

Model download:
  https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/master/models/VballNetV1_seq9_grayscale_330_h288_w512.onnx
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import onnxruntime as ort

# ── Model constants (must match the ONNX filename) ───────────────────────────
MODEL_H = 288
MODEL_W = 512
SEQ_LEN = 9  # number of frames the model sees at once

DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/asigatchov/"
    "fast-volleyball-tracking-inference/master/models/"
    "VballNetV1_seq9_grayscale_330_h288_w512.onnx"
)
DEFAULT_MODEL_NAME = "VballNetV1_seq9_grayscale_330_h288_w512.onnx"
DEFAULT_MODEL_SHA256 = "2f36bd129c51a4d8afb9a23fd4816be578c65041d463d55a81d1667735070b76"


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class Detection:
    frame_idx: int
    x: float  # pixel coords in original video resolution
    y: float
    confidence: float  # peak heatmap value at the detected centroid


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(
    dest_dir: Path,
    url: str = DEFAULT_MODEL_URL,
    expected_sha256: str = DEFAULT_MODEL_SHA256,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / DEFAULT_MODEL_NAME
    if dest.exists():
        if _sha256(dest) == expected_sha256:
            return dest
        print("Cached model failed integrity check — re-downloading.")
        dest.unlink()
    print(f"Downloading model → {dest}")
    urllib.request.urlretrieve(url, dest)
    actual = _sha256(dest)
    if actual != expected_sha256:
        dest.unlink()
        raise RuntimeError(
            f"Downloaded model failed SHA-256 check.\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}\n"
            "Do not use this file."
        )
    return dest


def _build_session(model_path: Path, use_gpu: bool) -> ort.InferenceSession:
    providers: list[str] = []
    if use_gpu:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(model_path), providers=providers)


def _heatmap_to_detection(
    heatmap: np.ndarray,  # (H, W) float32
    frame_idx: int,
    orig_w: int,
    orig_h: int,
    threshold: float,
) -> Detection | None:
    mask = (heatmap >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx_model = M["m10"] / M["m00"]
    cy_model = M["m01"] / M["m00"]

    # Scale from model resolution back to original frame resolution
    cx = cx_model * orig_w / MODEL_W
    cy = cy_model * orig_h / MODEL_H

    # Confidence = peak heatmap value inside the detected region
    confidence = float(heatmap[mask == 1].max())

    return Detection(frame_idx=frame_idx, x=cx, y=cy, confidence=confidence)


# ── Main tracker class ───────────────────────────────────────────────────────

class BallTracker:
    """
    Runs the ONNX ball-tracking model over a video and yields one
    Detection (or None) per frame.

    Uses a sliding window of SEQ_LEN frames. The first SEQ_LEN-1 frames
    are used to warm up the buffer; detections start from frame 0 but the
    first SEQ_LEN-1 predictions re-use the first frame as padding.
    """

    def __init__(
        self,
        model_path: Path,
        heatmap_threshold: float = 0.5,
        use_gpu: bool = False,
    ) -> None:
        self._session = _build_session(model_path, use_gpu)
        self._input_name = self._session.get_inputs()[0].name
        self._threshold = heatmap_threshold

    def track(
        self, video_path: Path
    ) -> Generator[Detection | None, None, None]:
        """
        Yields Detection | None for every frame in the video.
        Caller is responsible for progress display if desired.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Fixed-size circular buffer — only keeps the last SEQ_LEN frames.
        frame_buffer: deque[np.ndarray] = deque(maxlen=SEQ_LEN)
        frame_idx = 0

        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break

                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (MODEL_W, MODEL_H))
                normalised = resized.astype(np.float32) / 255.0
                frame_buffer.append(normalised)

                # Pad with the first frame until buffer is full
                if len(frame_buffer) < SEQ_LEN:
                    padded = [frame_buffer[0]] * (SEQ_LEN - len(frame_buffer)) + list(frame_buffer)
                else:
                    padded = list(frame_buffer)

                # Build input tensor: (1, SEQ_LEN, H, W)
                tensor = np.stack(padded, axis=0)[np.newaxis]  # (1, 9, 288, 512)
                heatmap = self._session.run(None, {self._input_name: tensor})[0]
                heatmap = heatmap[0, 0]  # (288, 512)

                yield _heatmap_to_detection(
                    heatmap, frame_idx, orig_w, orig_h, self._threshold
                )
                frame_idx += 1
        finally:
            cap.release()
