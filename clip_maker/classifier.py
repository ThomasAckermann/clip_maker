"""
Action spotting inference using a fine-tuned VideoMAE model.

Runs a sliding window of CLIP_LEN frames over a rally video at a given stride,
classifies each window, then applies per-label non-maximum suppression (NMS)
to produce a clean list of detected action events.

Supports both:
  - Merged checkpoints (plain VideoMAEForVideoClassification)
  - PEFT/LoRA adapter checkpoints (adapter_config.json present)

Usage:
    spotter = ActionSpotter("models/videomae-vnl-lora/best")
    events  = spotter.spot("clips/rally_001.mp4")
    for e in events:
        print(f"{e.label:10s} @ {e.sec:.2f}s  (conf {e.confidence:.2f})")
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ── Constants (must match training/dataset.py) ────────────────────────────────

CLASSES: list[str] = ["background", "block", "receive", "score", "serve", "set", "spike"]
CLIP_LEN: int = 16
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Inference transform — no augmentation, matches the val transform in dataset.py
_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ActionEvent:
    frame: int
    sec: float
    label: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "frame":      self.frame,
            "sec":        self.sec,
            "label":      self.label,
            "confidence": self.confidence,
        }


# ── Main class ────────────────────────────────────────────────────────────────

class ActionSpotter:
    """
    Detects volleyball action events in a video using a sliding window.

    Args:
        checkpoint_dir: Path to a trained checkpoint directory.
                        May be a PEFT adapter dir (contains adapter_config.json)
                        or a merged full-model dir (contains config.json).
        device:         Torch device. Auto-detected if None.
        stride:         Frame step between consecutive inference windows.
                        stride=8 at 25fps → one inference per 0.32s.
        threshold:      Minimum softmax confidence to keep a detection.
        nms_radius:     Suppress duplicate detections of the same label within
                        this many frames of a higher-confidence detection.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        device: torch.device | None = None,
        stride: int = 8,
        threshold: float = 0.5,
        nms_radius: int = 16,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.stride = stride
        self.threshold = threshold
        self.nms_radius = nms_radius
        self.device = device or _auto_device()
        self.model = self._load_model().to(self.device).eval()

    # ── Inference ─────────────────────────────────────────────────────────────

    def spot(self, video_path: Path | str) -> list[ActionEvent]:
        """
        Run action spotting on a video file.

        Returns a list of ActionEvent sorted by frame, with background
        detections and low-confidence predictions already removed.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_buf: deque[torch.Tensor] = deque(maxlen=CLIP_LEN)
        raw: list[dict] = []
        frame_idx  = 0

        print(f"  Spotting actions ({total} frames, stride={self.stride}) …")

        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break

                frame_buf.append(_preprocess(bgr))

                if len(frame_buf) == CLIP_LEN and frame_idx % self.stride == 0:
                    label, conf = self._infer(frame_buf)
                    if label != "background" and conf >= self.threshold:
                        # Center frame of the window is the event frame
                        raw.append({
                            "frame": max(0, frame_idx - CLIP_LEN // 2),
                            "label": label,
                            "confidence": conf,
                        })

                frame_idx += 1
        finally:
            cap.release()

        events = _nms(raw, radius=self.nms_radius)
        print(f"  Found {len(events)} action(s)" + (":" if events else ""))
        for e in events:
            print(f"    {e['label']:10s}  frame {e['frame']:5d}  "
                  f"{e['frame'] / fps:6.2f}s  conf {e['confidence']:.2f}")

        return [
            ActionEvent(
                frame=e["frame"],
                sec=round(e["frame"] / fps, 3),
                label=e["label"],
                confidence=round(e["confidence"], 4),
            )
            for e in events
        ]

    @torch.no_grad()
    def _infer(self, frame_buf: deque[torch.Tensor]) -> tuple[str, float]:
        """Run one forward pass on a 16-frame window."""
        # Stack to (16, 3, 224, 224) then add batch dim → (1, 16, 3, 224, 224)
        pixel_values = torch.stack(list(frame_buf)).unsqueeze(0).to(self.device)
        logits = self.model(pixel_values=pixel_values).logits[0]
        probs  = torch.softmax(logits, dim=-1)
        idx    = int(probs.argmax())
        return CLASSES[idx], float(probs[idx])

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> nn.Module:
        from transformers import VideoMAEForVideoClassification

        adapter_cfg_path = self.checkpoint_dir / "adapter_config.json"

        if adapter_cfg_path.exists():
            # PEFT adapter checkpoint — read base model ID from adapter_config.json
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    "peft is required to load adapter checkpoints. "
                    "Install it with: pip install 'peft>=0.10.0'"
                )
            adapter_cfg = json.loads(adapter_cfg_path.read_text())
            base_id = adapter_cfg.get("base_model_name_or_path", "MCG-NJU/videomae-base")
            print(f"  Loading adapter checkpoint (base: {base_id}) …")
            base = VideoMAEForVideoClassification.from_pretrained(
                base_id,
                num_labels=len(CLASSES),
                ignore_mismatched_sizes=True,
                label2id={c: i for i, c in enumerate(CLASSES)},
                id2label={i: c for i, c in enumerate(CLASSES)},
            )
            return PeftModel.from_pretrained(base, self.checkpoint_dir)

        # Merged / full fine-tune checkpoint
        print(f"  Loading checkpoint from {self.checkpoint_dir} …")
        return VideoMAEForVideoClassification.from_pretrained(self.checkpoint_dir)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _preprocess(bgr: np.ndarray) -> torch.Tensor:
    """
    Resize to height 224 (matching frame extraction with `scale=-2:224`),
    CenterCrop to 224×224, apply ImageNet normalisation.
    Returns a (3, 224, 224) float tensor.
    """
    h, w = bgr.shape[:2]
    new_w = max(1, int(round(w * 224 / h)))
    resized = cv2.resize(bgr, (new_w, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return _TRANSFORM(Image.fromarray(rgb))


def _nms(detections: list[dict], radius: int = 16) -> list[dict]:
    """
    Per-label non-maximum suppression.

    For each label, repeatedly picks the highest-confidence detection and
    removes all detections of the same label within `radius` frames of it.
    Returns results sorted by frame.
    """
    if not detections:
        return []

    by_label: dict[str, list[dict]] = defaultdict(list)
    for d in detections:
        by_label[d["label"]].append(d)

    result: list[dict] = []
    for dets in by_label.values():
        remaining = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        while remaining:
            best = remaining[0]
            result.append(best)
            remaining = [
                d for d in remaining[1:]
                if abs(d["frame"] - best["frame"]) > radius
            ]

    return sorted(result, key=lambda d: d["frame"])
