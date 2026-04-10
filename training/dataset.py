"""
VNL action-spotting dataset for VideoMAE fine-tuning.

Label format (train.json / val.json / test.json):
  Each entry is one rally:
    {
      "video": "<match>/<rally>",      # relative path under frames_224p/
      "num_frames": 160,
      "fps": 25.0,
      "events": [
        {"frame": 52, "label": "serve", "xy": [0.79, 0.46]},
        ...
      ]
    }

Strategy — window classification:
  For each labeled event at frame F, extract a clip of CLIP_LEN frames
  centered on F and assign the event label.  Background clips are sampled
  from regions that are at least CLIP_LEN frames away from every event.
  One background clip is drawn per event to keep classes roughly balanced.

Output per sample:
  pixel_values : Float tensor  (CLIP_LEN, C, H, W) — ImageNet normalised
  label        : int           — index into CLASSES

Classes (order matches class.txt):
  0 background  1 block  2 receive  3 score  4 serve  5 set  6 spike
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES: list[str] = ["background", "block", "receive", "score", "serve", "set", "spike"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

CLIP_LEN = 16           # frames fed to VideoMAE
FRAME_SIZE = 224        # VideoMAE expects 224×224 RGB

# ImageNet statistics (VideoMAE was pre-trained with these)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ── Transforms ───────────────────────────────────────────────────────────────

def _build_transforms(augment: bool) -> transforms.Compose:
    """
    Return a per-frame transform pipeline.
    Frames arrive as PIL Images (398×224); we center-crop to 224×224 and
    optionally apply training augmentations.
    """
    steps: list = [
        transforms.CenterCrop(FRAME_SIZE),  # 398×224 → 224×224
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
    if augment:
        # Insert spatial augmentations before CenterCrop so the crop is always clean.
        steps = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        ] + steps
    return transforms.Compose(steps)


# ── Dataset ──────────────────────────────────────────────────────────────────

class VNLDataset(Dataset):
    """
    Args:
        json_path:    Path to train.json / val.json / test.json.
        frames_root:  Root directory that contains the per-match subdirectories
                      (i.e. the ``frames_224p`` folder).
        clip_len:     Number of frames per clip (default 16, matches VideoMAE).
        bg_ratio:     Background clips sampled per event (default 1.0).
        augment:      Apply random augmentations (should be True only for train).
        seed:         RNG seed for reproducible background sampling.
    """

    def __init__(
        self,
        json_path: Path | str,
        frames_root: Path | str,
        clip_len: int = CLIP_LEN,
        bg_ratio: float = 1.0,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.frames_root = Path(frames_root)
        self.clip_len = clip_len
        self.transform = _build_transforms(augment)

        with open(json_path) as f:
            entries = json.load(f)

        rng = random.Random(seed)
        self.samples: list[tuple[Path, int, int]] = []  # (rally_path, center_frame, label_idx)

        for entry in entries:
            rally_path = self.frames_root / entry["video"]
            # Guard against path traversal in the JSON (e.g. "../../etc/passwd")
            if not rally_path.resolve().is_relative_to(self.frames_root.resolve()):
                raise ValueError(
                    f"Unsafe video path in dataset JSON: {entry['video']!r} "
                    f"escapes frames_root {self.frames_root}"
                )
            num_frames = entry["num_frames"]
            events = sorted(entry["events"], key=lambda e: e["frame"])

            # ── Event clips ──────────────────────────────────────────────────
            event_frames: list[int] = []
            for ev in events:
                label_idx = CLASS_TO_IDX.get(ev["label"])
                if label_idx is None:
                    continue  # unknown label — skip
                center = ev["frame"]
                self.samples.append((rally_path, center, num_frames, label_idx))
                event_frames.append(center)

            # ── Background clips ─────────────────────────────────────────────
            n_bg = max(1, int(len(event_frames) * bg_ratio))
            bg_candidates = _background_candidates(
                num_frames=num_frames,
                event_frames=event_frames,
                clip_len=clip_len,
            )
            if bg_candidates:
                chosen = rng.choices(bg_candidates, k=n_bg)
                for center in chosen:
                    self.samples.append((rally_path, center, num_frames, CLASS_TO_IDX["background"]))

        # Flatten the 4-tuple to match __getitem__ expectations
        # (rally_path, center_frame, num_frames, label_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rally_path, center_frame, num_frames, label_idx = self.samples[idx]
        frames = self._load_clip(rally_path, center_frame, num_frames)
        # frames: list of CLIP_LEN PIL Images → tensor (CLIP_LEN, C, H, W)
        tensors = torch.stack([self.transform(f) for f in frames])
        return tensors, label_idx

    def _load_clip(self, rally_path: Path, center_frame: int, num_frames: int) -> list[Image.Image]:
        half = self.clip_len // 2
        start = center_frame - half
        images: list[Image.Image] = []
        for i in range(self.clip_len):
            frame_idx = max(0, min(start + i, num_frames - 1))
            img_path = rally_path / f"{frame_idx:06d}.jpg"
            images.append(Image.open(img_path).convert("RGB"))
        return images


# ── Helpers ──────────────────────────────────────────────────────────────────

def _background_candidates(
    num_frames: int,
    event_frames: list[int],
    clip_len: int,
) -> list[int]:
    """
    Return valid center frames for background clips — at least clip_len frames
    away from every event and at least clip_len//2 from either end of the rally.
    """
    margin = clip_len  # exclusion zone around each event
    half = clip_len // 2

    # Build a set of excluded frames
    excluded: set[int] = set()
    for ef in event_frames:
        for f in range(ef - margin, ef + margin + 1):
            excluded.add(f)

    candidates = [
        f for f in range(half, num_frames - half)
        if f not in excluded
    ]
    return candidates


# ── Class-weight helper ───────────────────────────────────────────────────────

def class_weights(dataset: VNLDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for use with CrossEntropyLoss.
    Returns a tensor of shape (num_classes,).
    """
    counts = torch.zeros(len(CLASSES))
    for *_, label_idx in dataset.samples:
        counts[label_idx] += 1
    counts = counts.clamp(min=1)
    weights = counts.sum() / (len(CLASSES) * counts)
    return weights
