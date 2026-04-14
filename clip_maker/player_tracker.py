"""
Player tracker wrapping YOLOv8 + ByteTrack (via Ultralytics).

For each rally clip, runs person detection and multi-object tracking to produce
consistent track IDs per player across the clip's frames.

Output per clip:
  List[PlayerDetection] — one entry per (frame, track) pair where a person was detected.

The tracker is lazy-imported so `clip-maker run` without --track-players has
no dependency on ultralytics / torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PlayerDetection:
    frame_idx: int
    track_id: int
    # Bounding box in original video pixel coordinates
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidence": round(self.confidence, 4),
        }


class PlayerTracker:
    """
    Runs YOLOv8 person detection + ByteTrack on a video clip.

    Args:
        model_name: Ultralytics model name or path (default: 'yolov8n.pt' — smallest/fastest).
        confidence: Minimum detection confidence (default 0.3).
        device: Torch device string — 'cpu', 'cuda', 'mps' (default: auto-detect).
    """

    PERSON_CLASS_ID = 0  # COCO class 0 = person

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.3,
        device: str | None = None,
    ) -> None:
        # Lazy import — only pay the torch startup cost when tracking is requested.
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for player tracking. "
                "Install it with: pip install 'clip-maker[track]'"
            ) from e

        self._model = YOLO(model_name)
        self._confidence = confidence
        self._device = device or _auto_device()

    def track(self, video_path: Path) -> list[PlayerDetection]:
        """
        Run tracking on a single video clip.

        Returns a flat list of PlayerDetection across all frames.
        The list is sorted by (frame_idx, track_id).
        """
        results = self._model.track(
            source=str(video_path),
            classes=[self.PERSON_CLASS_ID],
            conf=self._confidence,
            tracker="bytetrack.yaml",
            device=self._device,
            verbose=False,
            stream=True,  # generator — avoids loading all frames into RAM
        )

        detections: list[PlayerDetection] = []
        for frame_idx, result in enumerate(results):
            if result.boxes is None:
                continue
            boxes = result.boxes
            if boxes.id is None:
                # ByteTrack occasionally returns frames with no tracks yet
                continue

            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for track_id, (x1, y1, x2, y2), conf in zip(ids, xyxy, confs):
                detections.append(
                    PlayerDetection(
                        frame_idx=frame_idx,
                        track_id=int(track_id),
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=float(conf),
                    )
                )

        return sorted(detections, key=lambda d: (d.frame_idx, d.track_id))


def summarise_tracks(
    detections: list[PlayerDetection],
    fps: float,
    sample_count: int = 5,
) -> list[dict]:
    """
    Collapse a flat detection list into one summary dict per unique track_id.

    Each summary contains:
      track_id        — integer track identifier (clip-local)
      num_frames      — how many frames this track appears in
      duration_sec    — approximate on-screen duration
      bbox_samples    — up to `sample_count` evenly-spaced bbox snapshots for the identity UI
                        Each sample: {frame_idx, bbox: [x1,y1,x2,y2]}
    """
    from collections import defaultdict

    by_track: dict[int, list[PlayerDetection]] = defaultdict(list)
    for d in detections:
        by_track[d.track_id].append(d)

    summaries: list[dict] = []
    for track_id in sorted(by_track):
        track_dets = sorted(by_track[track_id], key=lambda d: d.frame_idx)
        n = len(track_dets)

        # Pick evenly-spaced indices for thumbnail samples
        if n <= sample_count:
            sample_indices = list(range(n))
        else:
            step = (n - 1) / (sample_count - 1)
            sample_indices = [round(i * step) for i in range(sample_count)]

        bbox_samples = [
            {
                "frame_idx": track_dets[i].frame_idx,
                "bbox": [
                    track_dets[i].x1,
                    track_dets[i].y1,
                    track_dets[i].x2,
                    track_dets[i].y2,
                ],
            }
            for i in sample_indices
        ]

        summaries.append(
            {
                "track_id": track_id,
                "num_frames": n,
                "duration_sec": round(n / fps, 2),
                "bbox_samples": bbox_samples,
            }
        )

    return summaries


def _auto_device() -> str:
    """Pick the best available device without requiring torch at import time."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
