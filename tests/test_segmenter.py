"""Tests for rally boundary detection logic."""

import warnings

import pytest

from clip_maker.segmenter import Rally, detect_rallies
from clip_maker.tracker import Detection


# ── Helpers ──────────────────────────────────────────────────────────────────

FPS = 25.0


def _det(frame_idx: int) -> Detection:
    return Detection(frame_idx=frame_idx, x=100.0, y=100.0, confidence=0.9)


def _frames(total: int, detected: set[int]) -> list[Detection | None]:
    return [_det(i) if i in detected else None for i in range(total)]


# ── Rally dataclass ───────────────────────────────────────────────────────────


class TestRally:
    def test_duration_frames(self) -> None:
        r = Rally(start_frame=10, end_frame=19)
        assert r.duration_frames() == 10

    def test_duration_sec(self) -> None:
        r = Rally(start_frame=0, end_frame=24)
        assert r.duration_sec(FPS) == pytest.approx(1.0)

    def test_single_frame_duration(self) -> None:
        r = Rally(start_frame=5, end_frame=5)
        assert r.duration_frames() == 1


# ── detect_rallies ────────────────────────────────────────────────────────────


class TestDetectRallies:
    def test_empty_returns_no_rallies(self) -> None:
        assert detect_rallies([], FPS) == []

    def test_all_none_returns_no_rallies(self) -> None:
        assert detect_rallies([None] * 100, FPS) == []

    def test_single_long_rally(self) -> None:
        # 5 seconds of continuous detections at 25 fps = 125 frames
        detected = set(range(125))
        dets = _frames(125, detected)
        rallies = detect_rallies(dets, FPS, min_duration_sec=3.0)
        assert len(rallies) == 1
        assert rallies[0].start_frame == 0
        assert rallies[0].end_frame == 124

    def test_short_rally_discarded(self) -> None:
        # 1 second rally — below default min_duration_sec=3.0
        detected = set(range(25))
        dets = _frames(100, detected)
        rallies = detect_rallies(dets, FPS, min_duration_sec=3.0)
        assert rallies == []

    def test_two_rallies_separated_by_long_gap(self) -> None:
        # Rally 1: frames 0-99 (4 s), gap: frames 100-149 (2 s > gap_threshold 1.5 s)
        # Rally 2: frames 150-249 (4 s)
        detected = set(range(100)) | set(range(150, 250))
        dets = _frames(250, detected)
        rallies = detect_rallies(dets, FPS, gap_threshold_sec=1.5, min_duration_sec=3.0)
        assert len(rallies) == 2
        assert rallies[0].start_frame == 0
        assert rallies[0].end_frame == 99
        assert rallies[1].start_frame == 150
        assert rallies[1].end_frame == 249

    def test_short_gap_is_bridged(self) -> None:
        # Gap of 1 s (25 frames) is below gap_threshold_sec=1.5 s → single rally
        detected = set(range(100)) | set(range(125, 225))
        dets = _frames(225, detected)
        rallies = detect_rallies(dets, FPS, gap_threshold_sec=1.5, min_duration_sec=3.0)
        assert len(rallies) == 1

    def test_rally_at_end_of_video_is_closed(self) -> None:
        # Detection runs all the way to the last frame
        detected = set(range(100, 200))
        dets = _frames(200, detected)
        rallies = detect_rallies(dets, FPS, min_duration_sec=3.0)
        assert len(rallies) == 1
        assert rallies[0].end_frame == 199

    def test_oversized_rally_emits_warning(self) -> None:
        # 130 s rally > default max_duration_sec=120 s
        n_frames = int(130 * FPS)
        detected = set(range(n_frames))
        dets = _frames(n_frames, detected)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rallies = detect_rallies(dets, FPS, min_duration_sec=3.0, max_duration_sec=120.0)
        assert len(rallies) == 1  # still returned
        assert any("missed boundary" in str(w.message) for w in caught)

    def test_sorted_by_start_frame(self) -> None:
        detected = set(range(0, 100)) | set(range(200, 300))
        dets = _frames(300, detected)
        rallies = detect_rallies(dets, FPS, gap_threshold_sec=1.5, min_duration_sec=3.0)
        starts = [r.start_frame for r in rallies]
        assert starts == sorted(starts)
