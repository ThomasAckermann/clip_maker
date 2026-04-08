"""
Rally boundary detection.

A rally is a contiguous sequence of frames where the ball is being tracked.
A boundary is detected when ball detections are absent for longer than
`gap_threshold` seconds.  Short spurious gaps within a rally (e.g. ball
briefly occluded) are bridged if they are shorter than the threshold.

Output: list of Rally(start_frame, end_frame) in original frame indices.
Filtered by min/max duration so single-frame noise and inter-set breaks
are excluded automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tracker import Detection


@dataclass
class Rally:
    start_frame: int
    end_frame: int  # inclusive

    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def duration_sec(self, fps: float) -> float:
        return self.duration_frames() / fps


def detect_rallies(
    detections: list[Detection | None],
    fps: float,
    gap_threshold_sec: float = 1.5,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 120.0,
) -> list[Rally]:
    """
    Args:
        detections:        One entry per frame; None means no ball detected.
        fps:               Video frame rate.
        gap_threshold_sec: Gaps in detection shorter than this are bridged
                           (ball briefly hidden); gaps longer than this signal
                           a dead-ball period between rallies.
        min_duration_sec:  Rallies shorter than this are discarded (noise).
        max_duration_sec:  Rallies longer than this trigger a warning; they
                           are still kept but likely contain a missed boundary.

    Returns:
        List of Rally objects sorted by start_frame.
    """
    gap_frames = int(gap_threshold_sec * fps)
    min_frames = int(min_duration_sec * fps)
    max_frames = int(max_duration_sec * fps)

    # Step 1: find raw segments where the ball is detected
    # A segment is a maximal run of frames with at least one detection
    # within gap_frames of the previous detection.
    segments: list[tuple[int, int]] = []  # (start, end) inclusive
    seg_start: int | None = None
    last_detected: int | None = None

    for i, det in enumerate(detections):
        if det is not None:
            if seg_start is None:
                seg_start = i
            last_detected = i
        else:
            if last_detected is not None and (i - last_detected) > gap_frames:
                # Gap too long — close the current segment
                segments.append((seg_start, last_detected))
                seg_start = None
                last_detected = None

    # Close any open segment at end of video
    if seg_start is not None and last_detected is not None:
        segments.append((seg_start, last_detected))

    # Step 2: convert segments to Rally objects and apply duration filters
    rallies: list[Rally] = []
    for start, end in segments:
        rally = Rally(start_frame=start, end_frame=end)
        duration = rally.duration_frames()

        if duration < min_frames:
            continue  # too short — discard

        if duration > max_frames:
            import warnings
            warnings.warn(
                f"Rally at frames {start}–{end} is {rally.duration_sec(fps):.1f} s "
                f"(> max {max_duration_sec} s). Check for a missed boundary.",
                stacklevel=2,
            )

        rallies.append(rally)

    return rallies
