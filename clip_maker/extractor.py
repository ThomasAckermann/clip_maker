"""
Clip extraction via FFmpeg.

For each Rally, cuts a clip from the source video using stream-copy (no
re-encode) for speed.  Applies configurable pre/post padding.

Output:
  <output_dir>/rally_001.mp4
  <output_dir>/rally_002.mp4
  …
  <output_dir>/manifest.json
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .segmenter import Rally


@dataclass
class ClipInfo:
    index: int
    filename: str
    start_sec: float
    end_sec: float
    duration_sec: float
    start_frame: int
    end_frame: int
    actions: list[dict] = field(default_factory=list)


def extract_clips(
    video_path: Path,
    rallies: list[Rally],
    output_dir: Path,
    fps: float,
    pre_padding_sec: float = 1.5,
    post_padding_sec: float = 1.5,
    video_duration_sec: float | None = None,
) -> list[ClipInfo]:
    """
    Extract one MP4 clip per rally using FFmpeg stream-copy.

    Args:
        video_path:        Source video file.
        rallies:           Rally segments (frame indices).
        output_dir:        Directory for output clips + manifest.json.
        fps:               Video frame rate (used for frame→time conversion).
        pre_padding_sec:   Seconds to include before the detected rally start.
        post_padding_sec:  Seconds to include after the detected rally end.
        video_duration_sec: Total video duration; used to clamp the end time.

    Returns:
        List of ClipInfo for the manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    clips: list[ClipInfo] = []

    for i, rally in enumerate(rallies, start=1):
        raw_start = rally.start_frame / fps
        raw_end = rally.end_frame / fps

        start_sec = max(0.0, raw_start - pre_padding_sec)
        end_sec = raw_end + post_padding_sec
        if video_duration_sec is not None:
            end_sec = min(end_sec, video_duration_sec)

        duration = end_sec - start_sec
        filename = f"rally_{i:03d}.mp4"
        out_path = output_dir / filename

        _ffmpeg_extract(video_path, out_path, start_sec, duration)

        clips.append(
            ClipInfo(
                index=i,
                filename=filename,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=duration,
                start_frame=rally.start_frame,
                end_frame=rally.end_frame,
            )
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps([asdict(c) for c in clips], indent=2), encoding="utf-8")

    return clips


def _ffmpeg_extract(src: Path, dst: Path, start_sec: float, duration_sec: float) -> None:
    """
    Use FFmpeg stream-copy to cut a clip without re-encoding.

    -ss before -i seeks quickly to the keyframe before start_sec.
    -to with -copyts preserves correct timestamps in the output.
    Stream copy is near-instant regardless of clip length.
    """
    cmd = [
        "ffmpeg",
        "-y",  # overwrite without prompt
        "-ss",
        f"{start_sec:.3f}",  # fast seek (before input)
        "-i",
        str(src),
        "-t",
        f"{duration_sec:.3f}",
        "-c",
        "copy",  # stream copy — no re-encode
        "-avoid_negative_ts",
        "1",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {dst.name}:\n{result.stderr[-2000:]}")


def get_video_info(video_path: Path) -> tuple[float, float]:
    """
    Return (fps, duration_sec) for a video file using FFmpeg.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")

    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            # fps from avg_frame_rate "30000/1001" or "30/1"
            avg_fr = stream.get("avg_frame_rate", "30/1")
            num, den = avg_fr.split("/")
            den_f = float(den)
            if den_f == 0:
                raise RuntimeError(f"ffprobe returned invalid frame rate: {avg_fr!r}")
            fps = float(num) / den_f
            duration = float(stream.get("duration", 0))
            return fps, duration

    raise RuntimeError("No video stream found in file.")
