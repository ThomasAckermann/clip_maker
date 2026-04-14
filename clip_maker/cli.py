"""
clip-maker CLI

Commands:
  run            Process a video and extract rally clips.
  label          Detect rallies, extract frames, and open the labeling tool.
  download-model Download the default ONNX ball-tracking model.

Example:
  clip-maker download-model
  clip-maker run match.mp4 ./clips
  clip-maker label match.mp4 ./data
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Annotated, Optional

import typer
import json
from tqdm import tqdm

from .extractor import extract_clips, get_video_info
from .segmenter import detect_rallies
from .tracker import BallTracker, DEFAULT_MODEL_NAME, download_model

app = typer.Typer(help="Volleyball rally clip extractor.", add_completion=False)

# ── Default config values ────────────────────────────────────────────────────

DEFAULTS = {
    "heatmap_threshold": 0.5,
    "gap_threshold_sec": 1.5,
    "min_rally_duration_sec": 3.0,
    "max_rally_duration_sec": 120.0,
    "pre_padding_sec": 1.5,
    "post_padding_sec": 1.5,
}


def _load_config(config_path: Path | None) -> dict:
    cfg = dict(DEFAULTS)
    if config_path is not None:
        with open(config_path, "rb") as f:
            user_cfg = tomllib.load(f)
        cfg.update(user_cfg.get("clip_maker", {}))
    return cfg


# ── Commands ─────────────────────────────────────────────────────────────────


@app.command()
def run(
    video: Annotated[Path, typer.Argument(help="Input video file.")],
    output_dir: Annotated[Path, typer.Argument(help="Directory for output clips.")],
    config: Annotated[Optional[Path], typer.Option(help="TOML config file.")] = None,
    model: Annotated[Optional[Path], typer.Option(help="Path to ONNX model file.")] = None,
    gpu: Annotated[bool, typer.Option(help="Use GPU (CUDA) for inference.")] = False,
    checkpoint: Annotated[
        Optional[Path],
        typer.Option(
            help="Trained VideoMAE checkpoint dir for action spotting. "
            "If omitted, action spotting is skipped."
        ),
    ] = None,
    action_threshold: Annotated[
        float, typer.Option(help="Minimum confidence for action detections (default 0.5).")
    ] = 0.5,
    filter_action: Annotated[
        Optional[str],
        typer.Option(help="Only include rallies containing this action (e.g. 'spike')."),
    ] = None,
    stride: Annotated[
        int,
        typer.Option(
            help="Run ball-tracker inference every N frames (default 1 = every frame). "
            "stride=2–4 halves/quarters tracking time with negligible quality loss for "
            "rally segmentation, since gaps are ~45 frames at 30 fps."
        ),
    ] = 1,
    track_players: Annotated[
        bool,
        typer.Option(
            help="Run YOLOv8+ByteTrack player tracking on each clip. "
            "Requires: pip install 'clip-maker[track]'. "
            "Adds player_tracks to the manifest and (if --checkpoint is set) "
            "associates each action with the nearest player."
        ),
    ] = False,
    yolo_model: Annotated[
        str,
        typer.Option(
            help="YOLOv8 model name or path for player tracking (default: yolov8n.pt)."
        ),
    ] = "yolov8n.pt",
) -> None:
    """Process a video and extract one clip per detected rally."""
    if not video.exists():
        typer.echo(f"Error: video file not found: {video}", err=True)
        raise typer.Exit(1)

    cfg = _load_config(config)

    # Resolve model path
    if model is None:
        models_dir = Path(__file__).parent.parent / "models"
        model = models_dir / DEFAULT_MODEL_NAME
        if not model.exists():
            typer.echo(
                f"Model not found at {model}.\nRun `clip-maker download-model` first.",
                err=True,
            )
            raise typer.Exit(1)

    typer.echo(f"Video     : {video}")
    typer.echo(f"Output    : {output_dir}")
    typer.echo(f"Model     : {model}")
    typer.echo(f"GPU       : {gpu}")
    if track_players:
        typer.echo(f"YOLO      : {yolo_model}")

    # ── Step 1: read video metadata ──────────────────────────────────────────
    typer.echo("\nReading video metadata…")
    fps, video_duration = get_video_info(video)
    total_frames = int(fps * video_duration)
    typer.echo(f"  {fps:.2f} fps  |  {video_duration:.1f} s  |  ~{total_frames} frames")

    # ── Step 2: ball tracking ────────────────────────────────────────────────
    typer.echo("\nRunning ball tracker…")
    tracker = BallTracker(
        model_path=model,
        heatmap_threshold=cfg["heatmap_threshold"],
        use_gpu=gpu,
        stride=stride,
    )

    detections: list = []
    with tqdm(total=total_frames, unit="frame", desc="Tracking") as pbar:
        for det in tracker.track(video):
            detections.append(det)
            pbar.update(1)

    detected = sum(1 for d in detections if d is not None)
    typer.echo(
        f"  Ball detected in {detected}/{len(detections)} frames "
        f"({100 * detected / max(len(detections), 1):.1f}%)"
    )

    # ── Step 3: rally segmentation ───────────────────────────────────────────
    typer.echo("\nDetecting rally boundaries…")
    rallies = detect_rallies(
        detections=detections,
        fps=fps,
        gap_threshold_sec=cfg["gap_threshold_sec"],
        min_duration_sec=cfg["min_rally_duration_sec"],
        max_duration_sec=cfg["max_rally_duration_sec"],
    )
    typer.echo(f"  Found {len(rallies)} rally candidate(s)")

    if not rallies:
        typer.echo("No rallies detected. Try lowering --heatmap-threshold or gap_threshold_sec.")
        raise typer.Exit(0)

    # ── Step 4: clip extraction ──────────────────────────────────────────────
    typer.echo(f"\nExtracting clips → {output_dir}")
    clips = extract_clips(
        video_path=video,
        rallies=rallies,
        output_dir=output_dir,
        fps=fps,
        pre_padding_sec=cfg["pre_padding_sec"],
        post_padding_sec=cfg["post_padding_sec"],
        video_duration_sec=video_duration,
    )

    # ── Step 5: player tracking (optional) ──────────────────────────────────
    # Maps clip filename → flat list of PlayerDetection (for action association below)
    clip_player_detections: dict[str, list] = {}
    if track_players:
        typer.echo(f"\nRunning player tracker ({yolo_model}) …")
        from .player_tracker import PlayerTracker, summarise_tracks

        player_tracker = PlayerTracker(model_name=yolo_model)
        for clip in clips:
            typer.echo(f"  {clip.filename}")
            clip_path = output_dir / clip.filename
            pdet = player_tracker.track(clip_path)
            clip_player_detections[clip.filename] = pdet
            clip.player_tracks = summarise_tracks(pdet, fps)
        typer.echo(f"  Player tracking done for {len(clips)} clip(s)")

    # ── Step 6: action spotting (optional) ──────────────────────────────────
    if checkpoint is not None:
        if not checkpoint.exists():
            typer.echo(
                f"Warning: checkpoint not found at {checkpoint} — skipping action spotting.",
                err=True,
            )
        else:
            typer.echo(f"\nRunning action spotter (checkpoint: {checkpoint}) …")
            from .classifier import ActionSpotter

            spotter = ActionSpotter(
                checkpoint_dir=checkpoint,
                threshold=action_threshold,
            )
            for clip in clips:
                typer.echo(f"\n{clip.filename}")
                events = spotter.spot(output_dir / clip.filename)
                raw_actions = [e.to_dict() for e in events]

                # Associate actions with players if tracking was run
                if track_players and clip.filename in clip_player_detections:
                    from .associator import associate_actions_to_players

                    # Ball detections are match-level; slice to the clip's frame range
                    clip_start_frame = clip.start_frame
                    clip_end_frame = clip.end_frame
                    clip_ball = detections[clip_start_frame : clip_end_frame + 1]
                    raw_actions = associate_actions_to_players(
                        actions=raw_actions,
                        ball_detections=clip_ball,
                        player_detections=clip_player_detections[clip.filename],
                    )

                clip.actions = raw_actions

            # Filter rallies by action if requested
            if filter_action is not None:
                before = len(clips)
                clips = [c for c in clips if any(a["label"] == filter_action for a in c.actions)]
                typer.echo(
                    f"\nFiltered to rallies containing '{filter_action}': {len(clips)}/{before}"
                )

    # ── Write final manifest ─────────────────────────────────────────────────
    if track_players or checkpoint is not None:
        import dataclasses

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps([dataclasses.asdict(c) for c in clips], indent=2),
            encoding="utf-8",
        )

    typer.echo(f"\nDone. {len(clips)} clip(s) written to {output_dir}/")
    typer.echo(f"Manifest : {output_dir}/manifest.json")

    for c in clips:
        action_str = ""
        if c.actions:
            labels = ", ".join(a["label"] for a in c.actions)
            action_str = f"  [{labels}]"
        typer.echo(f"  {c.filename}  ({c.duration_sec:.1f} s){action_str}")


def _extract_frames(clip_path: Path, frames_dir: Path) -> int:
    """
    Extract frames from a clip into frames_dir at 224p height.
    Skips if frames already exist. Returns the frame count.
    """
    if frames_dir.exists() and any(frames_dir.glob("*.jpg")):
        count = len(list(frames_dir.glob("*.jpg")))
        typer.echo(f"  Skipped (already extracted: {count} frames)")
        return count

    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(clip_path),
        "-vf",
        "scale=-2:224",
        "-q:v",
        "2",
        str(frames_dir / "%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Frame extraction failed for {clip_path.name}:\n{result.stderr[-1000:]}"
        )
    count = len(list(frames_dir.glob("*.jpg")))
    typer.echo(f"  {count} frames → {frames_dir}")
    return count


@app.command()
def label(
    video: Annotated[Path, typer.Argument(help="Input video file.")],
    data_dir: Annotated[
        Path,
        typer.Argument(help="Output directory. Receives clips/, frames_224p/, and labels.json."),
    ],
    output: Annotated[
        Optional[Path], typer.Option(help="Labels JSON path (default: <data_dir>/labels.json).")
    ] = None,
    port: Annotated[int, typer.Option(help="Port for the labeling tool.")] = 8000,
    config: Annotated[Optional[Path], typer.Option(help="TOML config file.")] = None,
    model: Annotated[Optional[Path], typer.Option(help="Path to ONNX model file.")] = None,
    gpu: Annotated[bool, typer.Option(help="Use GPU (CUDA) for ball tracking.")] = False,
    stride: Annotated[
        int,
        typer.Option(
            help="Run ball-tracker inference every N frames (default 1). "
            "stride=2–4 is recommended on MacBook for faster processing."
        ),
    ] = 1,
) -> None:
    """Detect rallies, extract frames, and open the labeling tool."""
    if not video.exists():
        typer.echo(f"Error: video file not found: {video}", err=True)
        raise typer.Exit(1)

    cfg = _load_config(config)
    clips_dir = data_dir / "clips"
    frames_root = data_dir / "frames_224p"
    output = output or data_dir / "labels.json"

    # Resolve ONNX model
    if model is None:
        models_dir = Path(__file__).parent.parent / "models"
        model = models_dir / DEFAULT_MODEL_NAME
        if not model.exists():
            typer.echo(
                f"Model not found at {model}.\nRun `clip-maker download-model` first.",
                err=True,
            )
            raise typer.Exit(1)

    # ── Step 1: ball tracking ────────────────────────────────────────────────
    typer.echo("\nReading video metadata…")
    fps, video_duration = get_video_info(video)
    total_frames = int(fps * video_duration)
    typer.echo(f"  {fps:.2f} fps  |  {video_duration:.1f} s  |  ~{total_frames} frames")

    typer.echo("\nRunning ball tracker…")
    tracker = BallTracker(
        model_path=model,
        heatmap_threshold=cfg["heatmap_threshold"],
        use_gpu=gpu,
        stride=stride,
    )
    detections: list = []
    with tqdm(total=total_frames, unit="frame", desc="Tracking") as pbar:
        for det in tracker.track(video):
            detections.append(det)
            pbar.update(1)

    detected = sum(1 for d in detections if d is not None)
    typer.echo(
        f"  Ball detected in {detected}/{len(detections)} frames "
        f"({100 * detected / max(len(detections), 1):.1f}%)"
    )

    # ── Step 2: rally segmentation ───────────────────────────────────────────
    typer.echo("\nDetecting rally boundaries…")
    rallies = detect_rallies(
        detections=detections,
        fps=fps,
        gap_threshold_sec=cfg["gap_threshold_sec"],
        min_duration_sec=cfg["min_rally_duration_sec"],
        max_duration_sec=cfg["max_rally_duration_sec"],
    )
    typer.echo(f"  Found {len(rallies)} rally candidate(s)")

    if not rallies:
        typer.echo("No rallies detected. Try adjusting gap_threshold_sec in config.")
        raise typer.Exit(0)

    # ── Step 3: extract rally clips ──────────────────────────────────────────
    typer.echo(f"\nExtracting clips → {clips_dir}")
    clips = extract_clips(
        video_path=video,
        rallies=rallies,
        output_dir=clips_dir,
        fps=fps,
        pre_padding_sec=cfg["pre_padding_sec"],
        post_padding_sec=cfg["post_padding_sec"],
        video_duration_sec=video_duration,
    )

    # ── Step 4: extract frames ───────────────────────────────────────────────
    typer.echo(f"\nExtracting frames → {frames_root}")
    for clip in clips:
        typer.echo(f"  {clip.filename}")
        _extract_frames(
            clip_path=clips_dir / clip.filename,
            frames_dir=frames_root / clip.filename.replace(".mp4", ""),
        )

    # ── Step 5: launch labeling tool ─────────────────────────────────────────
    typer.echo(f"\nStarting labeling tool on http://localhost:{port} …")
    typer.echo(f"  Clips : {clips_dir}  ({len(clips)} clips)")
    typer.echo(f"  Output: {output}")
    typer.echo("  Press Ctrl+C to stop.\n")

    labeler = Path(__file__).parent.parent / "tools" / "labeler.py"
    subprocess.run(
        [
            sys.executable,
            str(labeler),
            "--clips-dir",
            str(clips_dir),
            "--output",
            str(output),
            "--port",
            str(port),
        ]
    )


@app.command()
def highlight(
    manifest: Annotated[Path, typer.Argument(help="manifest.json produced by `clip-maker run`.")],
    output: Annotated[Path, typer.Argument(help="Output highlight reel MP4.")],
    player: Annotated[
        Optional[str],
        typer.Option(help="Player name to filter by (must match identities.json)."),
    ] = None,
    track_id: Annotated[
        Optional[int],
        typer.Option(help="Raw track_id to filter by (skips identities.json lookup)."),
    ] = None,
    identities: Annotated[
        Optional[Path],
        typer.Option(
            help="JSON file mapping track_ids to player names. "
            "Defaults to identities.json in the same directory as the manifest."
        ),
    ] = None,
    pad_before: Annotated[
        float, typer.Option(help="Seconds to include before each action (default 2.0).")
    ] = 2.0,
    pad_after: Annotated[
        float, typer.Option(help="Seconds to include after each action (default 3.0).")
    ] = 3.0,
) -> None:
    """Stitch a highlight reel for one player from an annotated manifest."""
    if not manifest.exists():
        typer.echo(f"Error: manifest not found: {manifest}", err=True)
        raise typer.Exit(1)

    if player is None and track_id is None:
        typer.echo("Error: provide --player NAME or --track-id ID.", err=True)
        raise typer.Exit(1)

    clips_dir = manifest.parent
    data = json.loads(manifest.read_text(encoding="utf-8"))

    # ── Resolve identities ───────────────────────────────────────────────────
    id_path = identities or (clips_dir / "identities.json")
    identity_map: dict[str, list[int]] = {}  # player_name → [track_ids across rallies]
    if id_path.exists():
        raw_ids = json.loads(id_path.read_text(encoding="utf-8"))
        # Format: {rally_filename: {str(track_id): player_name}}
        for tracks in raw_ids.values():
            for tid_str, name in tracks.items():
                identity_map.setdefault(name, []).append(int(tid_str))

    # Resolve track_ids to filter on
    target_track_ids: set[int] | None = None
    if track_id is not None:
        target_track_ids = {track_id}
    elif player is not None:
        if player not in identity_map:
            typer.echo(
                f"Error: player '{player}' not found in {id_path}.\n"
                f"Available: {', '.join(sorted(identity_map)) or '(none)'}",
                err=True,
            )
            raise typer.Exit(1)
        target_track_ids = set(identity_map[player])

    # ── Collect action windows ───────────────────────────────────────────────
    import tempfile

    segments: list[tuple[Path, float, float]] = []  # (clip_path, start_sec, end_sec)

    for clip_data in data:
        clip_path = clips_dir / clip_data["filename"]
        if not clip_path.exists():
            typer.echo(f"  Warning: clip not found, skipping: {clip_path.name}", err=True)
            continue

        clip_fps_approx = (clip_data["end_frame"] - clip_data["start_frame"]) / max(
            clip_data["duration_sec"], 0.001
        )

        for action in clip_data.get("actions", []):
            # Filter by player if tracking data is present
            if target_track_ids is not None:
                action_tid = action.get("player_track_id")
                if action_tid not in target_track_ids:
                    continue

            action_frame: int = action["frame"]
            action_time = action_frame / max(clip_fps_approx, 1.0)
            seg_start = max(0.0, action_time - pad_before)
            seg_end = min(clip_data["duration_sec"], action_time + pad_after)
            segments.append((clip_path, seg_start, seg_end))

    if not segments:
        typer.echo("No matching action segments found. Check --player / --track-id and manifest.")
        raise typer.Exit(0)

    typer.echo(f"\nBuilding highlight reel: {len(segments)} segment(s) → {output}")

    # ── Cut segments and concat ──────────────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        segment_files: list[Path] = []

        for i, (clip_path, seg_start, seg_end) in enumerate(segments):
            seg_out = tmp_dir / f"seg_{i:04d}.mp4"
            duration = seg_end - seg_start
            _ffmpeg_cut_segment(clip_path, seg_out, seg_start, duration)
            segment_files.append(seg_out)

        _ffmpeg_concat(segment_files, output)

    typer.echo(f"Highlight reel written to {output}  ({len(segments)} clips)")


def _ffmpeg_cut_segment(src: Path, dst: Path, start_sec: float, duration_sec: float) -> None:
    """Cut a short segment, re-encoding to ensure clean concat boundaries."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(src),
        "-t", f"{duration_sec:.3f}",
        # Re-encode to H.264 so all segments have identical codec params for concat
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-avoid_negative_ts", "1",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg segment cut failed for {dst.name}:\n{result.stderr[-2000:]}")


def _ffmpeg_concat(segment_files: list[Path], output: Path) -> None:
    """Concatenate segments using FFmpeg concat demuxer (stream copy — no re-encode)."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as flist:
        for seg in segment_files:
            flist.write(f"file '{seg}'\n")
        flist_path = Path(flist.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(flist_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed:\n{result.stderr[-2000:]}")
    finally:
        flist_path.unlink(missing_ok=True)


@app.command("label-ui")
def label_ui(
    clips_dir: Annotated[Path, typer.Argument(help="Directory of MP4 clips to label.")],
    output: Annotated[
        Optional[Path], typer.Option(help="Labels JSON path (default: <clips_dir>/labels.json).")
    ] = None,
    port: Annotated[int, typer.Option(help="Port for the labeling tool.")] = 8000,
) -> None:
    """Open the labeling UI for an existing clips directory (skips processing)."""
    if not clips_dir.exists():
        typer.echo(f"Error: directory not found: {clips_dir}", err=True)
        raise typer.Exit(1)

    clips = sorted(clips_dir.glob("*.mp4"))
    if not clips:
        typer.echo(f"Error: no MP4 files found in {clips_dir}", err=True)
        raise typer.Exit(1)

    output = output or clips_dir / "labels.json"

    typer.echo(f"Clips  : {clips_dir}  ({len(clips)} clips)")
    typer.echo(f"Output : {output}")
    typer.echo(f"Opening http://localhost:{port} …\n")

    labeler = Path(__file__).parent.parent / "tools" / "labeler.py"
    subprocess.run(
        [
            sys.executable,
            str(labeler),
            "--clips-dir", str(clips_dir),
            "--output", str(output),
            "--port", str(port),
        ]
    )


@app.command("download-model")
def download_model_cmd(
    dest: Annotated[Path, typer.Option(help="Directory to save the model.")] = Path("models"),
) -> None:
    """Download the default ONNX ball-tracking model."""
    path = download_model(dest)
    typer.echo(f"Model ready: {path}")


if __name__ == "__main__":
    app()
