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
    checkpoint: Annotated[Optional[Path], typer.Option(
        help="Trained VideoMAE checkpoint dir for action spotting. "
             "If omitted, action spotting is skipped."
    )] = None,
    action_threshold: Annotated[float, typer.Option(
        help="Minimum confidence for action detections (default 0.5)."
    )] = 0.5,
    filter_action: Annotated[Optional[str], typer.Option(
        help="Only include rallies containing this action (e.g. 'spike')."
    )] = None,
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
                f"Model not found at {model}.\n"
                "Run `clip-maker download-model` first.",
                err=True,
            )
            raise typer.Exit(1)

    typer.echo(f"Video     : {video}")
    typer.echo(f"Output    : {output_dir}")
    typer.echo(f"Model     : {model}")
    typer.echo(f"GPU       : {gpu}")

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
    )

    detections: list = []
    with tqdm(total=total_frames, unit="frame", desc="Tracking") as pbar:
        for det in tracker.track(video):
            detections.append(det)
            pbar.update(1)

    detected = sum(1 for d in detections if d is not None)
    typer.echo(f"  Ball detected in {detected}/{len(detections)} frames "
               f"({100*detected/max(len(detections),1):.1f}%)")

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

    # ── Step 5: action spotting (optional) ──────────────────────────────────
    if checkpoint is not None:
        if not checkpoint.exists():
            typer.echo(f"Warning: checkpoint not found at {checkpoint} — skipping action spotting.",
                       err=True)
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
                clip.actions = [e.to_dict() for e in events]

            # Filter rallies by action if requested
            if filter_action is not None:
                before = len(clips)
                clips = [c for c in clips if any(
                    a["label"] == filter_action for a in c.actions
                )]
                typer.echo(
                    f"\nFiltered to rallies containing '{filter_action}': "
                    f"{len(clips)}/{before}"
                )

            # Re-write manifest with actions (and filtered clips if applicable)
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
        "ffmpeg", "-y", "-i", str(clip_path),
        "-vf", "scale=-2:224",
        "-q:v", "2",
        str(frames_dir / "%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Frame extraction failed for {clip_path.name}:\n{result.stderr[-1000:]}")
    count = len(list(frames_dir.glob("*.jpg")))
    typer.echo(f"  {count} frames → {frames_dir}")
    return count


@app.command()
def label(
    video: Annotated[Path, typer.Argument(help="Input video file.")],
    data_dir: Annotated[Path, typer.Argument(
        help="Output directory. Receives clips/, frames_224p/, and labels.json."
    )],
    output: Annotated[Optional[Path], typer.Option(
        help="Labels JSON path (default: <data_dir>/labels.json)."
    )] = None,
    port: Annotated[int, typer.Option(help="Port for the labeling tool.")] = 8000,
    config: Annotated[Optional[Path], typer.Option(help="TOML config file.")] = None,
    model: Annotated[Optional[Path], typer.Option(help="Path to ONNX model file.")] = None,
    gpu: Annotated[bool, typer.Option(help="Use GPU (CUDA) for ball tracking.")] = False,
) -> None:
    """Detect rallies, extract frames, and open the labeling tool."""
    if not video.exists():
        typer.echo(f"Error: video file not found: {video}", err=True)
        raise typer.Exit(1)

    cfg        = _load_config(config)
    clips_dir  = data_dir / "clips"
    frames_root = data_dir / "frames_224p"
    output     = output or data_dir / "labels.json"

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
    )
    detections: list = []
    with tqdm(total=total_frames, unit="frame", desc="Tracking") as pbar:
        for det in tracker.track(video):
            detections.append(det)
            pbar.update(1)

    detected = sum(1 for d in detections if d is not None)
    typer.echo(f"  Ball detected in {detected}/{len(detections)} frames "
               f"({100*detected/max(len(detections),1):.1f}%)")

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
    subprocess.run([
        sys.executable, str(labeler),
        "--clips-dir", str(clips_dir),
        "--output", str(output),
        "--port", str(port),
    ])


@app.command("download-model")
def download_model_cmd(
    dest: Annotated[Path, typer.Option(help="Directory to save the model.")] = Path("models"),
) -> None:
    """Download the default ONNX ball-tracking model."""
    path = download_model(dest)
    typer.echo(f"Model ready: {path}")


if __name__ == "__main__":
    app()
