"""
clip-maker CLI

Commands:
  run            Process a video and extract rally clips.
  download-model Download the default ONNX ball-tracking model.

Example:
  clip-maker download-model
  clip-maker run match.mp4 ./clips
  clip-maker run match.mp4 ./clips --config myconfig.toml --gpu
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Annotated, Optional

import typer
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

    typer.echo(f"\nDone. {len(clips)} clip(s) written to {output_dir}/")
    typer.echo(f"Manifest : {output_dir}/manifest.json")

    for c in clips:
        typer.echo(f"  {c.filename}  ({c.duration_sec:.1f} s)")


@app.command("download-model")
def download_model_cmd(
    dest: Annotated[Path, typer.Option(help="Directory to save the model.")] = Path("models"),
) -> None:
    """Download the default ONNX ball-tracking model."""
    path = download_model(dest)
    typer.echo(f"Model ready: {path}")


if __name__ == "__main__":
    app()
