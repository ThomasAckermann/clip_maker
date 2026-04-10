# clip-maker

Automatic volleyball rally extractor and action spotter. Feed it a match video and it outputs one MP4 clip per rally, each annotated with the actions that occurred (serve, spike, block, receive, set, score). Rallies can be filtered by action type.

## How it works

```text
Long match video
    │
    ▼
Ball tracker (VballNet ONNX)     — detects the ball in each frame
    │
    ▼
Rally segmenter                  — groups ball detections into rally boundaries
    │
    ▼
Clip extractor (FFmpeg)          — cuts one MP4 per rally
    │
    ▼
Action spotter (VideoMAE)        — classifies actions within each clip  [optional]
    │
    ▼
manifest.json                    — index of clips + detected actions
```

## Installation

Requires Python 3.11+ and FFmpeg on your PATH.

```bash
git clone <repo>
cd clip_maker
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
clip-maker download-model
```

For GPU inference (CUDA):

```bash
pip install -e ".[gpu]"
```

---

## Usage

### Rally extraction

```bash
clip-maker run match.mp4 ./clips
```

Outputs `./clips/rally_001.mp4`, `rally_002.mp4`, … and a `manifest.json`.

### Rally extraction + action spotting

Requires a trained checkpoint (see Training below).

```bash
clip-maker run match.mp4 ./clips --checkpoint models/videomae-vnl-lora/best
```

### Filter by action

Only keep rallies that contain a specific action:

```bash
clip-maker run match.mp4 ./clips \
  --checkpoint models/videomae-vnl-lora/best \
  --filter-action spike
```

Valid actions: `serve`, `spike`, `block`, `receive`, `set`, `score`

### All options

```text
clip-maker run [OPTIONS] VIDEO OUTPUT_DIR

  VIDEO                   Input video file
  OUTPUT_DIR              Directory for output clips

Options:
  --checkpoint PATH       Trained VideoMAE checkpoint for action spotting
  --action-threshold      Minimum detection confidence (default 0.5)
  --filter-action TEXT    Only keep rallies containing this action
  --config PATH           TOML config file (see config.toml)
  --model PATH            Custom ONNX ball-tracking model
  --gpu                   Use CUDA for ball tracking
```

### manifest.json

Without action spotting:

```json
[
  {
    "index": 1,
    "filename": "rally_001.mp4",
    "start_sec": 10.5,
    "end_sec": 28.9,
    "duration_sec": 18.4,
    "start_frame": 262,
    "end_frame": 722,
    "actions": []
  }
]
```

With action spotting:

```json
[
  {
    "filename": "rally_001.mp4",
    "duration_sec": 18.4,
    "actions": [
      {"frame": 30,  "sec": 1.20, "label": "serve",  "confidence": 0.96},
      {"frame": 217, "sec": 8.68, "label": "spike",  "confidence": 0.91},
      {"frame": 240, "sec": 9.60, "label": "block",  "confidence": 0.88}
    ]
  }
]
```

---

## Training

The action spotter is a fine-tuned [VideoMAE](https://huggingface.co/MCG-NJU/videomae-base) model trained on labeled volleyball rally clips. Training uses LoRA adapters for efficiency.

### 1. Collect and label data

Extract rally clips from your match footage:

```bash
clip-maker run match.mp4 ./clips
```

Label each clip using the browser-based labeling tool:

```bash
pip install -e ".[label]"
python tools/labeler.py --video clips/rally_001.mp4 --output data/my_labels.json
# repeat for each clip — output is appended each time
```

**Labeling guide:**

- Mark the single frame where an action *peaks* (ball contact moment)
- Click on the ball position in the frame
- Select the action label from the dropdown
- Do not label `background` — it is sampled automatically from unlabeled regions
- Precision matters by roughly ±4 frames

**Keyboard shortcuts in the labeler:**

| Key | Action |
| --- | --- |
| `Space` | Play / pause |
| `←` / `→` | Step one frame |
| `Shift+←` / `Shift+→` | Jump 1 second |
| `L` | Focus label dropdown |

### 2. Extract frames

The training pipeline reads pre-extracted JPEG frames, not video files directly. The `video` field in the label JSON must match a folder under `data/frames_224p/`.

```bash
# The folder name must match the video filename stem used during labeling
mkdir -p data/frames_224p/rally_001
ffmpeg -i clips/rally_001.mp4 -vf "scale=-2:224" -q:v 2 \
  data/frames_224p/rally_001/%06d.jpg
```

### 3. Split into train/val

Copy or symlink entries from your labels JSON into `data/train.json` and `data/val.json`. A typical split is 80/20 by rally.

### 4. Train

Install training dependencies:

```bash
pip install -e ".[train]"
```

**On CUDA (recommended):**

```bash
python -m training.train \
  --data-dir data \
  --output-dir models/videomae-vnl-lora \
  --adapter lora \
  --epochs 15 \
  --batch-size 8 \
  --lr 2e-4 \
  --merge-adapter
```

**On Apple Silicon (MPS):** use `--batch-size 1 --grad-accum 8` to stay within memory limits:

```bash
python -m training.train \
  --data-dir data \
  --output-dir models/videomae-vnl-lora \
  --adapter lora \
  --epochs 15 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 2e-4 \
  --merge-adapter
```

Training saves two checkpoints:

- `models/videomae-vnl-lora/best/` — best validation accuracy (adapter weights, ~2.5 MB)
- `models/videomae-vnl-lora/final_merged/` — final epoch, adapter merged into base model (~344 MB, no PEFT dependency at inference)

### Training options

| Option | Default | Description |
| --- | --- | --- |
| `--adapter` | `none` | `lora`, `ia3`, or `none` (full fine-tune) |
| `--lora-r` | `16` | LoRA rank. Higher = more capacity |
| `--lora-alpha` | `32` | LoRA scaling. Keep `alpha = 2 × r` |
| `--lora-dropout` | `0.05` | Dropout on LoRA layers |
| `--lora-target-modules` | `query value` | Which attention layers to inject LoRA into |
| `--merge-adapter` | off | Merge LoRA weights into base model after training |
| `--grad-accum` | `1` | Gradient accumulation steps |
| `--batch-size` | `8` | Per-step batch size |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate (`2e-4` recommended for LoRA) |

### 5. Run inference with your trained model

```bash
clip-maker run match.mp4 ./clips \
  --checkpoint models/videomae-vnl-lora/best
```

Use the merged checkpoint if you want to run without the `peft` library installed:

```bash
clip-maker run match.mp4 ./clips \
  --checkpoint models/videomae-vnl-lora/final_merged
```

---

## Action classes

| Label | Description |
| --- | --- |
| `serve` | Player contacts ball to start the rally |
| `spike` | Attacking hit from above the net |
| `block` | Defensive contact at the net |
| `receive` | First contact after opponent's attack |
| `set` | Setter's overhead contact to set up attack |
| `score` | Ball hits floor or goes out |
| `background` | No action (auto-generated, do not label manually) |

---

## Project structure

```text
clip_maker/
  cli.py          — Typer CLI (clip-maker run, download-model)
  tracker.py      — Ball detection via VballNet ONNX model
  segmenter.py    — Rally boundary detection
  extractor.py    — FFmpeg clip extraction
  classifier.py   — Action spotting inference (sliding window + NMS)
training/
  dataset.py      — VNLDataset for VideoMAE fine-tuning
  train.py        — Training loop (LoRA, gradient accumulation, AMP)
tools/
  labeler.py      — Browser-based video event labeling tool
models/           — ONNX ball-tracking model + VideoMAE checkpoints
data/
  frames_224p/    — Pre-extracted JPEG frames for training
  train.json      — Training labels
  val.json        — Validation labels
```
