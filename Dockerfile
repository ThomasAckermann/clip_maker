# Training image for clip-maker
# Requires: nvidia-container-toolkit on the host
#
# Build:
#   docker build -t clip-maker-train .
#
# Run (mount your data and model output dirs):
#   docker run --gpus all \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     clip-maker-train \
#     python -m training.train \
#       --data-dir data --output-dir models/videomae-vnl-lora \
#       --adapter lora --lora-r 16 --lora-alpha 32 \
#       --epochs 15 --batch-size 8 --grad-accum 1 --lr 2e-4 --merge-adapter

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install ffmpeg (needed by extractor) and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY clip_maker/ clip_maker/
COPY training/ training/
COPY config.toml .

# Install project with training dependencies
RUN pip install --no-cache-dir -e ".[train]"

# Data and model output are mounted at runtime — not baked into the image
VOLUME ["/app/data", "/app/models"]
