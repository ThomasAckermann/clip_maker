"""
Fine-tune VideoMAE-small on the VNL volleyball action dataset.

Usage (from repo root):
  python -m training.train \\
      --data-dir data \\
      --output-dir models/videomae-vnl \\
      [--epochs 10] [--batch-size 8] [--lr 1e-4] [--num-workers 4]

  # Mixed validation: add club-footage splits alongside the VNL val set
  python -m training.train \\
      --data-dir data \\
      --val-extra-json data/club/val.json data/club2/val.json \\
      --val-extra-frames data/club/frames_224p data/club2/frames_224p \\
      --output-dir models/videomae-vnl

The script:
  1. Loads train/val splits from dataset.py
  2. Optionally merges additional val JSON files (e.g. club footage) via --val-extra-json
  3. Downloads MCG-NJU/videomae-small from HuggingFace (cached after first run)
  4. Replaces the classification head with a 7-class head
  5. Fine-tunes with cross-entropy loss + inverse-frequency class weights
  6. Saves the best checkpoint (by val accuracy) to --output-dir
  7. Logs loss / accuracy / LR to TensorBoard (default: runs/<output-dir-name>)

Hardware: designed for an RTX 3070 (8 GB VRAM).
  - With batch_size=8 and gradient checkpointing, VRAM usage ≈ 5–6 GB.
  - Increase batch_size if you have more VRAM; decrease if you OOM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import VideoMAEForVideoClassification

from .dataset import VNLDataset, class_weights, CLASSES

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune VideoMAE on VNL volleyball data")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("models/videomae-vnl"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed-precision (AMP)")
    p.add_argument(
        "--model-id",
        type=str,
        default="MCG-NJU/videomae-base",
        help="HuggingFace model ID to fine-tune from",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps. Use to simulate a larger batch "
        "with less VRAM (e.g. --batch-size 1 --grad-accum 8).",
    )
    # Adapter fine-tuning
    p.add_argument(
        "--adapter",
        type=str,
        choices=["none", "lora", "ia3"],
        default="none",
        help="Adapter fine-tuning mode. 'none' = full fine-tune (default).",
    )
    p.add_argument(
        "--lora-r", type=int, default=16, help="LoRA rank (default 16). Higher = more capacity."
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling. Effective scale = alpha/r (default 32).",
    )
    p.add_argument(
        "--lora-dropout", type=float, default=0.05, help="Dropout on LoRA layers (default 0.05)."
    )
    p.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=["query", "value"],
        help="Linear modules to inject LoRA into (default: query value).",
    )
    p.add_argument(
        "--merge-adapter",
        action="store_true",
        help="After training, merge adapter weights into base model and save "
        "a standalone checkpoint (no PEFT needed at inference).",
    )
    # Mixed validation
    p.add_argument(
        "--val-extra-json",
        type=Path,
        nargs="+",
        default=[],
        metavar="JSON",
        help="Additional val JSON files (e.g. club footage labels.json). "
        "Must be paired 1-to-1 with --val-extra-frames.",
    )
    p.add_argument(
        "--val-extra-frames",
        type=Path,
        nargs="+",
        default=[],
        metavar="FRAMES_ROOT",
        help="frames_224p roots corresponding to each --val-extra-json file.",
    )
    # TensorBoard
    p.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging (enabled by default).",
    )
    p.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="Directory for TensorBoard event files "
        "(default: runs/<output-dir-name>).",
    )
    return p.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────


def print_trainable_parameters(model: nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0.0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


def build_model(
    model_id: str,
    num_classes: int,
    args: argparse.Namespace | None = None,
) -> nn.Module:
    """
    Load VideoMAE and optionally wrap with LoRA or IA3 adapters via PEFT.

    Returns a plain VideoMAEForVideoClassification when args.adapter == 'none',
    or a peft.PeftModel wrapping it otherwise.
    """
    model = VideoMAEForVideoClassification.from_pretrained(
        model_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # head size changes
        label2id={c: i for i, c in enumerate(CLASSES)},
        id2label={i: c for i, c in enumerate(CLASSES)},
    )
    # Enable gradient checkpointing before PEFT wrapping.
    # use_reentrant=False is required for compatibility with PEFT and MPS.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    adapter_mode = getattr(args, "adapter", "none")

    if adapter_mode == "none":
        print_trainable_parameters(model)
        return model

    # Lazy import — peft is only required when --adapter is set.
    try:
        from peft import LoraConfig, IA3Config, get_peft_model, TaskType
    except ImportError:
        raise ImportError(
            "peft is required for adapter fine-tuning. "
            "Install it with: pip install 'clip-maker[train]' or pip install 'peft>=0.10.0'"
        )

    if adapter_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            # Save the 7-class classifier head alongside the adapter weights
            # so it is restored on checkpoint load.
            modules_to_save=["classifier"],
            bias="none",
        )
    else:  # ia3
        peft_config = IA3Config(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["key", "value", "intermediate.dense"],
            feedforward_modules=["intermediate.dense"],
            modules_to_save=["classifier"],
        )

    peft_model = get_peft_model(model, peft_config)
    # Required when gradient checkpointing is active with a frozen base model.
    # Without this, the backward pass through checkpointed blocks loses gradients
    # because no frozen parameter anchors the autograd graph.
    peft_model.enable_input_require_grads()

    print_trainable_parameters(peft_model)
    return peft_model


def save_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> None:
    """
    Save model checkpoint. For adapter modes, saves only the adapter weights
    (~2.5 MB). If --merge-adapter is set, also saves a merged full-model
    checkpoint with no PEFT dependency at inference time.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)

    adapter_mode = getattr(args, "adapter", "none")
    if adapter_mode != "none" and getattr(args, "merge_adapter", False):
        merged_dir = checkpoint_dir.parent / (checkpoint_dir.name + "_merged")
        merged_dir.mkdir(parents=True, exist_ok=True)
        # merge_and_unload() returns a new plain VideoMAEForVideoClassification;
        # it does not modify the in-memory model.
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        print(f"  Merged checkpoint saved to: {merged_dir}")


# ── Training loop ─────────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_accum_steps: int = 1,
    epoch: int = 1,
    total_epochs: int = 1,
    log_interval: int = 10,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_updates = 0

    optimizer.zero_grad()

    for step, (pixel_values, labels) in enumerate(loader):
        # VideoMAE expects (batch, time, channels, height, width)
        pixel_values = pixel_values.permute(0, 1, 2, 3, 4).to(device)  # (B, T, C, H, W)
        labels = labels.to(device)

        is_last_step = (step + 1) == len(loader)
        do_update = ((step + 1) % grad_accum_steps == 0) or is_last_step

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels) / grad_accum_steps
            scaler.scale(loss).backward()
            if do_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels) / grad_accum_steps
            loss.backward()
            if do_update:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        step_loss = loss.item() * grad_accum_steps
        total_loss += step_loss * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if do_update:
            num_updates += 1
            if num_updates % log_interval == 0 or is_last_step:
                running_acc = correct / total
                print(
                    f"  Epoch {epoch}/{total_epochs} | "
                    f"step {step + 1}/{len(loader)} | "
                    f"loss {step_loss:.4f} | "
                    f"acc {running_acc:.3f}",
                    flush=True,
                )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for pixel_values, labels in loader:
        pixel_values = pixel_values.permute(0, 1, 2, 3, 4).to(device)
        labels = labels.to(device)
        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.val_extra_json) != len(args.val_extra_frames):
        raise SystemExit(
            f"--val-extra-json ({len(args.val_extra_json)} files) and "
            f"--val-extra-frames ({len(args.val_extra_frames)} roots) must have the same length."
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if device.type == "mps" and args.batch_size > 1:
        print(
            f"WARNING: MPS (Apple Silicon) detected with --batch-size {args.batch_size}.\n"
            f"  VideoMAE attention maps require ~{args.batch_size * 118:.0f} MB per layer "
            f"({args.batch_size} samples × 12 heads × 1568² tokens).\n"
            f"  Recommended: --batch-size 1 --grad-accum {args.batch_size}\n"
            f"  This keeps the effective batch size the same but fits in MPS memory."
        )

    # ── TensorBoard ──────────────────────────────────────────────────────────
    tb_writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = args.tensorboard_dir or Path("runs") / args.output_dir.name
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"TensorBoard: tensorboard --logdir {tb_dir}")
        except ImportError:
            print(
                "WARNING: tensorboard not installed — skipping TensorBoard logging.\n"
                "  Install it with: pip install 'clip-maker[train]' or pip install tensorboard"
            )

    # ── Datasets & loaders ───────────────────────────────────────────────────
    frames_root = args.data_dir / "frames_224p"
    train_ds = VNLDataset(args.data_dir / "train.json", frames_root, augment=True)
    vnl_val_ds = VNLDataset(args.data_dir / "val.json", frames_root, augment=False)

    extra_val_datasets = [
        VNLDataset(json_path, extra_frames_root, augment=False)
        for json_path, extra_frames_root in zip(args.val_extra_json, args.val_extra_frames)
    ]
    val_ds = ConcatDataset([vnl_val_ds, *extra_val_datasets]) if extra_val_datasets else vnl_val_ds

    if extra_val_datasets:
        print(
            f"Val sources: VNL ({len(vnl_val_ds)}) + "
            + " + ".join(
                f"{p.parent.name} ({len(ds)})"
                for p, ds in zip(args.val_extra_json, extra_val_datasets)
            )
            + f" = {len(val_ds)} total"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading {args.model_id} …")
    model = build_model(args.model_id, num_classes=len(CLASSES), args=args)
    model = model.to(device)

    # ── Loss, optimiser, scheduler ───────────────────────────────────────────
    weights = class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10
    )

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            grad_accum_steps=args.grad_accum,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
        }
        history.append(row)

        flag = " ← best" if val_acc > best_val_acc else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}{flag}"
        )

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("Accuracy/train", train_acc, epoch)
            tb_writer.add_scalar("Accuracy/val", val_acc, epoch)
            tb_writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, args.output_dir / "best", args)
            print(f"  Saved best checkpoint (val_acc={best_val_acc:.3f})")

    # Save final checkpoint and history
    save_checkpoint(model, args.output_dir / "final", args)
    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if tb_writer is not None:
        tb_writer.close()

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
