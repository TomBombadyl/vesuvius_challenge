from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PatchDataset, read_metadata, split_records_by_fold
from .losses import CompositeTopologyLoss
from .metrics import evaluate_metrics
from .models import build_model
from .utils import (
    AverageMeter,
    MetricTracker,
    configure_logging,
    detect_anomaly,
    get_scheduler,
    load_config,
    log_gpu_memory,
    log_patch_stats,
    save_config,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Vesuvius 3D surface detector")
    parser.add_argument("--config", required=True, help="Path to YAML config/experiment file")
    parser.add_argument("--device", default="cuda", help="Training device")
    return parser.parse_args()


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.ema = build_shadow(model)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

    def to(self, device: torch.device):
        self.ema.to(device)


def build_shadow(model: torch.nn.Module) -> torch.nn.Module:
    import copy

    shadow = copy.deepcopy(model)
    for param in shadow.parameters():
        param.requires_grad_(False)
    return shadow


def create_dataloaders(cfg: Dict, logger) -> Tuple[DataLoader, Optional[DataLoader]]:
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]
    augmentation_cfg = cfg.get("augmentation")

    metadata = read_metadata(Path(paths_cfg["train_csv"]), Path(paths_cfg["data_root"]))
    train_records, val_records = split_records_by_fold(metadata, data_cfg.get("train_folds", []), data_cfg.get("val_folds", []))
    logger.info("Loaded %d train and %d val records", len(train_records), len(val_records))

    train_iters = cfg["training"].get("train_iterations_per_epoch")
    if train_iters is None:
        patches_per_volume = cfg["training"].get("patches_per_volume", 16)
        train_iters = max(1, len(train_records) * patches_per_volume)

    train_dataset = PatchDataset(
        train_records,
        data_cfg,
        data_cfg["patch_size"],
        data_cfg["sampler"],
        augmentation_cfg,
        iterations_per_epoch=train_iters,
        mode="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["train_batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("workers", 4),
        pin_memory=False,  # Disabled: incompatible with spawn context, causes "Pin memory thread exited unexpectedly"
        drop_last=True,
        prefetch_factor=1,  # Reduce prefetch to limit memory per worker
        persistent_workers=False,  # Avoid memory accumulation across epochs
        multiprocessing_context="spawn",  # Use spawn to avoid copy-on-write memory issues
    )

    if not val_records:
        return train_loader, None

    val_iters = cfg["training"].get("val_iterations_per_epoch", len(val_records))
    val_dataset = PatchDataset(
        val_records,
        data_cfg,
        data_cfg["patch_size"],
        data_cfg["sampler"],
        augmentation_cfg=None,
        iterations_per_epoch=val_iters,
        mode="val",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["val_batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("workers", 4),
        pin_memory=False,  # Disabled: incompatible with spawn context, causes "Pin memory thread exited unexpectedly"
        drop_last=False,
        prefetch_factor=1,  # Reduce prefetch to limit memory per worker
        persistent_workers=False,  # Avoid memory accumulation across epochs
        multiprocessing_context="spawn",  # Use spawn to avoid copy-on-write memory issues
    )
    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CompositeTopologyLoss,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Dict,
    logger,
    ema: Optional[ModelEma] = None,
    batch_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, float]:
    model.train()
    tracker = MetricTracker()
    accum_steps = cfg["training"].get("accumulate_steps", 1)
    grad_clip = cfg["training"].get("gradient_clip")
    log_gpu = cfg["training"].get("log_gpu_memory", False)
    log_patches = cfg["training"].get("log_patch_stats", False)

    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        outputs = model(images)
        logits = outputs["logits"]
        if outputs.get("aux_logits"):
            aux_loss = 0.0
            for aux in outputs["aux_logits"]:
                resized_mask = F.interpolate(masks, size=aux.shape[-3:], mode="nearest")
                aux_loss = aux_loss + loss_fn(aux, resized_mask)[0]
            primary_loss, loss_details = loss_fn(logits, masks)
            loss = primary_loss + aux_loss * 0.5
        else:
            loss, loss_details = loss_fn(logits, masks)

        detect_anomaly(loss, "train_loss", logger)
        loss = loss / accum_steps
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if grad_clip:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
            if batch_scheduler:
                batch_scheduler.step()

        tracker.update("loss", loss.item() * accum_steps, images.size(0))
        for name, value in loss_details.items():
            tracker.update(name, value, images.size(0))

        if log_gpu and (step + 1) % 50 == 0:
            log_gpu_memory(logger, prefix=f"[Train step {step}] ")
        if log_patches:
            stats_dict = batch["patch_stats"]
            averaged = {}
            for k, v in stats_dict.items():
                if isinstance(v, torch.Tensor):
                    averaged[k] = float(v.float().mean().item())
                else:
                    # Handle list/array of values from collate_fn
                    v_tensor = torch.tensor(v, dtype=torch.float32) if not isinstance(v, (int, float)) else torch.tensor([v], dtype=torch.float32)
                    averaged[k] = float(v_tensor.mean().item())
            log_patch_stats(averaged, logger)

    return tracker.averages()


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: CompositeTopologyLoss,
    device: torch.device,
    tolerance_mm: float,
    spacing: Tuple[float, float, float],
    logger,
) -> Dict[str, float]:
    model.eval()
    tracker = MetricTracker()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            logits = outputs["logits"]
            loss, loss_details = loss_fn(logits, masks)
            tracker.update("loss", loss.item(), images.size(0))
            for name, value in loss_details.items():
                tracker.update(name, value, images.size(0))
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            for b in range(images.size(0)):
                pred_np = preds[b, 0].detach().cpu().numpy()
                mask_np = masks[b, 0].detach().cpu().numpy()
                metrics = evaluate_metrics(pred_np > 0.5, mask_np > 0.5, tolerance_mm, spacing)
                for k, v in metrics.items():
                    tracker.update(k, v, 1)
    logger.info("Validation metrics: %s", tracker.averages())
    return tracker.averages()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    experiment_cfg = cfg["experiment"]
    seed_everything(experiment_cfg.get("seed", 42))

    output_dir = Path(experiment_cfg.get("output_dir", "runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_path)
    save_config(cfg, output_dir / "config_resolved.yaml")

    train_loader, val_loader = create_dataloaders(cfg, logger)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"].get("lr", 2e-4),
        weight_decay=cfg["optimizer"].get("weight_decay", 1e-2),
        betas=tuple(cfg["optimizer"].get("betas", [0.9, 0.999])),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["experiment"].get("precision", "amp") == "amp")

    steps_per_epoch = len(train_loader) // cfg["training"].get("accumulate_steps", 1)
    scheduler_cfg = cfg.get("scheduler")
    if scheduler_cfg:
        scheduler_cfg = dict(scheduler_cfg)
        scheduler_cfg.setdefault("epochs", cfg["training"]["max_epochs"])
    scheduler = None
    scheduler_step_on_batch = False
    if scheduler_cfg:
        scheduler, scheduler_step_on_batch = get_scheduler(optimizer, scheduler_cfg, steps_per_epoch)

    loss_fn = CompositeTopologyLoss(cfg["loss"]["components"])
    ema = None
    if cfg["training"].get("ema", {}).get("enabled"):
        ema = ModelEma(model, cfg["training"]["ema"].get("decay", 0.999))
        ema.to(device)

    best_metric = -1.0
    start = time.time()
    spacing = tuple(cfg["data"].get("resample_spacing", [0.04, 0.04, 0.04]))
    tolerance_mm = cfg["metrics"].get("surface_dice_tolerance_mm", 2.0)

    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        logger.info("Epoch %d/%d", epoch, cfg["training"]["max_epochs"])
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler,
            cfg,
            logger,
            ema,
            scheduler if scheduler_step_on_batch else None,
        )
        logger.info("Train metrics: %s", train_metrics)
        if scheduler and not scheduler_step_on_batch:
            scheduler.step()

        val_metrics = {}
        if val_loader and epoch % cfg["training"].get("val_interval", 1) == 0:
            model_for_eval = ema.ema if (ema and cfg["training"]["ema"].get("eval_ema", True)) else model
            val_metrics = validate(model_for_eval, val_loader, loss_fn, device, tolerance_mm, spacing, logger)
            surface_dice_metric = val_metrics.get("surface_dice", 0.0)
            if surface_dice_metric > best_metric:
                best_metric = surface_dice_metric
                torch.save(
                    {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
                    output_dir / "checkpoints" / "best.pt",
                )
                logger.info("New best SurfaceDice %.4f", best_metric)

        torch.save(
            {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            output_dir / "checkpoints" / "last.pt",
        )
        elapsed = time.time() - start
        logger.info("Epoch %d finished in %s", epoch, elapsed)

    logger.info("Training completed. Best SurfaceDice %.4f", best_metric)


if __name__ == "__main__":
    main()

