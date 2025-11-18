from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Optional, Tuple, Union

import numpy as np
import torch
import yaml

try:
    from torch.cuda import amp  # type: ignore
except ImportError:  # pragma: no cover
    amp = None


PathLike = Union[str, os.PathLike]


def _deep_update(base: MutableMapping[str, Any], override: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge two dictionaries."""
    for key, value in override.items():
        if isinstance(value, MutableMapping) and isinstance(base.get(key), MutableMapping):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: PathLike) -> Dict[str, Any]:
    """Load YAML config with optional `_base_` recursion."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as fp:
        raw_cfg = yaml.safe_load(fp) or {}

    base_entry = raw_cfg.pop("_base_", None)
    if base_entry:
        base_path = (path.parent / base_entry).resolve()
        base_cfg = load_config(base_path)
        return dict(_deep_update(base_cfg, raw_cfg))
    return raw_cfg


def save_config(config: Dict[str, Any], path: PathLike) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)


def json_pretty_dump(obj: Any, path: PathLike) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def configure_logging(log_path: Optional[PathLike] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("vesuvius")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return sum(int(p.numel()) for p in model.parameters())


def create_run_dir(base_dir: PathLike, experiment_name: str) -> Path:
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


class AverageMeter:
    def __init__(self, name: str, fmt: str = ":.4f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self) -> str:  # pragma: no cover - human readable
        fmt = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmt.format(name=self.name, val=self.val, avg=self.avg)


class MetricTracker:
    def __init__(self) -> None:
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter(name)
        self.meters[name].update(value, n)

    def averages(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.meters.items()}


def maybe_autocast(enabled: bool):
    if enabled and amp is not None:
        return amp.autocast()
    return _NullContext()


class _NullContext:
    def __enter__(self):  # pragma: no cover - trivial
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # pragma: no cover - trivial
        return


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], path: PathLike,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def save_checkpoint(state: Dict[str, Any], path: PathLike, logger: Optional[logging.Logger] = None) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)
    if logger:
        logger.info("Checkpoint saved: %s", ckpt_path.as_posix())


def log_gpu_memory(logger: logging.Logger, prefix: str = "") -> None:
    if not torch.cuda.is_available():  # pragma: no cover - GPU only
        logger.info("%sGPU not available", prefix)
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    logger.info(
        "%sGPU memory (GB): alloc %.2f | reserved %.2f | max_alloc %.2f",
        prefix, allocated, reserved, max_allocated
    )


def log_patch_stats(stats: Dict[str, float], logger: logging.Logger) -> None:
    message = ", ".join(f"{k}={v:.3f}" for k, v in stats.items())
    logger.info("Patch stats: %s", message)


def detect_anomaly(tensor: torch.Tensor, name: str, logger: Optional[logging.Logger] = None) -> None:
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        msg = f"Detected NaN/Inf in tensor {name}"
        if logger:
            logger.error(msg)
        raise FloatingPointError(msg)


def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_cfg: Dict[str, Any],
                  steps_per_epoch: int) -> Tuple[Optional[torch.optim.lr_scheduler._LRScheduler], bool]:
    sched_type = scheduler_cfg.get("type")
    if sched_type is None:
        return None, False
    sched_type = sched_type.lower()
    step_on_batch = False
    if sched_type == "cosine":
        t_max = scheduler_cfg.get("t_max", 100) * steps_per_epoch
        eta_min = scheduler_cfg.get("min_lr", 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        step_on_batch = False
    elif sched_type == "one_cycle":
        step_on_batch = True
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_cfg.get("max_lr", scheduler_cfg.get("lr", 1e-3)),
            steps_per_epoch=steps_per_epoch,
            epochs=scheduler_cfg.get("epochs", 100),
            pct_start=scheduler_cfg.get("pct_start", 0.3),
            anneal_strategy=scheduler_cfg.get("anneal_strategy", "linear"),
            div_factor=scheduler_cfg.get("div_factor", 25.0),
            final_div_factor=scheduler_cfg.get("final_div_factor", 1e4),
        )
    elif sched_type == "poly":
        step_on_batch = True
        power = scheduler_cfg.get("power", 0.9)
        max_iters = scheduler_cfg.get("max_iters", steps_per_epoch * scheduler_cfg.get("epochs", 100))

        class PolyLR(torch.optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, max_iters, power):
                self.max_iters = max_iters
                self.power = power
                self.iter = 0
                super().__init__(optimizer)

            def get_lr(self):
                factor = (1 - self.iter / max(1, self.max_iters)) ** self.power
                return [base_lr * factor for base_lr in self.base_lrs]

            def step(self, epoch: Optional[int] = None):
                self.iter += 1
                super().step()

        scheduler = PolyLR(optimizer, max_iters=max_iters, power=power)
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")

    return scheduler, step_on_batch


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, MutableMapping):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


__all__ = [
    "load_config",
    "save_config",
    "json_pretty_dump",
    "seed_everything",
    "configure_logging",
    "count_parameters",
    "create_run_dir",
    "AverageMeter",
    "MetricTracker",
    "maybe_autocast",
    "load_checkpoint",
    "save_checkpoint",
    "log_gpu_memory",
    "log_patch_stats",
    "detect_anomaly",
    "get_scheduler",
    "format_seconds",
    "flatten_dict",
]

