# Dev Log – Vesuvius Challenge Surface Detection

Chronological record of major implementation steps, experiments, and decisions.

---

## 2025-11-19 — System Validation & Optimization

### Critical Fixes
- **Checkpoint saving:** Fixed `RuntimeError` by creating `checkpoints/` directory before `torch.save()` in `train.py`
- **Logging:** Fixed potential `FileNotFoundError` by creating log directory in `configure_logging()` in `utils.py`
- **DataLoader optimization:** Increased workers to 2, `prefetch_factor` to 2, enabled `persistent_workers`

### Memory Optimizations
- **Surface distance loss:** Serialized with `threading.Lock()` to prevent OOM when multiple batches compute distance transforms simultaneously
- **VolumeCache:** Implemented LRU cache with `max_size=50` to prevent unbounded memory accumulation
- **Loss computation:** Increased `surface_distance_interval` from 4 to 16 steps (93.75% reduction in expensive operations)

### Test Run Results
- **10-step test:** Completed all 32 epochs successfully (46 minutes total)
- **Checkpoints:** Verified saving correctly (386MB `last.pt`)
- **Metrics:** Loss improved 13.8% (1.794 → 1.547), morph_skeleton improved 66.7%
- **Status:** ✅ System validated, ready for full training

### Documentation
- Created `STEPS_PER_EPOCH_GUIDE.md` - comprehensive guide on choosing steps per epoch
- Created `LOSS_COMPONENTS_EXPLAINED.md` - detailed explanation of all loss components
- Created `TEST_RUN_EVALUATION.md` - analysis of test run results
- Created `WHY_METRICS_INCREASED.md` - explanation of metric behavior

### Next Steps
- **Full training:** Update config to 4 workers, batch_size=2, 2000 steps/epoch
- **Expected:** ~6-7 hours per epoch, ~9-10 days total for 32 epochs

---

## 2025-11-18 — Production Configuration

### exp001_full Setup
- **Config:** `configs/experiments/exp001_full.yaml`
- **Model:** ResidualUNet3D (base_channels=40, deep_supervision=True, activation checkpointing)
- **Patch:** [72, 136, 136], stride [36, 104, 104]
- **Memory:** Peak VRAM 57.6 GB (validated with 50-step smoke run)
- **Runtime:** Capped at ≤48 hours (4000 steps/epoch, 32 epochs)

---

## 2025-11-17 — Phase 2 Implementation

### Core Features
- **Topology-aware losses:** clDice, morphological skeleton, surface-distance, soft TopoLoss
- **Augmentations:** Elastic deformation, anisotropic scaling, slice jitter, patch dropout, gamma noise
- **Models:** Residual UNet with deep supervision, lightweight UNet, optional Swin UNETR
- **Training:** EMA, gradient accumulation/clipping, scheduler factory, GPU memory logging
- **Inference:** Sliding-window with Gaussian blending, TTA, post-processing

### Infrastructure
- **Data access:** GCS bucket mounted via `gcsfuse` at `/home/dylant/gcs_mount`
- **VM:** GCP `a2-ultragpu-1g` (A100 80GB, 12 vCPUs, 170GB RAM)
- **Storage:** `gs://vesuvius-kaggle-data` with service account access

---

## Future Work
- Full training run with optimized configuration (4 workers, batch_size=2, 2000 steps/epoch)
- Validation set configuration and monitoring
- Kaggle inference notebook preparation
- Model performance evaluation and comparison
