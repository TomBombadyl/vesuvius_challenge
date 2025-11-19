# Dev Log – Vesuvius Challenge Surface Detection

Chronological record of major implementation steps, experiments, and follow-ups. Use this log to coordinate future work and quickly recall decisions/next actions.

---

## 2025‑11‑17 — Baseline Implementation (Phase 1)
- **Code skeleton** under `src/vesuvius/` with modular data/model/loss/train/infer components.
- **Config system** established (`configs/vesuvius_baseline.yaml`) capturing preprocessing, residual UNet, BCE+Dice+topology stubs, cosine LR, sliding-window inference.
- **Tests**: Added `tests/test_synthetic_pipeline.py` synthetic pass-through to validate dataset, model, and loss interplay.
- Result: Solid baseline ready for GCP training; inference produces Kaggle-compatible masks.

### Follow-ups
- Collect baseline metrics (SurfaceDice / VOI) per fragment.
- Package baseline checkpoint into Kaggle dataset for submission notebook.

---

## 2025‑11‑17 — Phase 2 Upgrade
- **Topology-aware losses**: Implemented full clDice, morphological skeleton loss, surface-distance loss, and soft TopoLoss (Betti/hole approximations).
- **Augmentations**: Added elastic deformation, anisotropic scaling, slice jitter, patch dropout, gamma noise, etc.
- **Models**: Refined residual UNet with deep supervision, lightweight UNet, optional Swin UNETR backend.
- **Training loop**: Config inheritance, EMA, gradient accumulation/clipping, scheduler factory (cosine/one-cycle/poly), GPU memory + patch stats logging, NaN detection, SurfaceDice/VOI/TopoScore metrics.
- **Inference**: Sliding-window + Gaussian blending, flip/full TTA, CC filtering + morph closing.
- **Experiments**: Added `exp001_3d_unet_topology.yaml` and `exp002_swinunetr.yaml` under `configs/experiments/`.
- **Docs**: Authored project `README.md` (now documents GCP command recipes + gcsfuse streaming) and this `DEVLOG.md`.
- **Data access**: Mounted `gs://vesuvius-kaggle-data` via gcsfuse under `/home/dylant/gcs_mount` (previous `/mnt/vesuvius_gcs` attempt left a root-owned mount). Service account has `storage.objectViewer`, so training streams CT fragments without rsync delays.
- **Sanity checks**: Plan a “double-check” pass (dry-run configs, re-run synthetic test, lint configs) before launching long GCP experiments.
- **Memory notes**: Full exp001 (80×144×144 patches, base_channels=40, deep supervision, full topology losses) peaks at ~60 GB on A100. Added README guidance for a low-memory smoke run (base_channels=32, smaller patch, no deep supervision, surface-distance loss disabled, workers=2) to verify the pipeline before scaling back up.

### 2025‑11‑17 — Smoke Run & Memory Audit
- Created `configs/experiments/exp001_smoke.yaml` (inherits from exp001 but shrinks patch to `[64,128,128]`, base_channels=32, turns off deep supervision, sets surface-distance weight=0, workers=2, single epoch). This mimics nnU-Net’s residual-UNet “planner” configs so we can test the whole pipeline under ~20 GB GPU load.
- Ran the smoke config on the GCP A100; GPU memory stayed at ~19 GB (per `nvidia-smi` and our patch-stat logs). Confirms data pipeline, sampler, augmentation, and loss stack run end-to-end.
- Documented the smoke workflow + scaling strategy in README plus noted in `google_cloud_details`.
- Next up: reintroduce features one at a time (deep supervision, larger patch, surface-distance weight, more workers) while watching for ≈70 GB peak, and optionally add activation checkpointing if needed.

### 2025‑11‑18 — exp001_full 80 GB plan & runtime cap (≤48 h)
- Authored `configs/experiments/exp001_full.yaml` (inherits exp001, patch `[72,136,136]`, stride `[36,104,104]`, base_channels = 40, deep supervision on, activation checkpointing over bottleneck/decoder with 2 segments).
- Ran a 50-step validation pass (`runs/exp001_full_vramcheck`) with Gaussian-blended logging + `nvidia-smi` sampling; peak VRAM 57.6 GB (alloc 57 GB, reserved 57.6 GB) leaving ~22 GB headroom. Loss snapshot: 1.82 total (BCE 0.88 / Dice 0.74 / clDice 0.72 / morph 0.066 / surface-distance 0.60 / toploss 0.42).
- Aligned runtime budget to ≤48 h by explicitly setting `training.train_iterations_per_epoch = 4000` and `training.max_epochs = 32` (~70–75 min per epoch → 36–40 h total). Added instructions to restart runs if patches_per_volume ever drifts upward.
- Launched the production run on the A100 via `nohup python -m src.vesuvius.train --config configs/experiments/exp001_full.yaml --device cuda`. Monitoring commands documented (tail log, `pgrep`, `nvidia-smi` CSV).
- Next: capture first-epoch metrics, keep GPU trace in `runs/exp001_full/logs/vram_watch.txt`, and update README/infra docs with monitoring + runtime expectations.

### 2025-11-19 — Critical Fixes & Test Run Analysis
- **Fixed checkpoint directory creation:** Added `(output_dir / "checkpoints").mkdir()` in `train.py` to prevent `RuntimeError` after epoch completion. Root cause: `torch.save()` doesn't create parent directories.
- **Fixed logging directory creation:** Added directory creation in `configure_logging()` in `utils.py` to prevent potential `FileNotFoundError`.
- **Optimized DataLoader:** Increased workers to 2, `prefetch_factor` to 2, enabled `persistent_workers` for better throughput.
- **Test run completed:** 8 epochs with 10 steps each, all checkpoints saving successfully (386MB `last.pt` verified). Loss decreasing: 1.794 → 1.593. GPU memory stable at 67GB (84% utilization), GPU utilization low at 17% (data-bound).
- **Analysis:** Created `TRAINING_FIXES_AND_ANALYSIS.md` with comprehensive breakdown. Recommendations: increase workers to 4, batch size to 2, adjust steps/epoch to 2000 for ~6-7 hour epochs.
- **Next:** Phase 2 optimization (4 workers, batch_size=2) before full 32-epoch run.

### Follow-ups
- Run exp001 on GCP (target ≥0.70 SurfaceDice@2). Log metrics + patch stats in `runs/exp001...`.
- Monitor GPU memory for large patches; adjust overlap if nearing 80 GB.
- Benchmark Swin UNETR (exp002) vs. residual UNet; compare topology metrics and runtime.
- Prepare Kaggle inference notebook that loads best checkpoint, runs `infer.py` logic, and writes `submission.zip`.

---

## Future Ideas
- Integrate nnU-Net style auto-configuration for patch sizes/spacing per fragment.
- Add curriculum sampling that ramps FG ratio across epochs.
- Explore lightweight diffusion-based post-processing to refine skeletons.

