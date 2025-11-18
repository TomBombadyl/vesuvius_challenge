# Vesuvius Challenge – Surface Detection

Comprehensive 3D segmentation toolkit for the [Vesuvius Challenge](https://scrollprize.org/) and (https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection).
 The project targets large CT volumes of carbonized papyri, delivering a topology-aware segmentation stack that trains on a Google Cloud A100 VM and runs deterministically inside Kaggle notebooks for submission.

---

## Highlights
- **Config-driven pipeline** with YAML inheritance (`_base_` key) and experiment presets in `configs/experiments/`.
- **3D data stack**: isotropic resampling, clipping, denoising, FG-cropping, FG-aware patch sampling, Gaussian-blended sliding-window inference, and flip-based TTA.
- **Model zoo**: residual UNet variants (deep supervision, InstanceNorm), lightweight UNet, optional Swin UNETR backend (via MONAI).
- **Loss cocktail**: weighted BCE + soft Dice + clDice + morphological skeleton + surface-distance + simple TopoLoss.
- **Augmentations**: elastic deformation, anisotropic scaling, slice jitter, patch dropout, gamma noise, blur, cutout, etc.
- **Training extras**: AMP, gradient accumulation/clipping, EMA, NaN detection, GPU-memory stats, patch-stat logging, cosine/one-cycle/poly schedulers.
- **Validation metrics**: SurfaceDice@2, variation-of-information (VOI), TopoScore approximation.

---

## Repository Layout
```
├── configs/
│   ├── vesuvius_baseline.yaml
│   └── experiments/
│       ├── exp001_3d_unet_topology.yaml
│       └── exp002_swinunetr.yaml
├── src/vesuvius/
│   ├── data.py               # volume loading, preprocessing, datasets
│   ├── transforms.py         # spatial/intensity augmentations
│   ├── patch_sampler.py      # FG-aware sampler for patches
│   ├── models.py             # residual/lightweight UNet + Swin UNETR
│   ├── losses.py             # composite topology-aware loss stack
│   ├── metrics.py            # SurfaceDice/VOI/TopoScore
│   ├── postprocess.py        # CC filtering, morph closing, hole fill
│   ├── train.py              # experiment-driven trainer CLI
│   ├── infer.py              # sliding-window inference + TTA
│   └── utils.py              # config merge, logging, EMA, schedulers
├── tests/test_synthetic_pipeline.py
└── DEVLOG.md                 # chronological progress notes
```

---

## Environment & Data
- **Hardware**: GCP `a2-ultragpu-1g` VM (A100 80 GB, Debian 12) for training; Kaggle GPU notebook (≤ 9 h runtime) for inference/submission.
- **Python**: 3.10+ recommended; dependencies primarily PyTorch + SciPy ecosystem (+ MONAI if using Swin UNETR).
- **Data location**: `gs://vesuvius-kaggle-data` mounted via Cloud Storage FUSE at `/home/<user>/gcs_mount` (e.g. `/home/dylant/gcs_mount`), with an optional NVMe mirror at `/mnt/disks/data/vesuvius_kaggle_data`. Kaggle notebooks receive `test_images/` automatically.
- Ensure `PYTHONPATH` includes repo root (`export PYTHONPATH=src`).

### Streaming data via Cloud Storage FUSE

```bash
# Install once (Debian 12)
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb http://packages.cloud.google.com/apt ${GCSFUSE_REPO} main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/gcsfuse-archive-keyring.gpg >/dev/null
sudo apt-get update && sudo apt-get install -y gcsfuse

# Grant IAM role from your admin account (run once):
gcloud storage buckets add-iam-policy-binding gs://vesuvius-kaggle-data \
  --member=serviceAccount:137233308900-compute@developer.gserviceaccount.com \
  --role=roles/storage.objectViewer

# Mount on the VM
mkdir -p ~/gcs_mount
gcsfuse --implicit-dirs vesuvius-kaggle-data ~/gcs_mount
ls ~/gcs_mount  # train_images/, train_labels/, etc.
```

Set `paths.data_root: /home/<user>/gcs_mount` (e.g. `/home/dylant/gcs_mount`) in experiment configs to stream directly from the bucket without long `gsutil rsync` steps. Adjust for your username if different.

---

## Config System & Experiments
1. **Base config** lives in `configs/vesuvius_baseline.yaml` (paths, preprocessing defaults, losses, scheduler, etc.).
2. **Experiment overrides** set `_base_: ../vesuvius_baseline.yaml` then override only changed fields.
3. **Resolution**: `load_config()` recursively loads parents, deep-merges child dicts (lists are replaced), then writes the resolved config to each run folder.
4. **Run directories**: `experiment.output_dir` (defaults to `runs/<name>`). `train.py` creates `runs/<exp>/{checkpoints,logs}` and stores `config_resolved.yaml`.

---

## Training
```bash
cd /mnt/disks/data/repos/vesuvius_challenge
export PYTHONPATH=src

# Baseline residual UNet
python src/vesuvius/train.py --config configs/vesuvius_baseline.yaml

# Phase 2 residual UNet with full topology losses (exp001)
python src/vesuvius/train.py --config configs/experiments/exp001_3d_unet_topology.yaml

# Swin UNETR variant (requires MONAI)
python src/vesuvius/train.py --config configs/experiments/exp002_swinunetr.yaml
```

**GCP A100 single-GPU examples (recommended)**

```bash
cd /mnt/disks/data/repos/vesuvius_challenge
export PYTHONPATH=src

# Residual UNet + topology cocktail (exp001)
python -m torch.distributed.run --nproc_per_node=1 src/vesuvius/train.py \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --device cuda

# Swin UNETR transformer variant (exp002, requires MONAI)
python -m torch.distributed.run --nproc_per_node=1 src/vesuvius/train.py \
  --config configs/experiments/exp002_swinunetr.yaml \
  --device cuda
```

Key runtime features:
- Mixed precision via AMP (toggle under `experiment.precision`).
- Gradient accumulation (`training.accumulate_steps`), gradient clipping, and optional EMA for evaluation.
- Scheduler factory supporting cosine / poly / one-cycle (automatic per-batch vs per-epoch stepping).
- Logging of GPU memory and patch statistics (enable in `training.log_gpu_memory` / `log_patch_stats`).
- Validation metrics integrated into the train loop (`metrics.*`).

### Smoke-run preset (memory-safe)
Large Phase‑2 configs can momentarily exceed 60 GB of GPU memory. Before committing to a long job, run the dedicated smoke config (keeps footprint ≈20 GB):

```bash
python -m src.vesuvius.train --config configs/experiments/exp001_smoke.yaml --device cuda
```

Want to tweak manually instead? Use `--override` exactly as the smoke config does (base_channels=32, deep_supervision=false, patch_size `[64,128,128]`, stride `[32,96,96]`, workers=2, surface_distance weight=0, max_epochs=1). Once the loop finishes cleanly, gradually restore the original values while watching `nvidia-smi`. If memory spikes again, enable activation checkpointing around the deep decoder blocks before increasing patch size further.

**Evaluation reminders:**
- Track learning rate and loss curve each run (TensorBoard or the logged scalars) to verify optimizer behavior before scaling up.
- Monitor GPU metrics via `nvidia-smi` (temp, usage, alloc/reserved) and log the peak per experiment.

### exp001_full (80 GB plan) & ≤48 h production cap
- Config lives in `configs/experiments/exp001_full.yaml` (patch `[72,136,136]`, stride `[36,104,104]`, base_channels = 40, deep supervision enabled, activation checkpointing over bottleneck/decoder with 2 segments each). Inherits all topology losses from exp001.
- Before any long run, execute a **50-step validation smoke**:
  ```bash
  cd /mnt/disks/data/repos/vesuvius_challenge
  source .venv/bin/activate
  export PYTHONPATH=/mnt/disks/data/repos/vesuvius_challenge
  python /tmp/remote_50step.sh  # or replicate: create temp config, run 50 steps, tee nvidia-smi output
  ```
  Expect peak VRAM ≈57.6 GB (alloc ~57 GB, reserved ~57.6 GB) and composite loss ≈1.82 (BCE 0.88 / Dice 0.74 / clDice 0.72 / morph 0.066 / surface-distance 0.60 / toploss 0.42). Abort if >80 GB and shrink patch size one notch.
- Production runs are capped to **≤48 hours** by setting:
  ```yaml
  training:
    train_iterations_per_epoch: 4000   # ≈70–75 min / epoch on A100
    max_epochs: 32                    # ~36–40 h wall-clock
  ```
- Launch command (nohup-friendly):
```bash
  cd /mnt/disks/data/repos/vesuvius_challenge
  source .venv/bin/activate
  export PYTHONPATH=/mnt/disks/data/repos/vesuvius_challenge
  nohup python -m src.vesuvius.train --config configs/experiments/exp001_full.yaml --device cuda \
    > runs/exp001_full/logs/train_full.log 2>&1 &
  ```
- Monitoring cheatsheet (run locally via PowerShell):
  ```
  gcloud compute ssh dylant@vesuvius-challenge --project vesuvius-challenge-478512 --zone us-central1-a --command \
    "cd /mnt/disks/data/repos/vesuvius_challenge && tail -n 50 runs/exp001_full/logs/train_full.log"
  gcloud compute ssh dylant@vesuvius-challenge --project vesuvius-challenge-478512 --zone us-central1-a --command \
    "pgrep -af 'python -m src.vesuvius.train'"
  gcloud compute ssh dylant@vesuvius-challenge --project vesuvius-challenge-478512 --zone us-central1-a --command "nvidia-smi"
  ```
- Leave the VM unattended; `nohup` + background DataLoader workers keep the run alive even if your local session closes.

---

## Inference & Submission
```bash
export PYTHONPATH=src
python src/vesuvius/infer.py \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --checkpoint runs/exp001_3d_unet_topology/checkpoints/best.pt \
  --output-dir runs/exp001_3d_unet_topology/infer
```

Features:
- Sliding-window inference with configurable patch size/overlap, Gaussian blending, and flip-only or 8× TTA.
- Post-processing pipeline (component filtering, hole fill, morph closing) controlled via `postprocess` block.
- Output masks saved as `.tif` (or `.npy` fallback) ready for Kaggle submission packaging.

**Kaggle Notebook Tips**
- Preload the trained checkpoint into the Kaggle dataset or upload via notebook data tab (no internet).
- Keep inference under 9 h by matching patch sizes/overlaps from configs and disabling unneeded TTA if close to the limit.
- Notebook must produce `submission.zip` with `{fragment_id}.tif` binaries inside; reuse `infer.py` logic directly.

---

## Testing
```bash
export PYTHONPATH=src
pytest tests/test_synthetic_pipeline.py
```
The synthetic test builds a toy volume, runs the residual UNet forward pass, and checks composite losses stay finite—useful sanity test before long training runs.

---

## Development Notes
- All new experiment files or notebooks should reference environment paths via config (avoid hardcoding).
- Swin UNETR requires `pip install monai` on the VM; guard blocks handle absence gracefully.
- Keep dev progress documented in `DEVLOG.md` (see template) to track experiments, metrics, and follow-ups.

---

## Licensing & Credits
This repo is maintained internally for Vesuvius Challenge experimentation. Cite Vesuvius Challenge resources and the Baseline Strategy whitepaper bundled in this repo when publishing derived work.

