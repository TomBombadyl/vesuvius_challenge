# Vesuvius Challenge VM Directory Map

Snapshot of the active Google Cloud VM (`vesuvius-challenge`, zone `us-central1-a`) so we know exactly where code, data, caches, and mounts live. This complements `google_cloud_details` by describing the live filesystem state.

---

## Top-Level Layout

| Path | Purpose / Notes |
|------|-----------------|
| `/home/dylant` | Primary user home. Holds SSH keys, shell config, and GCS mounts. |
| `/home/dylant/gcs_mount` | **Active `gcsfuse` mount** of `gs://vesuvius-kaggle-data`. Contains `train.csv`, `test.csv`, `train_images/`, `train_labels/`, `test_images/`. |
| `/home/dylant/vesuvius_gcs` | (Removed) Old root-owned mount. Directory recreated empty; main mount is `~/gcs_mount`. |
| `/mnt/disks/data` | 375 GB NVMe workspace disk. Mounted via `/dev/nvme0n1p1`. |
| `/mnt/disks/data/repos/vesuvius_challenge` | Git checkout + virtualenv + runs. All training commands should `cd` here first. |
| `/mnt/disks/data/pip-cache` | Redirected pip cache to avoid filling the 10 GB boot disk. |
| `/mnt/disks/data/tmp` | Scratch/TMPDIR for large downloads and CUDA JIT cache. |

---

## Detailed Map

### `/home/dylant`
```
~/
├── .ssh/                 # gcloud-managed keys
├── .venv/                # (not here; lives in repo)
├── gcs_mount/            # live bucket mount (see below)
└── vesuvius_gcs/         # leftover path, currently empty (mount removed)
```

### `/home/dylant/gcs_mount`
```
gcs_mount/
├── train.csv
├── test.csv
├── train_images/
├── train_labels/
└── test_images/
```

Mounted with:
```bash
gcsfuse --implicit-dirs vesuvius-kaggle-data ~/gcs_mount
```

### `/mnt/disks/data`
```
/mnt/disks/data
├── repos/
│   └── vesuvius_challenge/   # project repo
├── pip-cache/                # pip download cache
├── tmp/                      # TMPDIR, CUDA caches
└── lost+found/
```

### `/mnt/disks/data/repos/vesuvius_challenge`
```
vesuvius_challenge/
├── .venv/                    # Python 3.11 venv (torch, monai, etc.)
├── configs/
│   ├── vesuvius_baseline.yaml
│   └── experiments/
├── src/
├── tests/
├── runs/                     # training outputs/checkpoints
├── DEVLOG.md
├── README.md
└── Baseline Strategy for Vesuvius Challenge Surface Detection.pdf
```

---

## Usage Notes

1. **Always mount bucket before training**  
   ```bash
   gcsfuse --implicit-dirs vesuvius-kaggle-data ~/gcs_mount
   ```
   Configs reference `/home/dylant/gcs_mount`, so training fails if the mount drops.

2. **Run commands from the repo root**  
   ```bash
   cd /mnt/disks/data/repos/vesuvius_challenge
   source .venv/bin/activate
   export PYTHONPATH=/mnt/disks/data/repos/vesuvius_challenge
   ```

3. **Keep caches off the boot disk**  
   Set once per session:
   ```bash
   export TMPDIR=/mnt/disks/data/tmp
   export PIP_CACHE_DIR=/mnt/disks/data/pip-cache
   export CUDA_CACHE_PATH=/mnt/disks/data/tmp

   ```

5. **Directory diff check**  
   Use `tree -L 2 /mnt/disks/data/repos/vesuvius_challenge` when you need a quick sanity check after syncing or pulling new code.

6. **80 GB exp001_full workflow**  
   - Config: `configs/experiments/exp001_full.yaml` (patch `[72,136,136]`, stride `[36,104,104]`, base_channels=40, deep supervision on, activation checkpointing applied to bottleneck/decoder).  
   - Validation: `runs/exp001_full_vramcheck` stores the 50-step dry run logs plus `logs/vram_watch.txt` (peak VRAM 57.6 GB).  
   - Production command (run via `nohup` so you can disconnect):
     ```bash
     cd /mnt/disks/data/repos/vesuvius_challenge
     source .venv/bin/activate
     export PYTHONPATH=/mnt/disks/data/repos/vesuvius_challenge
     nohup python -m src.vesuvius.train --config configs/experiments/exp001_full.yaml --device cuda \
       > runs/exp001_full/logs/train_full.log 2>&1 &
     ```
   - Runtime cap: `train_iterations_per_epoch=4000`, `max_epochs=32` (≈36‑40 h total).  
   - Monitoring commands from your laptop (PowerShell):
     ```
     gcloud compute ssh dylant@vesuvius-challenge --project vesuvius-challenge-478512 --zone us-central1-a --command \
       "cd /mnt/disks/data/repos/vesuvius_challenge && tail -n 50 runs/exp001_full/logs/train_full.log"
     gcloud compute ssh dylant@vesuvius-challenge --project vesuvius-challenge-478512 --zone us-central1-a --command \
       "pgrep -af 'python -m src.vesuvius.train'"
     gcloud compute ssh dylant@vesuvius-challenge --project vesuvius-challenge-478512 --zone us-central1-a --command "nvidia-smi"
     ```
   - Checkpoints & logs land in `runs/exp001_full/{checkpoints,logs}`. Kill/restart via `pkill -f "python -m src.vesuvius.train"` if you need to regenerate.

---

_Last updated: 2025‑11‑18_***

