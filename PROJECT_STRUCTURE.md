# Project Structure & File Reference

## 📚 Documentation Files (Essential)

| File | Purpose | Status |
|------|---------|--------|
| `START_HERE.md` | **Entry point** - Read this first | ✅ Current |
| `CURRENT_STATUS.md` | Project status & checklist | ✅ Current |
| `README.md` | Project overview & architecture | ✅ Current |
| `DEVLOG.md` | Development history & decisions | ✅ Current |
| `QUICK_START.md` | Fast command reference | ✅ Current |
| `TECHNICAL_ROOT_CAUSE_ANALYSIS.md` | Deep dive: inference issue explanation | ✅ Current |
| `INFERENCE_ARCHITECTURE_MISMATCH.md` | Simple explanation of current issue | ✅ Current |
| `VALIDATION_PIPELINE_PLAN.md` | **Next steps** - Validation strategy | ✅ Current |
| `PROJECT_READY.txt` | Visual project state summary | ✅ Current |

---

## 🛠️ Source Code

```
src/vesuvius/
├── __init__.py                 # Package initialization
├── train.py                    # Training loop & CLI
├── infer.py                    # Inference & sliding window
├── evaluate.py                 # Validation metrics computation
├── models.py                   # ResidualUNet3D, LightUNet3D, SwinUNETR
├── losses.py                   # Composite loss: BCE, Dice, clDice, TopoLoss
├── data.py                     # Datasets, dataloaders, transforms
├── transforms.py               # Augmentation pipeline
├── metrics.py                  # Evaluation metrics
├── postprocess.py              # Post-processing operations
├── patch_sampler.py            # Patch sampling for training
└── utils.py                    # Utilities & helpers
```

---

## ⚙️ Configuration Files

```
configs/
├── vesuvius_baseline.yaml      # Base config (inherited by experiments)
└── experiments/
    └── exp001_3d_unet_topology.yaml  # **ACTIVE** - Main experiment config
```

**Note:** Only `exp001_3d_unet_topology.yaml` is used. Others cleaned up.

---

## 📊 Checkpoints & Results

```
checkpoints/
└── last_exp001.pt             # Final trained model (33.67M params)

runs/
├── exp001_3d_unet_topology_full/
│   ├── checkpoints/
│   │   └── last.pt            # Model checkpoint
│   ├── config_resolved.yaml    # Resolved config from training
│   ├── logs/                   # Training logs
│   └── infer_val/              # Validation inference outputs
│       ├── config_resolved.yaml
│       └── infer.log           # Inference log (troubleshooting)
└── exp001_full/                # (Old run, keep for reference)
```

---

## 📦 Data

```
vesuvius_kaggle_data/
├── train_images/               # 806 training volumes (3D CT scans)
├── train_labels/               # Ground truth labels (0/1/2)
├── test_images/                # Test data (1 file: 1407735.tif)
├── train.csv                   # Metadata (volume_id, fold, etc.)
└── test.csv                    # Test metadata
```

**Status:** ✅ All data on cloud VM at `/mnt/disks/data/repos/vesuvius_challenge/vesuvius_kaggle_data/`

---

## 🧪 Tests

```
tests/
└── test_synthetic_pipeline.py  # Synthetic smoke test
```

---

## 🚀 Deployment Files

| File | Purpose | When to Use |
|------|---------|------------|
| `run_cloud_validation.ps1` | SSH into cloud VM & run inference | Phase 1 |
| `kaggle_notebook_template.py` | Template for Kaggle submission notebook | Phase 5 |

---

## 🗑️ Cleaned Up (Removed)

The following files have been removed to reduce clutter:

**Old Documentation:**
- AGENT_EXECUTION_SUMMARY.md
- CLEANUP_SUMMARY.txt
- DATA_TRANSFER_IN_PROGRESS.md
- PHASE1_STATUS.md
- READY_FOR_VALIDATION.md
- REMOTE_VALIDATION_RUNNER.md
- MODEL_VERIFICATION_RESULTS.md
- TRAINING_RESULTS_COMPREHENSIVE.md
- train_log.txt

**Old Scripts:**
- monitor_inference.ps1
- sync_data_to_cloud.ps1
- verify_model.py

**Old Configs:**
- exp001_full.yaml
- exp002_swinunetr.yaml

---

## 📋 Current Project Status

### ✅ Completed
- Model architecture designed (ResidualUNet3D)
- Training pipeline implemented
- Model trained successfully (32 epochs)
- Checkpoint saved (33.67M params, 394.8 MB)
- Data transferred to cloud VM (27.49 GB)
- Inference code ready

### ⚠️ In Progress
- **Inference Config Fix:** Update patch_size from [72, 136, 136] → [256, 384, 384]
- **Root Cause Identified:** Encoder depth mismatch (5 levels × 2× downsampling = 32× reduction)

### ⏳ Next Steps (VALIDATION_PIPELINE_PLAN.md)
1. Fix config & quick test (10 volumes) - ~30 min
2. Validation inference (100 volumes, fold 3) - ~1 hour
3. Compute metrics (Surface Dice, VOI, TopoScore) - ~10 min
4. Threshold sweep (0.30 → 0.55) - ~20 min
5. TTA comparison (none/flips/full_8x) - ~2-4 hours
6. Post-processing tuning - ~30 min
7. Full inference (806 volumes) - ~8-10 hours
8. Kaggle submission - ~1-2 hours

### 🎯 Success Criteria
- Surface Dice > 0.65 ✅ (target from training)
- All metrics computed & logged
- Optimal threshold identified
- Post-processing tuned
- Kaggle notebook ready

---

## 🔍 How to Navigate This Project

### Quick Start
1. **Read:** `START_HERE.md` (2 min)
2. **Understand Current Issue:** `INFERENCE_ARCHITECTURE_MISMATCH.md` (5 min)
3. **Deep Dive (optional):** `TECHNICAL_ROOT_CAUSE_ANALYSIS.md` (15 min)
4. **Next Steps:** `VALIDATION_PIPELINE_PLAN.md` (5 min)

### For Execution
- **Run inference:** See `QUICK_START.md`
- **Run validation:** See `VALIDATION_PIPELINE_PLAN.md`
- **Kaggle submission:** See `kaggle_notebook_template.py`

### For Development
- **Train model:** `src/vesuvius/train.py`
- **Run inference:** `src/vesuvius/infer.py`
- **Evaluate:** `src/vesuvius/evaluate.py`
- **Config:** `configs/experiments/exp001_3d_unet_topology.yaml`

### For Understanding
- **Model architecture:** `src/vesuvius/models.py`
- **Loss functions:** `src/vesuvius/losses.py`
- **Data loading:** `src/vesuvius/data.py`
- **Augmentations:** `src/vesuvius/transforms.py`
- **Metrics:** `src/vesuvius/metrics.py`

---

## 📈 Project Timeline

```
Phase 1: Fix Config          (NOW) - 30 minutes
Phase 2: Validation Inference     - 1-2 hours
Phase 3: Metrics Computation      - 30 minutes
Phase 4: Threshold Sweep          - 20 minutes
Phase 5: TTA Comparison           - 2-4 hours (optional)
Phase 6: Post-Processing Tuning   - 30 minutes
Phase 7: Full Inference (806 vol) - 8-10 hours
Phase 8: Kaggle Submission        - 1-2 hours

TOTAL: ~12-15 hours to submission
```

---

## 🔧 Key Configuration Parameters

**Inference (NEEDS FIX):**
```yaml
inference:
  patch_size: [256, 384, 384]    # ← CHANGED (was [72, 136, 136])
  overlap: [128, 256, 256]
  tta: none                       # Start here, can upgrade to 'flips' or 'full_8x'
  threshold: 0.42                 # Will optimize during sweep
  min_component_voxels: 600
  morph_closing_radius: 3
```

**Model:**
```yaml
model:
  type: unet3d_residual
  base_channels: 40
  channel_multipliers: [1, 2, 2, 4, 4]  # 5 encoder levels
  blocks_per_stage: 3
  deep_supervision: true
```

---

## ✨ What Makes This Project Special

1. **Production-Ready Code:** Modular, configurable, type-hinted
2. **Composite Loss Function:** BCE + Dice + clDice + TopoLoss + Surface Distance + Morphology
3. **3D Augmentation:** Realistic transforms for medical imaging
4. **TTA Support:** None / Flips / 8× combinations
5. **Post-Processing:** Component removal, hole filling, morphological operations
6. **Kaggle Submission Ready:** Template notebook included

---

## 🎯 Model Performance (Expected)

| Metric | Expected | Target |
|--------|----------|--------|
| Surface Dice @ 2mm | 0.65-0.72 | >0.65 |
| VOI (split+merge) | 3.5-4.5 | Lower is better |
| TopoScore | 0.75-0.85 | >0.7 |
| Precision | 0.70-0.80 | >0.65 |
| Recall | 0.70-0.80 | >0.65 |

---

## 📞 Questions?

- **How to run training?** → See `QUICK_START.md` or `src/vesuvius/train.py`
- **How to run inference?** → See `run_cloud_validation.ps1` or `QUICK_START.md`
- **Why is inference failing?** → See `TECHNICAL_ROOT_CAUSE_ANALYSIS.md`
- **What are next steps?** → See `VALIDATION_PIPELINE_PLAN.md`
- **How to submit to Kaggle?** → See `kaggle_notebook_template.py`


