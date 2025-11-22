# Quick Start – Validation & Submission

## Current Status
✅ **Training Complete** (32 epochs, 20.4% loss reduction)  
✅ **Model Validated** (Phase 1: 10 volumes, 0.324 mean IoU)  
⏳ **Next:** External validation → Full inference → Kaggle submission

---

## Option A: External Validation on Your Dataset (Optional)

**If you have external paired volumes (image.tif + mask.tif):**

```bash
# On VM or locally
python -m src.vesuvius.validate_external \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --checkpoint checkpoints/last_exp001.pt \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --output-dir runs/external_validation \
  --max-volumes 10
```

**Output:**
- CSV with per-volume metrics (Dice, IoU, Precision, Recall) across thresholds 0.30-0.55
- Summary statistics and best threshold recommendation
- Spot-check for generalization

**Expected Results:**
- Dice > 0.65 ✅ Model generalizes well
- Dice 0.55-0.65 ⚠️ Acceptable, may need threshold tuning
- Dice < 0.55 ❌ Domain mismatch detected

---

## Phase 1: Full Inference (6-7 hours)

**Start inference on cloud A100 GPU:**
```powershell
.\run_cloud_validation.ps1
```

Generates 806 predictions on training dataset.

**Monitor progress:**
```bash
gcloud compute ssh dylant@vesuvius-challenge --zone=us-central1-a \
  --command="tail -20 runs/phase1_quick_test/infer.log"
```

---

## Phase 2: Evaluate Results (30 minutes)

**Compute metrics:**
```bash
python -m src.vesuvius.evaluate \
  --predictions runs/predictions \
  --output runs/evaluation_results.csv
```

Expected:
- Surface Dice: 0.68–0.72
- VOI: 3.5–4.5

---

## Phase 3: Threshold Sweep (20 minutes)

**Find optimal threshold:**
```bash
python optimize_threshold.py \
  --predictions runs/predictions \
  --output runs/threshold_analysis.csv
```

Result: Optimal threshold typically 0.38–0.46

---

## Phase 4: Kaggle Submission (1-2 hours on Kaggle)

1. Download checkpoint & predictions from VM
2. Use `kaggle_notebook_template.py` as starting template
3. Run in Kaggle notebook environment
4. Submit `submission.zip`

---

## Timeline

| Task | Time |
|------|------|
| Optional: External Validation | 2-3 hours |
| Phase 1: Full Inference | 6-7 hours |
| Phase 2: Metrics | 30 min |
| Phase 3: Threshold Sweep | 20 min |
| Phase 4: Kaggle Submission | 1-2 hours |
| **Total** | **8-14 hours** |

---

## Essential Files

| File | Purpose |
|------|---------|
| `run_cloud_validation.ps1` | Phase 1 execution |
| `download_to_gcs.ps1` | Upload external data |
| `validate_external_gcs.py` | External validation |
| `kaggle_notebook_template.py` | Kaggle template |
| `START_HERE.md` | Project overview |
| `TECHNICAL_BREAKDOWN.md` | Architecture details |

---

## Troubleshooting

**Inference fails?**
```bash
gcloud compute ssh dylant@vesuvius-challenge --zone=us-central1-a --command="nvidia-smi"
```

**External validation Dice low?**
- Check image normalization
- Try different threshold: `--threshold 0.50`
- Verify checkpoint loads correctly

---

**Ready?** Execute `.\run_cloud_validation.ps1` 🚀
