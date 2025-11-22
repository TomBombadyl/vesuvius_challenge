# Quick Start – Validation & Submission

## Current Status
✅ **Training Complete** (32 epochs, 20.4% loss reduction)  
⏳ **Next:** Run validation → threshold sweep → Kaggle submission

---

## Phase 1: Validation Inference (1-2 hours)

**Start inference on cloud A100 GPU:**
```powershell
.\run_cloud_validation.ps1
```

This generates 806 prediction `.tif` files.

**Monitor progress:**
```powershell
gcloud compute ssh dylant@vesuvius-challenge --project=vesuvius-challenge-478512 --zone=us-central1-a --command="ls /mnt/disks/data/repos/vesuvius_challenge/runs/exp001_3d_unet_topology_full/infer_val/*.tif 2>/dev/null | wc -l"
```

---

## Phase 2: Evaluate Results (5-10 minutes)

**After inference completes, evaluate predictions:**
```bash
python -m src.vesuvius.evaluate \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --predictions runs/exp001_3d_unet_topology_full/infer_val \
  --split train \
  --output-csv runs/exp001_3d_unet_topology_full/evaluation.csv
```

Expected output:
- Surface Dice: ~0.68–0.72
- VOI: ~3.5–4.5
- TopoScore: ~0.75–0.85

---

## Phase 3: Threshold Sweep (10-30 minutes)

**Find optimal threshold:**
```bash
for THRESH in 0.25 0.30 0.35 0.40 0.45 0.50; do
  python -m src.vesuvius.evaluate \
    --config configs/experiments/exp001_3d_unet_topology.yaml \
    --predictions runs/exp001_3d_unet_topology_full/infer_val \
    --split train \
    --threshold $THRESH \
    --output-csv runs/exp001_3d_unet_topology_full/eval_thresh_${THRESH}.csv
done
```

Find threshold with highest Surface Dice (typically 0.38–0.46)

---

## Phase 4: Kaggle Submission (1-2 hours on Kaggle)

1. Download checkpoint from VM
2. Use `kaggle_notebook_template.py` as template
3. Run in Kaggle notebook with hidden test data
4. Submit `submission.zip`

---

## Key Files

| File | Purpose |
|------|---------|
| `AGENT_EXECUTION_SUMMARY.md` | Full detailed guide (read if stuck) |
| `REMOTE_VALIDATION_RUNNER.md` | Cloud inference details |
| `TRAINING_RESULTS_COMPREHENSIVE.md` | Training metrics analysis |
| `run_cloud_validation.ps1` | Automation script |
| `kaggle_notebook_template.py` | Kaggle submission template |

---

## Troubleshooting

**Inference not starting?**
- Verify SSH: `gcloud compute ssh dylant@vesuvius-challenge --project=vesuvius-challenge-478512 --zone=us-central1-a --command="echo test"`
- Check GPU: `gcloud compute ssh dylant@vesuvius-challenge --project=vesuvius-challenge-478512 --zone=us-central1-a --command="nvidia-smi"`

**Evaluation failing?**
- Check predictions exist: `ls runs/exp001_3d_unet_topology_full/infer_val/*.tif | wc -l` (should be 806)
- Verify ground truth: `ls vesuvius_kaggle_data/train_labels/*.tif | wc -l` (should be 806)

---

## Timeline
- **Phase 1:** 1-2 hours
- **Phase 2:** 5-10 minutes
- **Phase 3:** 10-30 minutes
- **Phase 4:** 1-2 hours on Kaggle
- **Total:** ~3-5 hours (excluding Kaggle)

---

**Status:** Ready to begin! 🚀

