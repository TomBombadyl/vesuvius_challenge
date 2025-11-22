# Full External Validation - In Progress 🚀

**Start Time:** November 22, 2025, ~18:15 UTC  
**Status:** ⏳ RUNNING ON VM  
**Volumes:** 1,755 (all external validation data)  
**Expected Duration:** ~1.5 hours  
**Expected Completion:** ~19:45 UTC

---

## 📊 What We're Running

```bash
python3 -m src.vesuvius.validate_external \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --checkpoint runs/exp001_3d_unet_topology_full/checkpoints/last.pt \
  --image-dir /tmp/external_validation/dl.ash2txt.org/datasets/seg-derived-recto-surfaces/imagesTr \
  --mask-dir /tmp/external_validation/dl.ash2txt.org/datasets/seg-derived-recto-surfaces/labelsTr \
  --output-dir runs/external_validation_full
```

**Configuration:**
- Model: ResidualUNet3D (10.8M parameters)
- Checkpoint: `last.pt` from full training
- Volumes: 1,755 external samples (300×300×300 each)
- Thresholds: 0.30 to 0.55 (25 steps)
- Metrics: Dice, IoU, Precision, Recall per threshold

---

## 🎯 What We Expect

### From Previous 5-Volume Test
- Mean Dice: 0.411
- Best Dice: 0.463
- Worst Dice: 0.380
- Optimal Threshold: 0.48

### From Full 1,755-Volume Run
We expect:
- ✅ Comprehensive mean Dice (with confidence intervals)
- ✅ Full distribution (mean, std, min, max)
- ✅ Per-threshold statistics
- ✅ Rock-solid generalization metrics
- ✅ Professional publication-ready results

---

## 📈 Performance Timeline

| Milestone | Expected | Description |
|-----------|----------|-------------|
| 0-15 min | ⏳ Now | Loading model, starting inference |
| 15-75 min | ⏳ Running | Processing volumes 1-1,000 (~13/min) |
| 75-90 min | ⏳ Running | Processing volumes 1,000-1,755 |
| 90 min | ✅ Done | All inference complete |
| 90-95 min | ✅ Done | Computing metrics & saving results |

**Total: ~1.5 hours**

---

## 📝 Output Files

When complete, will produce:
```
runs/external_validation_full/
├── external_validation_results.csv
│   └── 43,875 rows (1,755 volumes × 25 thresholds)
│   └── Columns: volume_id, threshold, dice, iou, precision, recall, tp, fp, fn
│
├── validate_external.log
│   └── Detailed execution log with per-volume timing
│
└── Summary statistics (printed to log)
```

**CSV Size:** ~15-20 MB

---

## 🔍 How to Monitor Progress

### Check Log File (Every 5-10 minutes)

```bash
# SSH into VM
gcloud compute ssh dylant@vesuvius-challenge --zone=us-central1-a

# Tail the log
tail -100 /mnt/disks/data/repos/vesuvius_challenge/runs/external_validation_full.log
```

### Watch Throughput

```bash
# Should see: [N/1755] Processing volume_name...
# Speed: ~13-15 volumes/minute
```

### GPU Status

```bash
# Check GPU memory usage
nvidia-smi
```

---

## 📊 What This Gives Us

### Rock-Solid v1.0 Story

✅ **Trained on:** Kaggle training data only (806 volumes)  
✅ **Validated on:** Fully independent external data (1,755 volumes)  
✅ **Result:** Mean Dice = X.XXX ± Y.YYY  
✅ **Generalization:** ✅ VERIFIED at scale

### Publication-Ready Metrics

- Complete distribution: mean, std, min, max, percentiles
- Per-threshold analysis
- Confidence intervals
- Professional graphs

### For Kaggle Submission

- Defensible statement: "Validated on 1,755 independent volumes"
- Confidence in predictions
- Data-driven decision on domain adaptation

---

## 🎯 Success Criteria

✅ **If mean Dice ≥ 0.40** → Model generalizes well, ready to submit  
⚠️ **If mean Dice 0.35-0.40** → Good generalization, may benefit from fine-tuning  
❌ **If mean Dice < 0.35** → Significant domain gap, fine-tuning recommended

---

## 🚀 Next Steps After Completion

1. **Download Results** (take 10 minutes)
   ```bash
   gcloud compute scp --zone=us-central1-a \
     --recurse \
     dylant@vesuvius-challenge:/mnt/disks/data/repos/vesuvius_challenge/runs/external_validation_full \
     ./external_validation_full_results/
   ```

2. **Analyze Results** (30 minutes)
   - Load CSV
   - Generate summary statistics
   - Create plots
   - Document findings

3. **Update v1.0 Release** (30 minutes)
   - Add full validation metrics to RELEASE_V1_0.md
   - Update EXTERNAL_VALIDATION_RESULTS.md
   - Document generalization confidence

4. **Decide Next Phase** (15 minutes)
   - If Dice ≥ 0.40 → Ready for Kaggle submission
   - If Dice 0.35-0.40 → Consider domain adaptation (v1.1)
   - If Dice < 0.35 → Plan investigation & fixes

---

## 📞 Monitoring

**Start monitoring when ready:**
- Check log file every 5-10 minutes
- Expected completion: ~19:45 UTC
- Will update this document with results

**Status:** ⏳ Running...

---

## 🎉 Expected Outcome

When this completes in ~90 minutes, you'll have:

✅ **Comprehensive validation metrics** on 1,755 independent volumes  
✅ **Professional-grade statistics** ready for publication  
✅ **Confidence in model generalization** backed by data  
✅ **Clear decision path** for next steps (submit vs. fine-tune)  
✅ **Defensible v1.0 story** for Kaggle & beyond

---

**Status:** ⏳ IN PROGRESS  
**Last Updated:** November 22, 2025, 18:15 UTC  
**Check logs regularly for progress updates**


