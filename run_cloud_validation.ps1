# Cloud VM Validation Runner for PowerShell
# Runs inference on GCP A100 VM

Write-Host "================================================================="
Write-Host "  Running Validation Inference on Cloud VM (A100 GPU)"
Write-Host "================================================================="
Write-Host ""

# Configuration
$PROJECT = "vesuvius-challenge-478512"
$ZONE = "us-central1-a"
$INSTANCE = "dylant@vesuvius-challenge"
$REPO_PATH = "/mnt/disks/data/repos/vesuvius_challenge"

# Build the command to run on VM - use semicolons for bash, not backticks
$VM_COMMAND = "cd $REPO_PATH; source .venv/bin/activate; export PYTHONPATH=$REPO_PATH; python -m src.vesuvius.infer --config configs/experiments/exp001_3d_unet_topology.yaml --checkpoint runs/exp001_3d_unet_topology_full/checkpoints/last.pt --output-dir runs/exp001_3d_unet_topology_full/infer_val --split train --device cuda"

Write-Host "Starting SSH connection to cloud VM..."
Write-Host "Project: $PROJECT"
Write-Host "Zone: $ZONE"
Write-Host "Instance: $INSTANCE"
Write-Host ""
Write-Host "Running inference on A100 GPU..."
Write-Host "Expected duration: 1-2 hours"
Write-Host ""
Write-Host "================================================================="
Write-Host ""

# Execute the command
gcloud compute ssh $INSTANCE `
  --project=$PROJECT `
  --zone=$ZONE `
  --command=$VM_COMMAND

Write-Host ""
Write-Host "================================================================="
Write-Host "Inference job submitted to cloud VM"
Write-Host "Check progress with:"
Write-Host "  gcloud compute ssh $INSTANCE --project=$PROJECT --zone=$ZONE --command='tail -f $REPO_PATH/runs/exp001_3d_unet_topology_full/infer_val/infer.log'"
Write-Host "================================================================="
Write-Host ""
