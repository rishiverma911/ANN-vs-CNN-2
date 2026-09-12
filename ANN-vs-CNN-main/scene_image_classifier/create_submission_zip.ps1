# Package the assignment deliverable for upload.
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ZipPath = Join-Path (Split-Path $ProjectRoot -Parent) "ANN_vs_CNN_Submission.zip"

$items = @(
    "notebooks/ann_vs_cnn_experiment.ipynb",
    "README.md",
    "SUBMISSION.md",
    "requirements.txt",
    "model_architectures",
    "project_utils",
    "results/metrics",
    "results/figures",
    "results/models/class_names.json"
)

Push-Location $ProjectRoot
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path $items -DestinationPath $ZipPath -Force
Pop-Location

Write-Host "Created submission archive:" -ForegroundColor Green
Write-Host "  $ZipPath"
Write-Host ""
Write-Host "Primary upload file: notebooks/ann_vs_cnn_experiment.ipynb" -ForegroundColor Cyan
