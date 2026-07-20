# Rebuild the whole PlantVillage matrix on the leaf-grouped split.
#
# The previous results used a random split that leaked same-leaf duplicates into
# the test set. This retrains everything from the baselines up. ResNet-18 runs
# first because every later stage depends on it, so a broken split surfaces in
# the first arm rather than after hours.
#
#   .\rebuild.ps1            full rebuild
#   .\rebuild.ps1 -WhatIf    print the plan and exit

param([switch]$WhatIf)
$ErrorActionPreference = "Continue"

$baselines = @("resnet18", "custom_cnn", "mobilenet_v2")

Write-Host "`n=== rebuild plan ===" -ForegroundColor Cyan
Write-Host "  baselines (128px): $($baselines -join ', ')"
Write-Host "  ensemble weight search"
Write-Host "  full matrix via run_all (E1-E18)"
if ($WhatIf) { exit 0 }

$started = Get-Date
$failed = @()

foreach ($m in $baselines) {
    Write-Host "`n=== baseline $m ===" -ForegroundColor Cyan
    uv run python -m scripts.train --model $m
    if ($LASTEXITCODE -ne 0) { $failed += "train_$m" }
}

Write-Host "`n=== ensemble ===" -ForegroundColor Cyan
uv run python -m scripts.ensemble
if ($LASTEXITCODE -ne 0) { $failed += "ensemble" }

Write-Host "`n=== full matrix ===" -ForegroundColor Cyan
uv run python -m scripts.run_all
if ($LASTEXITCODE -ne 0) { $failed += "run_all" }

Write-Host "`n=== done in $([int]((Get-Date) - $started).TotalMinutes) min ===" -ForegroundColor Cyan
if ($failed.Count) { Write-Host "failed: $($failed -join ', ')" -ForegroundColor Red }
uv run python -m scripts.status
