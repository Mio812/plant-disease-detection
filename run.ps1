<#
.SYNOPSIS
    One-command driver for the whole COMP9444 experiment pipeline.

.DESCRIPTION
    Checks the environment, emits the severity annotation sheet first (so the
    team can grade while the GPU works), optionally runs a 2-epoch smoke test,
    then runs the full resumable pipeline and writes outputs\results_summary.md.
    Safe to re-run: finished stages are skipped.

.EXAMPLE
    .\run.ps1                 # full pipeline (smoke test first)
    .\run.ps1 -Quick          # smoke test only
    .\run.ps1 -SkipSmoke      # straight to the full run
    .\run.ps1 -FullAblation   # also p=1.0 and ResNet-50
#>
[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$SkipSmoke,
    [switch]$FullAblation,
    [switch]$SkipSync
)

Set-Location -Path $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$script:Log = "logs\run_$stamp.log"

function Say([string]$msg, [string]$colour = "Cyan") {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $line -ForegroundColor $colour
    Add-Content -Path $script:Log -Value $line
}

function Invoke-Step([string]$label, [string[]]$cmd) {
    Say "--> $label"
    & uv @cmd 2>&1 | Tee-Object -FilePath $script:Log -Append
    if ($LASTEXITCODE -ne 0) {
        Say "FAILED: $label (exit $LASTEXITCODE)" "Red"
        return $false
    }
    return $true
}

Say "COMP9444 pipeline | log -> $script:Log"

# ---------------------------------------------------------------- environment
if (-not $SkipSync) {
    if (-not (Invoke-Step "sync environment (CUDA 13.0 wheels)" @("sync", "--extra", "cu130"))) {
        Say "Environment sync failed. Fix this before continuing." "Red"; exit 1
    }
}

& uv run python -c "import torch,sys; ok=torch.cuda.is_available(); print(torch.__version__, ok, torch.cuda.get_device_name(0) if ok else 'NO GPU'); sys.exit(0 if ok else 1)" 2>&1 |
    Tee-Object -FilePath $script:Log -Append
if ($LASTEXITCODE -ne 0) {
    Say "CUDA not available. RTX 50-series needs recent CUDA wheels (uv sync --extra cu130)." "Red"
    exit 1
}

# --------------------------------------------- severity sheet (unblocks team)
if (-not (Test-Path "outputs\severity_annotations.csv")) {
    [void](Invoke-Step "sample leaves for severity grading" @("run", "python", "-m", "scripts.severity_sample", "--n", "150"))
    Say "Grade 'manual_grade' (0-3) in outputs\severity_annotations.csv while training runs." "Yellow"
}

# ------------------------------------------------------------------ smoke run
if (-not $SkipSmoke) {
    if (-not (Invoke-Step "smoke test (2 epochs, isolated in outputs_quick\)" @("run", "python", "-m", "scripts.run_all", "--quick"))) {
        Say "Smoke test failed - stopping before the long run." "Red"; exit 1
    }
    Say "Smoke test passed." "Green"
}
if ($Quick) { Say "Quick mode: stopping after the smoke test." "Green"; exit 0 }

# ------------------------------------------------------------------ full run
$full = @("run", "python", "-m", "scripts.run_all")
if ($FullAblation) { $full += "--full-ablation" }
[void](Invoke-Step "full pipeline (resumable)" $full)

# ------------------------------------------------- severity validation if graded
$graded = & uv run python -c "import csv,sys,os; p='outputs/severity_annotations.csv'; r=list(csv.DictReader(open(p,encoding='utf-8'))) if os.path.exists(p) else []; print(sum(1 for x in r if x.get('manual_grade','').strip()))" 2>$null
if ([int]$graded -gt 0) {
    Say "$graded rows graded - validating severity"
    [void](Invoke-Step "severity validation" @("run", "python", "-m", "scripts.severity_validate", "--csv", "outputs/severity_annotations.csv"))
    [void](Invoke-Step "refresh summary" @("run", "python", "-m", "scripts.run_all"))
} else {
    Say "No severity grades yet. After grading, run:" "Yellow"
    Say "  uv run python -m scripts.severity_validate --csv outputs/severity_annotations.csv" "Yellow"
    Say "  uv run python -m scripts.run_all    # refreshes the summary" "Yellow"
}

# --------------------------------------------------------------------- report
Say "DONE" "Green"
if (Test-Path "outputs\results_summary.md") {
    Write-Host ""
    Get-Content "outputs\results_summary.md" | Write-Host
    Write-Host ""
    Say "Send outputs\results_summary.md (full log: $script:Log)" "Green"
}
