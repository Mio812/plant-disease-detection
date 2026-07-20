# Runs every remaining arm sequentially. Sequential is deliberate: the OOM in
# train_frozen_bg_random came from three concurrent jobs spawning 30 Windows
# worker processes, not from any one arm being too large.
#
#   .\push.ps1            run everything still missing
#   .\push.ps1 -WhatIf    print the plan and exit

param([switch]$WhatIf)

$ErrorActionPreference = "Continue"

# Ordered so the results that can still change a conclusion arrive first.
$plan = @(
    @{ id = "train_hier";             why = "E17 - the only remaining lever on zero-shot" },
    @{ id = "finetune_aug_only";      why = "E11 control - the arm the E11 claim rests on" },
    @{ id = "train_frozen_bg_random"; why = "E15 - completes the 2x2" },
    @{ id = "finetune_baseline";      why = "E11 comparison point" },
    @{ id = "eval_arm_bg_control";    why = "E16 breakdown" },
    @{ id = "eval_arm_bg_random";     why = "E16 breakdown" },
    @{ id = "eval_arm_frozen_ctrl";   why = "E16 breakdown" },
    @{ id = "eval_arm_frozen_bg";     why = "E16 breakdown" },
    @{ id = "eval_arm_hier";          why = "E16 breakdown" }
)

Write-Host "`n=== plan ===" -ForegroundColor Cyan
foreach ($s in $plan) { Write-Host ("  {0,-24} {1}" -f $s.id, $s.why) }
if ($WhatIf) { exit 0 }

$started = Get-Date
$failed = @()
foreach ($s in $plan) {
    Write-Host "`n=== $($s.id) ===" -ForegroundColor Cyan
    uv run python -m scripts.run_all --only $s.id
    if ($LASTEXITCODE -ne 0) { $failed += $s.id }
}

Write-Host "`n=== done in $([int]((Get-Date) - $started).TotalMinutes) min ===" -ForegroundColor Cyan
if ($failed.Count) { Write-Host "failed: $($failed -join ', ')" -ForegroundColor Red }
uv run python -m scripts.status
Get-Content outputs\results_summary.md
