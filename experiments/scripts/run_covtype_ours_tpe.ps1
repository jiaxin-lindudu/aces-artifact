# Adaptive Ours tuning; preserve the paused full batch. ASCII / PowerShell 5.1.
param(
    [string]$PythonPath = (Get-Command python -CommandType Application).Source,
    [int[]]$DataSeeds = @(0, 1, 2, 3, 4),
    [int]$MaxNewTrials = 8,
    [int]$Patience = 3,
    [double]$MinImprovement = 0.002,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectPath
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
foreach ($dataSeed in $DataSeeds) {
    $sourceDir = Join-Path $projectPath "experiments/results/covtype_10000/federated_${dataSeed}_reference"
    $outDir = Join-Path $projectPath "experiments/results/covtype_10000/federated_${dataSeed}_ours_tpe"
    $options = @('--dataset', 'covtype_10000', '--grid-split', "$dataSeed", '--clients', '20', '--beta', '0.5',
        '--init-seeds', '40,41,42,43,44', '--select-on', 'acc', '--eigen-solver', 'dense',
        '--max-new-trials', "$MaxNewTrials", '--patience', "$Patience", '--min-improvement', "$MinImprovement",
        '--data-dir', "$projectPath/data", '--source-dir', $sourceDir, '--out-dir', $outDir, '--resume')
    if ($DryRun) { $options += '--dry-run' }
    Write-Host "`nCovtype Ours adaptive search: data_seed=$dataSeed"
    & $PythonPath "$projectPath/experiments/scripts/tune_covtype_ours.py" @options
    if ($LASTEXITCODE -ne 0) { throw "Covtype Ours adaptive search failed: data_seed=$dataSeed" }
}
Write-Host "`nRequested adaptive Ours studies finished."
